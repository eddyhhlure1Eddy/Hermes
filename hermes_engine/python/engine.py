"""
HermesEngine — Python wrapper for the CUDA inference engine.

Usage:
    eng = HermesEngine.from_checkpoint("ckpt.pt", device="cuda:0")
    out = eng.forward(x, industry, style, regime, cg_state, risk_regime, events=events)
    # out: dict with keys 'price', 'cg_state', 'cg_score', 'risk_regime',
    #                     'signal', 'direction'
"""

from __future__ import annotations

import os
from typing import Optional, Dict, List, Tuple, Any

import torch

from . import _C   # compiled extension


def _build_weight_keys(num_layers: int = 4) -> List[str]:
    keys: List[str] = [
        "input_projection.weight",          # 0
        "input_projection.bias",            # 1
        "positional_encoding.pe",           # 2
        "event_projection.0.weight",        # 3
        "event_projection.0.bias",          # 4
        "event_gate.gate_proj.weight",      # 5
        "event_gate.gate_proj.bias",        # 6
    ]
    for l in range(num_layers):
        p = f"transformer_encoder.layers.{l}"
        keys.extend([
            f"{p}.norm1.weight",            f"{p}.norm1.bias",
            f"{p}.self_attn_qkv.weight",    f"{p}.self_attn_qkv.bias",
            f"{p}.self_attn_out.weight",    f"{p}.self_attn_out.bias",
            f"{p}.norm2.weight",            f"{p}.norm2.bias",
            f"{p}.linear1.weight",          f"{p}.linear1.bias",
            f"{p}.linear2.weight",          f"{p}.linear2.bias",
        ])
    keys.extend([
        "industry_embedding.weight",
        "style_embedding.weight",
        "regime_embedding.weight",
        "cg_state_embedding.weight",
        "risk_regime_embedding.weight",
        "fusion_layer.0.weight", "fusion_layer.0.bias",
        "fusion_layer.3.weight", "fusion_layer.3.bias",
        "prediction_head.0.weight", "prediction_head.0.bias",
        "prediction_head.3.weight", "prediction_head.3.bias",
        "state_head.0.weight",      "state_head.0.bias",
        "state_head.3.weight",      "state_head.3.bias",
        "score_head.0.weight",      "score_head.0.bias",
        "score_head.3.weight",      "score_head.3.bias",
        "risk_head.0.weight",       "risk_head.0.bias",
        "risk_head.3.weight",       "risk_head.3.bias",
        "signal_head.0.weight",     "signal_head.0.bias",
        "signal_head.3.weight",     "signal_head.3.bias",
        "direction_head.0.weight",  "direction_head.0.bias",
        "direction_head.3.weight",  "direction_head.3.bias",
    ])
    return keys


WEIGHT_KEYS = _build_weight_keys(num_layers=4)


def _resolve_state_dict(ckpt_path: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        for k in ("model_state_dict", "state_dict", "model"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                meta = {kk: vv for kk, vv in obj.items() if kk != k}
                return sd, meta
        return obj, {}
    raise ValueError(f"unsupported checkpoint format: {type(obj)}")


def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not sd:
        return sd
    pref_candidates = ["module.", "model.", "_orig_mod."]
    for p in pref_candidates:
        if all(k.startswith(p) for k in sd):
            return {k[len(p):]: v for k, v in sd.items()}
    return sd


class HermesEngine:
    def __init__(
        self,
        weights: List[torch.Tensor],
        device: torch.device,
        num_layers: int,
        has_event_branch: bool,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self.weights = weights
        self.device = device
        self.num_layers = num_layers
        self.has_event_branch = has_event_branch
        self.meta = meta or {}

    # ------------------------------------------------------------------ load
    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str,
        device: str | torch.device = "cuda:0",
        num_layers: int = 4,
    ) -> "HermesEngine":
        sd, meta = _resolve_state_dict(ckpt_path)
        sd = _strip_prefix(sd)
        return cls.from_state_dict(sd, device=device, num_layers=num_layers, meta=meta)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        device: str | torch.device = "cuda:0",
        num_layers: int = 4,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "HermesEngine":
        device = torch.device(device)
        keys = _build_weight_keys(num_layers=num_layers)
        has_event = "event_projection.0.weight" in state_dict and "event_gate.gate_proj.weight" in state_dict

        D = state_dict["input_projection.weight"].shape[0]

        def _stub(shape) -> torch.Tensor:
            return torch.zeros(*shape, dtype=torch.bfloat16, device=device)

        weights: List[torch.Tensor] = []
        for k in keys:
            if k in state_dict:
                t = state_dict[k].detach().to(device=device, dtype=torch.bfloat16).contiguous()
                weights.append(t)
            else:
                # only allowed missing entries are the event branch when disabled
                if k == "event_projection.0.weight":
                    weights.append(_stub((D, 1)))
                elif k == "event_projection.0.bias":
                    weights.append(_stub((D,)))
                elif k == "event_gate.gate_proj.weight":
                    weights.append(_stub((D, 2 * D)))
                elif k == "event_gate.gate_proj.bias":
                    weights.append(_stub((D,)))
                else:
                    raise KeyError(f"checkpoint missing required weight: {k}")
        return cls(weights=weights, device=device, num_layers=num_layers,
                   has_event_branch=has_event, meta=meta)

    # ------------------------------------------------------------------ forward
    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,                       # (B, T, 29) any float dtype
        industry_idx: torch.Tensor,            # (B,) int64
        style_idx: torch.Tensor,
        regime_idx: torch.Tensor,
        cg_state_idx: Optional[torch.Tensor] = None,
        risk_regime_idx: Optional[torch.Tensor] = None,
        events: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"x must be 3D (B, T, in_dim), got shape {tuple(x.shape)}")
        B = x.size(0)
        dev = self.device

        def _check_idx(name: str, t: torch.Tensor) -> None:
            if t.dim() != 1 or t.size(0) != B:
                raise ValueError(f"{name} must be 1D with length {B}, got shape {tuple(t.shape)}")

        def _to(t: Optional[torch.Tensor], dtype=None) -> Optional[torch.Tensor]:
            if t is None:
                return None
            if dtype is None:
                return t.to(device=dev, non_blocking=True).contiguous()
            return t.to(device=dev, dtype=dtype, non_blocking=True).contiguous()

        for name, t in (("industry_idx", industry_idx),
                        ("style_idx", style_idx),
                        ("regime_idx", regime_idx)):
            _check_idx(name, t)
        if cg_state_idx is not None:    _check_idx("cg_state_idx", cg_state_idx)
        if risk_regime_idx is not None: _check_idx("risk_regime_idx", risk_regime_idx)

        x = _to(x, torch.bfloat16)
        ind = _to(industry_idx, torch.int64)
        sty = _to(style_idx,    torch.int64)
        reg = _to(regime_idx,   torch.int64)
        cgs = _to(cg_state_idx, torch.int64) if cg_state_idx is not None else \
              torch.zeros(B, dtype=torch.int64, device=dev)
        rsk = _to(risk_regime_idx, torch.int64) if risk_regime_idx is not None else \
              torch.zeros(B, dtype=torch.int64, device=dev)
        if events is not None and not self.has_event_branch:
            raise ValueError("events provided but engine has no event branch (checkpoint has no event_projection)")
        if events is not None and events.dim() != 3:
            raise ValueError(f"events must be 3D (B, T, num_events), got shape {tuple(events.shape)}")
        ev = _to(events, torch.bfloat16) if (events is not None and self.has_event_branch) else None

        out_list = _C.hermes_forward(
            x=x,
            industry_idx=ind,
            style_idx=sty,
            regime_idx=reg,
            cg_state_idx=cgs,
            risk_regime_idx=rsk,
            events=ev,
            texts=None,
            weights=self.weights,
            has_event_branch=bool(self.has_event_branch),
            num_layers=int(self.num_layers),
        )
        # Order matches the C++ side: [price, cg_state, cg_score, risk_regime, signal, direction]
        return {
            "price":       out_list[0],
            "cg_state":    out_list[1],
            "cg_score":    out_list[2],
            "risk_regime": out_list[3],
            "signal":      out_list[4],
            "direction":   out_list[5],
        }

    # ------------------------------------------------------------------ misc
    def state_dict_keys_used(self) -> List[str]:
        return list(WEIGHT_KEYS)

    def num_parameters(self) -> int:
        return sum(t.numel() for t in self.weights)
