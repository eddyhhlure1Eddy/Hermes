"""Triton-optimized conditional model with torch.compile acceleration.

Author: eddy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, falling back to PyTorch operations")


@triton.jit
def fused_embedding_kernel(
    industry_ptr, style_ptr, regime_ptr,
    industry_emb_ptr, style_emb_ptr, regime_emb_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    industry_dim: tl.constexpr,
    style_dim: tl.constexpr,
    regime_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused embedding lookup and concatenation kernel"""
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    industry_idx = tl.load(industry_ptr + pid)
    style_idx = tl.load(style_ptr + pid)
    regime_idx = tl.load(regime_ptr + pid)
    
    output_offset = pid * (industry_dim + style_dim + regime_dim)
    
    for i in range(industry_dim):
        val = tl.load(industry_emb_ptr + industry_idx * industry_dim + i)
        tl.store(output_ptr + output_offset + i, val)
    
    for i in range(style_dim):
        val = tl.load(style_emb_ptr + style_idx * style_dim + i)
        tl.store(output_ptr + output_offset + industry_dim + i, val)
    
    for i in range(regime_dim):
        val = tl.load(regime_emb_ptr + regime_idx * regime_dim + i)
        tl.store(output_ptr + output_offset + industry_dim + style_dim + i, val)


class FusedEmbedding(nn.Module):
    """Fused embedding layer using Triton for faster conditional feature extraction"""
    
    def __init__(self, num_industries: int, num_styles: int, num_regimes: int):
        super().__init__()
        self.industry_embedding = nn.Embedding(num_industries, 32)
        self.style_embedding = nn.Embedding(num_styles, 16)
        self.regime_embedding = nn.Embedding(num_regimes, 16)
        self.use_triton = TRITON_AVAILABLE
    
    def forward(self, industry_idx: torch.Tensor, style_idx: torch.Tensor, regime_idx: torch.Tensor) -> torch.Tensor:
        if self.use_triton and industry_idx.is_cuda:
            return self._triton_forward(industry_idx, style_idx, regime_idx)
        else:
            return self._pytorch_forward(industry_idx, style_idx, regime_idx)
    
    def _triton_forward(self, industry_idx: torch.Tensor, style_idx: torch.Tensor, regime_idx: torch.Tensor) -> torch.Tensor:
        batch_size = industry_idx.size(0)
        output = torch.empty(batch_size, 64, device=industry_idx.device, dtype=torch.float32)
        
        grid = lambda meta: (batch_size,)
        fused_embedding_kernel[grid](
            industry_idx, style_idx, regime_idx,
            self.industry_embedding.weight, self.style_embedding.weight, self.regime_embedding.weight,
            output,
            batch_size, 32, 16, 16,
            BLOCK_SIZE=256
        )
        return output
    
    def _pytorch_forward(self, industry_idx: torch.Tensor, style_idx: torch.Tensor, regime_idx: torch.Tensor) -> torch.Tensor:
        industry_emb = self.industry_embedding(industry_idx)
        style_emb = self.style_embedding(style_idx)
        regime_emb = self.regime_embedding(regime_idx)
        return torch.cat([industry_emb, style_emb, regime_emb], dim=1)


class OptimizedConditionalModel(nn.Module):
    """Optimized conditional model with torch.compile and Triton kernels"""
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        num_industries: int = 11,
        num_style_factors: int = 3,
        num_market_regimes: int = 4,
        dropout: float = 0.1,
        pred_length: int = 1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fused_embedding = FusedEmbedding(num_industries, num_style_factors, num_market_regimes)
        
        conditional_dim = 64
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + conditional_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        
        self.prediction_head = nn.Linear(hidden_dim, pred_length)
    
    def forward(
        self,
        x: torch.Tensor,
        industry_idx: torch.Tensor,
        style_idx: torch.Tensor,
        regime_idx: torch.Tensor
    ) -> torch.Tensor:
        x_proj = self.input_projection(x)
        transformer_out = self.transformer_encoder(x_proj)
        backbone_features = transformer_out[:, -1, :]
        
        conditional_features = self.fused_embedding(industry_idx, style_idx, regime_idx)
        
        fused_features = torch.cat([backbone_features, conditional_features], dim=1)
        fused_features = self.fusion_layer(fused_features)
        
        output = self.prediction_head(fused_features)
        return output

