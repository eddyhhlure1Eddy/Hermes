import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        batch_first: bool = True,
        use_flash_attn: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.batch_first = batch_first
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        
        self.self_attn_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.self_attn_out = nn.Linear(d_model, d_model, bias=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.batch_first:
            src = src.transpose(0, 1)
        
        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
        x = x + self._ff_block(self.norm2(x))
        
        if not self.batch_first:
            x = x.transpose(0, 1)
        
        return x
    
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.self_attn_qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.d_model // self.nhead)

        if self.use_flash_attn and x.is_cuda and x.dtype in [torch.float16, torch.bfloat16]:
            qkv = qkv.permute(0, 2, 1, 3, 4)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=None,
                causal=False
            )
        else:
            q, k, v = qkv.permute(2, 0, 3, 1, 4)

            q = q.reshape(batch_size * self.nhead, seq_len, self.d_model // self.nhead)
            k = k.reshape(batch_size * self.nhead, seq_len, self.d_model // self.nhead)
            v = v.reshape(batch_size * self.nhead, seq_len, self.d_model // self.nhead)

            scale = 1.0 / (self.d_model // self.nhead) ** 0.5
            attn_weights = torch.bmm(q, k.transpose(1, 2)) * scale
            attn_weights = F.softmax(attn_weights, dim=-1)

            if self.training:
                attn_weights = self.dropout(attn_weights)

            attn_output = torch.bmm(attn_weights, v)

            attn_output = attn_output.reshape(batch_size, self.nhead, seq_len, self.d_model // self.nhead)
        
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
        attn_output = self.self_attn_out(attn_output)
        
        return self.dropout1(attn_output)
    
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FlashAttentionTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: FlashAttentionTransformerEncoderLayer,
        num_layers: int
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            FlashAttentionTransformerEncoderLayer(
                d_model=encoder_layer.d_model,
                nhead=encoder_layer.nhead,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout.p,
                batch_first=encoder_layer.batch_first,
                use_flash_attn=encoder_layer.use_flash_attn
            )
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers
    
    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        output = src
        
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        
        return output

