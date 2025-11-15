"""
Conditional Financial Model with Multi-Dimensional Context

Author: eddy
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict

from .model import PositionalEncoding
from .flash_attention_layer import (
    FlashAttentionTransformerEncoderLayer,
    FlashAttentionTransformerEncoder,
    FLASH_ATTN_AVAILABLE
)

class ConditionalFinancialModel(nn.Module):
    """
    Conditional model with shared backbone and context-aware prediction
    
    Architecture:
    1. Shared Transformer Backbone - extracts universal technical patterns with attention
    2. Conditional Input - industry, style factors, market regime
    3. Fusion Layer - combines backbone features with conditional context
    4. Prediction Head - outputs final prediction
    """
    
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
        use_flash_attention: bool = True,
        industry_embed_dim: int = 128,
        style_embed_dim: int = 64,
        regime_embed_dim: int = 64
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Transformer backbone for sequence modeling
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim, dropout=dropout)

        self.use_flash_attention = use_flash_attention and FLASH_ATTN_AVAILABLE

        if self.use_flash_attention:
            print(f"Using Flash Attention 2 for ConditionalFinancialModel")
            encoder_layer = FlashAttentionTransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                use_flash_attn=True
            )
            self.transformer_encoder = FlashAttentionTransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )
        else:
            print(f"Flash Attention 2 not available, using standard PyTorch attention")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

        # Embeddings for conditional inputs
        self.industry_embedding = nn.Embedding(num_industries, industry_embed_dim)
        self.style_embedding = nn.Embedding(num_style_factors, style_embed_dim)
        self.regime_embedding = nn.Embedding(num_market_regimes, regime_embed_dim)

        conditional_dim = industry_embed_dim + style_embed_dim + regime_embed_dim

        # Fusion layer combining sequence features and conditional context
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim + conditional_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, pred_length),
        )

    def forward(
        self,
        x: torch.Tensor,
        industry_idx: torch.Tensor,
        style_idx: torch.Tensor,
        regime_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_dim) - OHLCV data
            industry_idx: (batch,) - industry category index
            style_idx: (batch,) - style factor index
            regime_idx: (batch,) - market regime index
        
        Returns:
            output: (batch, pred_length) - predictions
        """
        
        batch_size = x.size(0)

        # Sequence encoding with Transformer backbone
        x_proj = self.input_projection(x)
        x_proj = x_proj.transpose(0, 1)
        x_proj = self.positional_encoding(x_proj)
        x_proj = x_proj.transpose(0, 1)

        transformer_out = self.transformer_encoder(x_proj)
        backbone_features = transformer_out[:, -1, :]

        # Conditional context embeddings
        industry_emb = self.industry_embedding(industry_idx)
        style_emb = self.style_embedding(style_idx)
        regime_emb = self.regime_embedding(regime_idx)

        conditional_features = torch.cat([industry_emb, style_emb, regime_emb], dim=1)

        # Fuse sequence features with conditional context
        fused_features = torch.cat([backbone_features, conditional_features], dim=1)
        fused_features = self.fusion_layer(fused_features)

        # Final prediction
        output = self.prediction_head(fused_features)

        return output

def create_conditional_model(config) -> ConditionalFinancialModel:
    """Create conditional Transformer-based conditional model from config"""

    model = ConditionalFinancialModel(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        pred_length=config.model.pred_length,
        use_flash_attention=config.model.use_flash_attention,
        industry_embed_dim=config.model.industry_embed_dim,
        style_embed_dim=config.model.style_embed_dim,
        regime_embed_dim=config.model.regime_embed_dim,
    )

    return model.to(config.model.device)

