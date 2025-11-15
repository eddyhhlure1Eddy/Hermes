"""
Conditional Model Inference

Author: eddy
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Optional, Tuple

from .config import Config
from .conditional_model import create_conditional_model
from .stock_metadata import IndustryClassifier, StyleFactorCalculator
from .market_regime import MarketRegimeDetector
from sklearn.preprocessing import StandardScaler


class ConditionalPredictor:
    """Predictor for conditional multi-dimensional model"""
    
    def __init__(self, checkpoint_path: str, config: Optional[Config] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if config is None:
            self.config = checkpoint.get('config', Config())
        else:
            self.config = config

        state_dict = checkpoint['model_state_dict']

        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected torch.compile checkpoint, removing _orig_mod. prefix")
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

        uses_flash_attn = any('self_attn_qkv' in key for key in state_dict.keys())

        if uses_flash_attn and not self.config.model.use_flash_attention:
            print("Checkpoint uses Flash Attention, enabling it in config")
            self.config.model.use_flash_attention = True
        elif not uses_flash_attn and self.config.model.use_flash_attention:
            print("Checkpoint uses standard attention, disabling Flash Attention in config")
            self.config.model.use_flash_attention = False

        self.model = create_conditional_model(self.config)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"Conditional model loaded from {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Using Flash Attention: {self.config.model.use_flash_attention}")
    
    def predict_stock(
        self,
        stock_code: str,
        recent_data: pd.DataFrame,
        future_steps: int = 5,
        features: Optional[list] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Predict future prices for a single stock
        
        Args:
            stock_code: Stock code (e.g., 'SH600000')
            recent_data: Recent OHLCV data (at least seq_length rows)
            future_steps: Number of future steps to predict
            features: Feature columns to use
        
        Returns:
            predictions: Array of future predictions
            metadata: Dict with industry, style, regime info
        """
        if features is None:
            features = self.config.data.features
        
        if len(recent_data) < self.config.model.seq_length:
            raise ValueError(
                f"Need at least {self.config.model.seq_length} rows, got {len(recent_data)}"
            )
        
        recent_data = recent_data.tail(self.config.model.seq_length).copy()
        
        industry_l1, industry_l2 = IndustryClassifier.get_industry(stock_code)
        industry_encoding = IndustryClassifier.get_industry_encoding(industry_l1, industry_l2)
        industry_idx = industry_encoding['industry_l1_idx']
        
        if len(recent_data) > 20:
            returns = recent_data['close'].pct_change()
            volatility_20d = returns.rolling(20).std().iloc[-1]
        else:
            volatility_20d = 0.02

        vol_factor = StyleFactorCalculator.calculate_volatility_factor(volatility_20d)
        vol_map = {'low_vol': 0, 'medium_vol': 1, 'high_vol': 2}
        style_idx = vol_map.get(vol_factor, 1)
        
        regime_detector = MarketRegimeDetector()
        current_regime = regime_detector.detect_regime(recent_data['close'])
        regime_idx = MarketRegimeDetector.get_regime_encoding(current_regime)
        
        feature_data = recent_data[features].values
        scaler = StandardScaler()
        feature_data_normalized = scaler.fit_transform(feature_data)
        
        predictions = []
        current_sequence = feature_data_normalized.copy()
        
        for step in range(future_steps):
            seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
            industry_tensor = torch.LongTensor([industry_idx]).to(self.device)
            style_tensor = torch.LongTensor([style_idx]).to(self.device)
            regime_tensor = torch.LongTensor([regime_idx]).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                    pred = self.model(seq_tensor, industry_tensor, style_tensor, regime_tensor)
            
            pred_value = pred.cpu().numpy()[0, 0]
            predictions.append(pred_value)
            
            last_features = current_sequence[-1].copy()
            last_features[features.index('close')] = pred_value
            
            current_sequence = np.vstack([current_sequence[1:], last_features])
        
        predictions = np.array(predictions)
        
        target_scaler = StandardScaler()
        target_scaler.fit(recent_data['close'].values.reshape(-1, 1))
        predictions_denorm = target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        metadata = {
            'stock_code': stock_code,
            'industry_l1': industry_l1,
            'industry_l2': industry_l2,
            'volatility': volatility_20d,
            'style': ['low_vol', 'medium_vol', 'high_vol'][style_idx],
            'regime': current_regime,
            'last_price': recent_data['close'].iloc[-1]
        }
        
        return predictions_denorm, metadata
    
    def predict_next_day(
        self,
        stock_code: str,
        recent_data: pd.DataFrame,
        features: Optional[list] = None
    ) -> Tuple[float, dict]:
        """
        Predict next day price for a single stock
        
        Args:
            stock_code: Stock code
            recent_data: Recent OHLCV data
            features: Feature columns
        
        Returns:
            next_price: Predicted next day price
            metadata: Stock metadata
        """
        predictions, metadata = self.predict_stock(
            stock_code, recent_data, future_steps=1, features=features
        )
        
        return predictions[0], metadata

