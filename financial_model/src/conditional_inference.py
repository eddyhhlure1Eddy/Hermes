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

        target_scaler = StandardScaler()
        target_scaler.fit(recent_data['close'].values.reshape(-1, 1))

        predictions = []
        predictions_denorm = []
        predicted_ohlcv = []
        current_sequence = feature_data_normalized.copy()
        last_close = recent_data['close'].iloc[-1]

        avg_volume = recent_data['volume'].tail(20).mean()
        recent_returns = recent_data['close'].pct_change().tail(20)
        avg_volatility = recent_returns.std()

        for step in range(future_steps):
            seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
            industry_tensor = torch.LongTensor([industry_idx]).to(self.device)
            style_tensor = torch.LongTensor([style_idx]).to(self.device)
            regime_tensor = torch.LongTensor([regime_idx]).to(self.device)

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.device.type == 'cuda'):
                    pred = self.model(seq_tensor, industry_tensor, style_tensor, regime_tensor)

            pred_value_norm = pred.cpu().numpy()[0, 0]
            predictions.append(pred_value_norm)

            pred_value_denorm = target_scaler.inverse_transform([[pred_value_norm]])[0, 0]
            predictions_denorm.append(pred_value_denorm)

            prev_close = current_sequence[-1, features.index('close')]
            prev_close_denorm = target_scaler.inverse_transform([[prev_close]])[0, 0]

            price_change_pct = (pred_value_denorm - prev_close_denorm) / prev_close_denorm

            daily_volatility = avg_volatility if step == 0 else avg_volatility * (1 + 0.1 * step)
            high_low_range = abs(pred_value_denorm * daily_volatility * 2)

            if price_change_pct > 0:
                next_open = prev_close_denorm * (1 + price_change_pct * 0.3)
                next_close = pred_value_denorm
                next_high = max(next_open, next_close) * (1 + daily_volatility * 0.5)
                next_low = min(next_open, next_close) * (1 - daily_volatility * 0.3)
            else:
                next_open = prev_close_denorm * (1 + price_change_pct * 0.3)
                next_close = pred_value_denorm
                next_high = max(next_open, next_close) * (1 + daily_volatility * 0.3)
                next_low = min(next_open, next_close) * (1 - daily_volatility * 0.5)

            volume_change_factor = 1 + abs(price_change_pct) * 2
            volume_random_factor = np.random.uniform(0.8, 1.2)
            next_volume = avg_volume * volume_change_factor * volume_random_factor

            predicted_ohlcv.append({
                'open': next_open,
                'high': next_high,
                'low': next_low,
                'close': next_close,
                'volume': next_volume
            })

            next_features_raw = np.array([next_open, next_high, next_low, next_close, next_volume]).reshape(1, -1)
            next_features_norm = scaler.transform(next_features_raw)[0]

            current_sequence = np.vstack([current_sequence[1:], next_features_norm])

        predictions_denorm = np.array(predictions_denorm)
        predicted_ohlcv_df = pd.DataFrame(predicted_ohlcv)
        
        metadata = {
            'stock_code': stock_code,
            'industry_l1': industry_l1,
            'industry_l2': industry_l2,
            'volatility': volatility_20d,
            'style': ['low_vol', 'medium_vol', 'high_vol'][style_idx],
            'regime': current_regime,
            'last_price': recent_data['close'].iloc[-1],
            'predicted_ohlcv': predicted_ohlcv_df
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

