"""
Market Regime Detection

Author: eddy
"""

import pandas as pd
import numpy as np
from typing import Literal

MarketRegime = Literal['bull', 'bear', 'sideways', 'volatile']

class MarketRegimeDetector:
    """Detect market regime (bull/bear/sideways)"""
    
    def __init__(self, ma_short: int = 20, ma_long: int = 60):
        self.ma_short = ma_short
        self.ma_long = ma_long
    
    def detect_regime(self, prices: pd.Series) -> MarketRegime:
        """Detect current market regime"""
        
        if len(prices) < self.ma_long:
            return 'sideways'
        
        ma_short = prices.rolling(window=self.ma_short).mean()
        ma_long = prices.rolling(window=self.ma_long).mean()
        
        current_price = prices.iloc[-1]
        current_ma_short = ma_short.iloc[-1]
        current_ma_long = ma_long.iloc[-1]
        
        volatility = prices.pct_change().rolling(window=20).std().iloc[-1]
        
        trend_strength = (current_ma_short - current_ma_long) / current_ma_long
        
        if volatility > 0.03:
            return 'volatile'
        
        if current_price > current_ma_short > current_ma_long:
            if trend_strength > 0.05:
                return 'bull'
            else:
                return 'sideways'
        
        elif current_price < current_ma_short < current_ma_long:
            if trend_strength < -0.05:
                return 'bear'
            else:
                return 'sideways'
        
        else:
            return 'sideways'
    
    def detect_regime_series(self, prices: pd.Series) -> pd.Series:
        """Detect regime for entire time series (vectorized version)"""

        n = len(prices)
        regimes = np.full(n, 'sideways', dtype=object)

        if n < self.ma_long:
            return pd.Series(regimes, index=prices.index)

        ma_short = prices.rolling(window=self.ma_short).mean()
        ma_long = prices.rolling(window=self.ma_long).mean()

        returns = prices.pct_change()
        volatility = returns.rolling(window=20).std()

        trend_strength = (ma_short - ma_long) / ma_long

        bull_mask = (prices > ma_short) & (ma_short > ma_long) & (trend_strength > 0.05) & (volatility <= 0.03)
        bear_mask = (prices < ma_short) & (ma_short < ma_long) & (trend_strength < -0.05) & (volatility <= 0.03)
        volatile_mask = volatility > 0.03

        regimes[bull_mask] = 'bull'
        regimes[bear_mask] = 'bear'
        regimes[volatile_mask] = 'volatile'

        regimes[:self.ma_long] = 'sideways'

        return pd.Series(regimes, index=prices.index)

    def detect_regime_series_old(self, prices: pd.Series) -> pd.Series:
        """Detect regime for entire time series (old loop version)"""

        regimes = []

        for i in range(len(prices)):
            if i < self.ma_long:
                regimes.append('sideways')
            else:
                window_prices = prices.iloc[max(0, i - 100):i + 1]
                regime = self.detect_regime(window_prices)
                regimes.append(regime)

        return pd.Series(regimes, index=prices.index)
    
    @staticmethod
    def get_regime_encoding(regime: MarketRegime) -> int:
        """Get numeric encoding for regime"""
        regime_map = {
            'bull': 0,
            'bear': 1,
            'sideways': 2,
            'volatile': 3
        }
        return regime_map.get(regime, 2)

class IndexDataProvider:
    """Provide index data for market regime detection"""
    
    @staticmethod
    def get_index_data(index_code: str = '000001') -> pd.DataFrame:
        """Get index data (placeholder - should connect to real data source)"""
        
        dates = pd.date_range(start='2020-01-01', end='2025-11-14', freq='D')
        
        np.random.seed(42)
        price = 3000
        prices = []
        
        for i in range(len(dates)):
            if i < 200:
                change = np.random.randn() * 0.5 + 0.05
            elif i < 400:
                change = np.random.randn() * 0.5 - 0.05
            elif i < 600:
                change = np.random.randn() * 0.3
            elif i < 800:
                change = np.random.randn() * 1.0
            else:
                change = np.random.randn() * 0.5 + 0.03
            
            price = price * (1 + change / 100)
            prices.append(price)
        
        return pd.DataFrame({
            'date': dates,
            'close': prices
        })
    
    @staticmethod
    def get_current_regime() -> MarketRegime:
        """Get current market regime"""
        detector = MarketRegimeDetector()
        index_data = IndexDataProvider.get_index_data()
        regime = detector.detect_regime(index_data['close'])
        return regime

