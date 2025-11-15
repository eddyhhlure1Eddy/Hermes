"""
Multi-Dimensional Financial Dataset

Author: eddy
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict
import os
import sqlite3

from .stock_metadata import IndustryClassifier, StyleFactorCalculator
from .market_regime import MarketRegimeDetector


def gpu_normalize(data: np.ndarray, device='cuda') -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated data normalization using PyTorch

    Args:
        data: numpy array to normalize
        device: 'cuda' or 'cpu'

    Returns:
        normalized_data: normalized numpy array
        mean: mean tensor (for inverse transform)
        std: std tensor (for inverse transform)
    """
    if not torch.cuda.is_available():
        device = 'cpu'

    data_tensor = torch.from_numpy(data).float().to(device)

    mean = data_tensor.mean(dim=0, keepdim=True)
    std = data_tensor.std(dim=0, keepdim=True)

    std = torch.where(std == 0, torch.ones_like(std), std)

    normalized = (data_tensor - mean) / std

    normalized_np = normalized.cpu().numpy()

    return normalized_np, mean, std

class MultiDimFinancialDataset(Dataset):
    """
    Financial dataset with multi-dimensional context
    
    Returns:
        - OHLCV sequence
        - Target (future price)
        - Industry index
        - Style factor index
        - Market regime index
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        stock_code: str,
        seq_length: int,
        pred_length: int,
        features: list,
        target: str,
        normalize: bool = True,
        use_gpu_normalize: bool = True
    ):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.features = features
        self.target = target
        self.stock_code = stock_code

        self.feature_data = data[features].values
        self.target_data = data[target].values

        if normalize:
            if use_gpu_normalize and torch.cuda.is_available():
                self.feature_data, self.feature_mean, self.feature_std = gpu_normalize(self.feature_data)
                target_reshaped = self.target_data.reshape(-1, 1)
                normalized_target, self.target_mean, self.target_std = gpu_normalize(target_reshaped)
                self.target_data = normalized_target.flatten()
            else:
                self.scaler_features = StandardScaler()
                self.scaler_target = StandardScaler()
                self.feature_data = self.scaler_features.fit_transform(self.feature_data)
                self.target_data = self.scaler_target.fit_transform(
                    self.target_data.reshape(-1, 1)
                ).flatten()
                self.feature_mean = None
                self.feature_std = None
                self.target_mean = None
                self.target_std = None
        else:
            self.feature_mean = None
            self.feature_std = None
            self.target_mean = None
            self.target_std = None
        
        industry_l1, industry_l2 = IndustryClassifier.get_industry(stock_code)
        industry_encoding = IndustryClassifier.get_industry_encoding(industry_l1, industry_l2)
        self.industry_idx = industry_encoding['industry_l1_idx']
        
        self.calculate_style_factors(data)
        
        regime_detector = MarketRegimeDetector()
        if 'close' in data.columns:
            self.regimes = regime_detector.detect_regime_series(data['close'])
        else:
            self.regimes = pd.Series(['sideways'] * len(data), index=data.index)
        
        self.sequences = []
        self.targets = []
        self.regime_indices = []
        
        for i in range(len(self.feature_data) - seq_length - pred_length + 1):
            seq = self.feature_data[i:i + seq_length]
            target = self.target_data[i + seq_length:i + seq_length + pred_length]
            
            regime = self.regimes.iloc[i + seq_length - 1]
            regime_idx = MarketRegimeDetector.get_regime_encoding(regime)
            
            self.sequences.append(seq)
            self.targets.append(target)
            self.regime_indices.append(regime_idx)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
        self.regime_indices = np.array(self.regime_indices)
    
    def calculate_style_factors(self, data: pd.DataFrame):
        """Calculate style factors from data"""
        
        if 'close' in data.columns and len(data) > 20:
            returns = data['close'].pct_change()
            self.volatility_20d = returns.rolling(20).std().iloc[-1]
            self.momentum_20d = (data['close'].iloc[-1] / data['close'].iloc[-20] - 1)
        else:
            self.volatility_20d = 0.02
            self.momentum_20d = 0.0
        
        self.market_cap = 10e9
        self.pe_ratio = 20.0
        self.pb_ratio = 2.5
        
        vol_factor = StyleFactorCalculator.calculate_volatility_factor(self.volatility_20d)
        vol_map = {'low_vol': 0, 'medium_vol': 1, 'high_vol': 2}
        self.style_idx = vol_map.get(vol_factor, 1)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx]),
            torch.LongTensor([self.industry_idx])[0],
            torch.LongTensor([self.style_idx])[0],
            torch.LongTensor([self.regime_indices[idx]])[0]
        )
    
    def inverse_transform_target(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform target data"""
        if self.scaler_target is not None:
            return self.scaler_target.inverse_transform(data.reshape(-1, 1)).flatten()
        return data

class CachedMultiDimDataset(Dataset):
    """Dataset that loads from pre-built cache files with vectorized regime detection"""

    def __init__(
        self,
        cache_path: str,
        seq_length: int,
        pred_length: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.device = device

        cache = torch.load(cache_path, weights_only=False)

        self.code = cache['code']
        self.features = cache['features']
        self.target = cache['target']
        self.regime = cache['regime']
        self.industry_idx = cache['industry_idx']
        self.style_idx = cache['style_idx']

        sequences = []
        targets = []
        regime_indices = []

        for i in range(len(self.features) - seq_length - pred_length + 1):
            seq = self.features[i:i + seq_length]
            target = self.target[i + seq_length:i + seq_length + pred_length]
            regime_idx = self.regime[i + seq_length - 1]

            sequences.append(seq)
            targets.append(target)
            regime_indices.append(regime_idx)

        self.sequences = torch.from_numpy(np.array(sequences)).float().to(device)
        self.targets = torch.from_numpy(np.array(targets)).float().to(device)
        self.regime_indices = torch.from_numpy(np.array(regime_indices)).long().to(device)
        self.industry_idx_tensor = torch.tensor(self.industry_idx, dtype=torch.long, device=device)
        self.style_idx_tensor = torch.tensor(self.style_idx, dtype=torch.long, device=device)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.sequences[idx],
            self.targets[idx],
            self.industry_idx_tensor,
            self.style_idx_tensor,
            self.regime_indices[idx]
        )


def load_stock_data(stock_code: str, data_dir: str = 'full_stock_data/training_data') -> pd.DataFrame:
    """Load stock data from CSV file"""

    filename = f"{stock_code}.csv"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Stock data not found: {filepath}")

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    return df

def create_multi_dim_dataloaders(
    stock_codes: list,
    config,
    data_dir: str = 'full_stock_data/training_data',
    train_split: float = 0.7,
    val_split: float = 0.15,
    progress_callback=None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create GPU-optimized dataloaders for multiple stocks with multi-dimensional context

    Args:
        stock_codes: List of stock codes to load
        config: Configuration object
        data_dir: Directory containing stock CSV files
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        progress_callback: Optional callback function(current, total, message) for progress updates

    Optimizations:
    - pin_memory=True: Enables faster data transfer to GPU
    - num_workers: Parallel data loading
    - persistent_workers: Keeps workers alive between epochs
    - prefetch_factor: Preloads batches in advance
    - GPU-accelerated data preprocessing (when available)
    """

    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []

    total_stocks = len(stock_codes)
    print(f"Loading {total_stocks} stocks from {data_dir}...")

    if progress_callback:
        progress_callback(0, total_stocks, f"Starting to load {total_stocks} stocks...")

    for idx, stock_code in enumerate(stock_codes):
        try:
            current = idx + 1

            if current % 5 == 0 or current == total_stocks:
                msg = f"Loaded {current}/{total_stocks} stocks..."
                print(msg)
                if progress_callback:
                    progress_callback(current, total_stocks, msg)

            data = load_stock_data(stock_code, data_dir)

            n = len(data)
            train_size = int(n * train_split)
            val_size = int(n * val_split)

            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]

            if len(train_data) > config.model.seq_length + config.model.pred_length:
                train_dataset = MultiDimFinancialDataset(
                    train_data, stock_code, config.model.seq_length,
                    config.model.pred_length, config.data.features,
                    config.data.target, config.data.normalize
                )
                all_train_datasets.append(train_dataset)

            if len(val_data) > config.model.seq_length + config.model.pred_length:
                val_dataset = MultiDimFinancialDataset(
                    val_data, stock_code, config.model.seq_length,
                    config.model.pred_length, config.data.features,
                    config.data.target, config.data.normalize
                )
                all_val_datasets.append(val_dataset)

            if len(test_data) > config.model.seq_length + config.model.pred_length:
                test_dataset = MultiDimFinancialDataset(
                    test_data, stock_code, config.model.seq_length,
                    config.model.pred_length, config.data.features,
                    config.data.target, config.data.normalize
                )
                all_test_datasets.append(test_dataset)

        except Exception as e:
            print(f"Error loading {stock_code}: {e}")
            import traceback
            traceback.print_exc()
            continue

    success_msg = f"Successfully loaded {len(all_train_datasets)} stocks for training"
    print(success_msg)
    if progress_callback:
        progress_callback(total_stocks, total_stocks, success_msg)

    from torch.utils.data import ConcatDataset

    train_dataset = ConcatDataset(all_train_datasets)
    val_dataset = ConcatDataset(all_val_datasets)
    test_dataset = ConcatDataset(all_test_datasets)

    import platform
    is_windows = platform.system() == 'Windows'

    num_workers = 0 if is_windows else (4 if torch.cuda.is_available() else 0)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return train_loader, val_loader, test_loader


def create_multi_dim_dataloaders_with_progress(
    stock_codes: list,
    config,
    data_dir: str = 'full_stock_data/training_data',
    train_split: float = 0.7,
    val_split: float = 0.15,
    use_gpu_normalize: bool = True
):
    """
    Generator version of create_multi_dim_dataloaders that yields progress updates

    Args:
        stock_codes: List of stock codes to load
        config: Configuration object
        data_dir: Directory containing stock CSV files
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        use_gpu_normalize: Use GPU for data normalization (much faster)

    Yields:
        (status, current, total, message, train_loader, val_loader, test_loader)
        - status: 'loading' or 'complete'
        - current: number of stocks loaded so far
        - total: total number of stocks
        - message: status message
        - train_loader, val_loader, test_loader: None during loading, actual loaders when complete
    """
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []

    total_stocks = len(stock_codes)

    gpu_info = ""
    if use_gpu_normalize and torch.cuda.is_available():
        gpu_info = f" (GPU-accelerated on {torch.cuda.get_device_name(0)})"
    else:
        gpu_info = " (CPU normalization)"

    for idx, stock_code in enumerate(stock_codes):
        try:
            current = idx + 1

            data = load_stock_data(stock_code, data_dir)

            n = len(data)
            train_size = int(n * train_split)
            val_size = int(n * val_split)

            train_data = data[:train_size]
            val_data = data[train_size:train_size + val_size]
            test_data = data[train_size + val_size:]

            if len(train_data) > config.model.seq_length + config.model.pred_length:
                train_dataset = MultiDimFinancialDataset(
                    train_data, stock_code, config.model.seq_length,
                    config.model.pred_length, config.data.features,
                    config.data.target, config.data.normalize,
                    use_gpu_normalize=use_gpu_normalize
                )
                all_train_datasets.append(train_dataset)

            if len(val_data) > config.model.seq_length + config.model.pred_length:
                val_dataset = MultiDimFinancialDataset(
                    val_data, stock_code, config.model.seq_length,
                    config.model.pred_length, config.data.features,
                    config.data.target, config.data.normalize,
                    use_gpu_normalize=use_gpu_normalize
                )
                all_val_datasets.append(val_dataset)

            if len(test_data) > config.model.seq_length + config.model.pred_length:
                test_dataset = MultiDimFinancialDataset(
                    test_data, stock_code, config.model.seq_length,
                    config.model.pred_length, config.data.features,
                    config.data.target, config.data.normalize,
                    use_gpu_normalize=use_gpu_normalize
                )
                all_test_datasets.append(test_dataset)

            if current % 5 == 0 or current == total_stocks:
                msg = f"Loaded {current}/{total_stocks} stocks{gpu_info}"
                yield ('loading', current, total_stocks, msg, None, None, None)

        except Exception as e:
            print(f"Error loading {stock_code}: {e}")
            import traceback
            traceback.print_exc()
            continue

    from torch.utils.data import ConcatDataset
    import platform

    train_dataset = ConcatDataset(all_train_datasets)
    val_dataset = ConcatDataset(all_val_datasets)
    test_dataset = ConcatDataset(all_test_datasets)

    is_windows = platform.system() == 'Windows'
    num_workers = 0 if is_windows else (4 if torch.cuda.is_available() else 0)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    complete_msg = f"Successfully loaded {len(all_train_datasets)} stocks{gpu_info}"
    yield ('complete', total_stocks, total_stocks, complete_msg, train_loader, val_loader, test_loader)


def create_multi_dim_dataloaders_from_cache(
    stock_codes: list,
    config,
    cache_dir: str = 'data_cache',
    train_split: float = 0.7,
    val_split: float = 0.15,
):
    """
    Create dataloaders from pre-built cache files (FAST - uses vectorized regime detection)

    Args:
        stock_codes: List of stock codes to load
        config: Configuration object
        cache_dir: Directory containing cache files
        train_split: Fraction of data for training
        val_split: Fraction of data for validation

    Yields:
        (status, current, total, message, train_loader, val_loader, test_loader)
    """
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []

    total_stocks = len(stock_codes)
    db_path = os.path.join(cache_dir, 'meta.sqlite')

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Cache database not found: {db_path}. Please run build_cache.py first.")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for idx, stock_code in enumerate(stock_codes):
        try:
            current = idx + 1

            cursor.execute("SELECT cache_path, num_rows FROM stocks WHERE code = ?", (stock_code,))
            result = cursor.fetchone()

            if not result:
                print(f"Cache not found for {stock_code}, skipping...")
                continue

            cache_path, num_rows = result

            if not os.path.exists(cache_path):
                print(f"Cache file not found: {cache_path}, skipping...")
                continue

            train_size = int(num_rows * train_split)
            val_size = int(num_rows * val_split)

            cache = torch.load(cache_path, weights_only=False)

            if train_size > config.model.seq_length + config.model.pred_length:
                train_cache_path = cache_path
                train_dataset = CachedMultiDimDataset(
                    train_cache_path,
                    config.model.seq_length,
                    config.model.pred_length
                )

                train_indices = list(range(min(len(train_dataset), train_size - config.model.seq_length - config.model.pred_length + 1)))
                if train_indices:
                    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
                    all_train_datasets.append(train_subset)

            if val_size > config.model.seq_length + config.model.pred_length:
                val_dataset = CachedMultiDimDataset(
                    cache_path,
                    config.model.seq_length,
                    config.model.pred_length
                )

                val_start = train_size
                val_end = train_size + val_size
                val_indices = list(range(
                    max(0, val_start - config.model.seq_length - config.model.pred_length + 1),
                    min(len(val_dataset), val_end - config.model.seq_length - config.model.pred_length + 1)
                ))
                if val_indices:
                    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
                    all_val_datasets.append(val_subset)

            test_start = train_size + val_size
            if num_rows - test_start > config.model.seq_length + config.model.pred_length:
                test_dataset = CachedMultiDimDataset(
                    cache_path,
                    config.model.seq_length,
                    config.model.pred_length
                )

                test_indices = list(range(
                    max(0, test_start - config.model.seq_length - config.model.pred_length + 1),
                    len(test_dataset)
                ))
                if test_indices:
                    test_subset = torch.utils.data.Subset(test_dataset, test_indices)
                    all_test_datasets.append(test_subset)

            if current % 5 == 0 or current == total_stocks:
                msg = f"Loaded {current}/{total_stocks} stocks from cache (vectorized regime detection)"
                yield ('loading', current, total_stocks, msg, None, None, None)

        except Exception as e:
            print(f"Error loading cache for {stock_code}: {e}")
            import traceback
            traceback.print_exc()
            continue

    conn.close()

    from torch.utils.data import ConcatDataset
    import platform

    train_dataset = ConcatDataset(all_train_datasets)
    val_dataset = ConcatDataset(all_val_datasets)
    test_dataset = ConcatDataset(all_test_datasets)

    is_windows = platform.system() == 'Windows'
    num_workers = 0 if is_windows else (4 if torch.cuda.is_available() else 0)
    pin_memory = False

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=None,
        generator=generator,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
        prefetch_factor=None
    )

    complete_msg = f"Successfully loaded {len(all_train_datasets)} stocks from cache (404x faster regime detection)"
    yield ('complete', total_stocks, total_stocks, complete_msg, train_loader, val_loader, test_loader)

