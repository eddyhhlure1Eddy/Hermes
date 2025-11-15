"""Build cached preprocessed stock data for fast training.

Author: eddy
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from financial_model.src.stock_metadata import IndustryClassifier, StyleFactorCalculator
from financial_model.src.market_regime import MarketRegimeDetector
from financial_model.src.multi_dim_dataset import gpu_normalize


class CacheDatabaseBuilder:
    """Build .cache files and a small SQLite index for stock data."""

    def __init__(
        self,
        data_dir: str = "full_stock_data/training_data",
        cache_dir: str = "data_cache",
        db_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.db_path = db_path or os.path.join(cache_dir, "meta.sqlite")
        self.device = device

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stocks (
                    code TEXT PRIMARY KEY,
                    cache_path TEXT NOT NULL,
                    num_rows INTEGER NOT NULL,
                    industry_idx INTEGER NOT NULL,
                    style_idx INTEGER NOT NULL,
                    start_date TEXT,
                    end_date TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def iter_stock_codes(self, max_stocks: Optional[int] = None) -> List[str]:
        files = [f for f in os.listdir(self.data_dir) if f.endswith(".csv")]
        files.sort()
        codes: List[str] = [os.path.splitext(f)[0] for f in files]
        if max_stocks is not None:
            codes = codes[:max_stocks]
        return codes

    def build_for_stock(self, code: str) -> None:
        csv_path = os.path.join(self.data_dir, f"{code}.csv")
        if not os.path.exists(csv_path):
            print(f"Skipping {code}, file not found: {csv_path}")
            return

        print(f"Building cache for {code}...")
        df = pd.read_csv(csv_path)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

        features = ["open", "high", "low", "close", "volume"]
        target_col = "close"

        for col in features + [target_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' is missing in {csv_path}")

        feature_data = df[features].values.astype(np.float32)
        target_data = df[target_col].values.astype(np.float32)

        norm_features, feature_mean, feature_std = gpu_normalize(
            feature_data, device=self.device
        )

        target_reshaped = target_data.reshape(-1, 1)
        norm_target, target_mean, target_std = gpu_normalize(
            target_reshaped, device=self.device
        )
        norm_target = norm_target.reshape(-1)

        industry_l1, industry_l2 = IndustryClassifier.get_industry(code)
        industry_encoding = IndustryClassifier.get_industry_encoding(
            industry_l1, industry_l2
        )
        industry_idx = int(industry_encoding["industry_l1_idx"])

        if "close" in df.columns and len(df) > 20:
            returns = df["close"].pct_change()
            volatility_20d = float(returns.rolling(20).std().iloc[-1])
            momentum_20d = float(df["close"].iloc[-1] / df["close"].iloc[-20] - 1)
        else:
            volatility_20d = 0.02
            momentum_20d = 0.0

        vol_factor = StyleFactorCalculator.calculate_volatility_factor(volatility_20d)
        vol_map = {"low_vol": 0, "medium_vol": 1, "high_vol": 2}
        style_idx = int(vol_map.get(vol_factor, 1))

        if "close" in df.columns:
            detector = MarketRegimeDetector()
            regime_series = detector.detect_regime_series(df["close"])
        else:
            regime_series = pd.Series(["sideways"] * len(df), index=df.index)

        regime_indices = np.array(
            [MarketRegimeDetector.get_regime_encoding(r) for r in regime_series],
            dtype=np.int64,
        )

        cache = {
            "code": code,
            "features": norm_features,
            "target": norm_target,
            "regime": regime_indices,
            "industry_idx": industry_idx,
            "style_idx": style_idx,
            "feature_mean": feature_mean.cpu().numpy(),
            "feature_std": feature_std.cpu().numpy(),
            "target_mean": target_mean.cpu().numpy(),
            "target_std": target_std.cpu().numpy(),
        }

        cache_path = os.path.join(self.cache_dir, f"{code}.cache")
        torch.save(cache, cache_path)

        start_date = str(df["date"].iloc[0]) if "date" in df.columns else ""
        end_date = str(df["date"].iloc[-1]) if "date" in df.columns else ""

        self._upsert_metadata(
            code=code,
            cache_path=cache_path,
            num_rows=len(df),
            industry_idx=industry_idx,
            style_idx=style_idx,
            start_date=start_date,
            end_date=end_date,
        )

        print(f"Saved cache: {cache_path}")

    def _upsert_metadata(
        self,
        code: str,
        cache_path: str,
        num_rows: int,
        industry_idx: int,
        style_idx: int,
        start_date: str,
        end_date: str,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO stocks (code, cache_path, num_rows, industry_idx, style_idx, start_date, end_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(code) DO UPDATE SET
                    cache_path=excluded.cache_path,
                    num_rows=excluded.num_rows,
                    industry_idx=excluded.industry_idx,
                    style_idx=excluded.style_idx,
                    start_date=excluded.start_date,
                    end_date=excluded.end_date
                """,
                (code, cache_path, num_rows, industry_idx, style_idx, start_date, end_date),
            )
            conn.commit()
        finally:
            conn.close()


def main() -> None:
    import time

    max_stocks: Optional[int] = None
    use_gpu: bool = False

    if len(sys.argv) > 1:
        try:
            max_stocks = int(sys.argv[1])
        except ValueError:
            max_stocks = None

    if len(sys.argv) > 2 and sys.argv[2].lower() in ["gpu", "cuda"]:
        use_gpu = True

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    if use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but CUDA not available, falling back to CPU")

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    builder = CacheDatabaseBuilder(device=device)
    codes = builder.iter_stock_codes(max_stocks=max_stocks or 2)

    print(f"Building cache for {len(codes)} stocks...")

    start_time = time.time()

    for idx, code in enumerate(codes, 1):
        stock_start = time.time()
        builder.build_for_stock(code)
        stock_elapsed = time.time() - stock_start
        print(f"  [{idx}/{len(codes)}] {code} completed in {stock_elapsed:.2f}s")

    total_elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Total: {len(codes)} stocks processed in {total_elapsed:.2f}s")
    print(f"Average: {total_elapsed/len(codes):.2f}s per stock")
    print(f"Device: {device}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

