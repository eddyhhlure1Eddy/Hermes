"""
Gradio Training and Inference Panel for Financial Model
Bilingual (Chinese/English) Interface

Author: eddy
"""

import gradio as gr
import torch
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import threading
import time
import json
from datetime import datetime
from pytdx.hq import TdxHq_API

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'financial_model'))

from src.config import Config
from src.model import create_model
from src.dataset import create_sample_data, create_dataloaders
from src.train import Trainer
from src.inference import Predictor
from src.conditional_inference import ConditionalPredictor
from src.multi_dim_dataset import create_multi_dim_dataloaders, create_multi_dim_dataloaders_with_progress, create_multi_dim_dataloaders_from_cache
import sqlite3
from src.stock_metadata import IndustryClassifier
from src.market_regime import MarketRegimeDetector, IndexDataProvider

global_trainer = None
training_status = {"running": False, "epoch": 0, "train_loss": 0, "val_loss": 0, "should_stop": False}
download_status = {"running": False, "progress": 0, "total": 0, "current_stock": ""}

LANG = {
    "zh": {
        "title": "A股多维度金融模型训练系统",
        "subtitle": "基于 PyTorch + Flash Attention 2.8.2 + CUDA 12.8",
        "flash_status": "Flash Attention 状态",
        "data_download_tab": "📥 数据下载",
        "basic_training_tab": "🎯 基础训练",
        "multi_dim_tab": "🎨 多维度训练",
        "inference_tab": "🔮 模型推理",
        "custom_data_tab": "📁 自定义数据",
    },
    "en": {
        "title": "A-Share Multi-Dimensional Financial Model",
        "subtitle": "Powered by PyTorch + Flash Attention 2.8.2 + CUDA 12.8",
        "flash_status": "Flash Attention Status",
        "data_download_tab": "📥 Data Download",
        "basic_training_tab": "🎯 Basic Training",
        "multi_dim_tab": "🎨 Multi-Dim Training",
        "inference_tab": "🔮 Inference",
        "custom_data_tab": "📁 Custom Data",
    }
}

current_lang = "zh"

def check_flash_attention():
    """Check if Flash Attention is available / 检查 Flash Attention 是否可用"""
    try:
        import flash_attn
        return f"✅ Flash Attention {flash_attn.__version__} 可用 / Available"
    except ImportError:
        return "❌ Flash Attention 不可用 / Not available"


def get_available_stocks(randomize=False):
    """
    Get list of available stock codes from training data directory

    Args:
        randomize: If True, shuffle the stock list for better diversity
    """
    import random

    data_dir = "full_stock_data/training_data"

    if not os.path.exists(data_dir):
        return []

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    stock_codes = sorted([f.replace('.csv', '') for f in files])

    if randomize:
        stock_codes_copy = stock_codes.copy()
        random.shuffle(stock_codes_copy)
        return stock_codes_copy

    return stock_codes


def get_available_checkpoints():
    """List available checkpoint files for selection in the UI."""
    checkpoint_dirs = [
        "financial_model/checkpoints",
        "checkpoints",
        "output"
    ]

    all_checkpoints = []

    for base_dir in checkpoint_dirs:
        if not os.path.exists(base_dir):
            continue

        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith(".pt") or f.endswith(".pth"):
                    full_path = os.path.join(root, f)
                    all_checkpoints.append(full_path)

    return sorted(all_checkpoints)


def normalize_stock_code(code):
    """
    Normalize stock code with intelligent prefix detection

    A-Share Stock Code Rules:
        Shanghai (SH):
            600xxx - Main Board (large caps)
            601xxx - Main Board (large caps)
            603xxx - Main Board (IPO)
            605xxx - Main Board (new)
            688xxx - STAR Market (tech)

        Shenzhen (SZ):
            000xxx - Main Board
            001xxx - Main Board (new)
            002xxx - SME Board (mid caps)
            003xxx - SME Board (new)
            300xxx - ChiNext (growth/tech)
            301xxx - ChiNext (new)

    Examples:
        '600017' -> 'SH600017'
        '000001' -> 'SZ000001'
        '300750' -> 'SZ300750'
        'SH600017' -> 'SH600017' (unchanged)
    """
    if not code:
        return code

    code = code.strip().upper()

    if code.startswith('SH') or code.startswith('SZ'):
        return code

    if len(code) == 6 and code.isdigit():
        first_char = code[0]
        first_three = code[:3]

        if first_char == '6':
            return f'SH{code}'
        elif first_three in ['688']:
            return f'SH{code}'
        elif first_char in ['0', '3']:
            return f'SZ{code}'
        elif first_three in ['002', '003', '300', '301']:
            return f'SZ{code}'
        else:
            return f'SH{code}'

    return code


def get_stock_info(stock_code):
    """Get information about a specific stock"""
    if not stock_code or stock_code == "No stocks available - Please download data first":
        return "Please select a valid stock code"

    stock_code = normalize_stock_code(stock_code)

    data_dir = "full_stock_data/training_data"
    stock_file = os.path.join(data_dir, f"{stock_code}.csv")

    if not os.path.exists(stock_file):
        return f"Stock data file not found: {stock_code}\n\nTip: Try entering just the 6-digit code (e.g., 600017 or 000001)"

    try:
        data = pd.read_csv(stock_file)

        info = f"""
Stock Information: {stock_code}
{'='*60}
Total Records: {len(data)} days
Date Range: {data['date'].min()} to {data['date'].max()}
Columns: {', '.join(data.columns.tolist())}

Latest 5 days:
{data.tail(5)[['date', 'open', 'high', 'low', 'close', 'volume']].to_string(index=False)}
"""
        return info
    except Exception as e:
        return f"Error reading stock data: {str(e)}"

def check_data_status():
    """Check downloaded data status"""
    data_dir = "full_stock_data/training_data"
    metadata_file = "full_stock_data/metadata.json"

    if not os.path.exists(data_dir):
        return "Data directory not found. Please download data first."

    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        info = f"""
Dataset Status
{'='*60}
Downloaded: {len(metadata.get('downloaded_stocks', []))}
Failed: {len(metadata.get('failed_stocks', []))}
Total Records: {metadata.get('total_records', 0):,}
Last Update: {metadata.get('last_update', 'N/A')}
CSV Files: {len(files)}

Available stocks for Basic Training:
"""
        stock_codes = get_available_stocks()
        for i, code in enumerate(stock_codes[:20]):
            info += f"  {code}"
            if (i + 1) % 5 == 0:
                info += "\n"

        if len(stock_codes) > 20:
            info += f"\n  ... and {len(stock_codes) - 20} more stocks\n"

        return info
    else:
        return f"Found {len(files)} CSV files\nMetadata file not found"


def get_stock_list():
    """Get complete A-share stock list / 获取完整A股列表"""
    stock_codes = []

    for prefix in ['600', '601', '603', '605', '688']:
        for i in range(1000):
            code = f"SH{prefix}{i:03d}"
            stock_codes.append(code)

    for prefix in ['000', '001', '002', '003', '300', '301']:
        for i in range(1000):
            code = f"SZ{prefix}{i:03d}"
            stock_codes.append(code)

    return stock_codes


def download_stock_data(max_stocks, start_date, progress=gr.Progress()):
    """Download A-share stock data / 下载A股数据"""
    global download_status

    try:
        download_status["running"] = True
        download_status["progress"] = 0

        data_dir = "full_stock_data/training_data"
        os.makedirs(data_dir, exist_ok=True)

        metadata_file = "full_stock_data/metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            metadata = {
                "downloaded_stocks": [],
                "failed_stocks": [],
                "total_records": 0,
                "last_update": ""
            }

        # Determine which stocks still need to be downloaded based on existing CSV files
        existing_files = [
            f.replace(".csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")
        ]
        existing_set = set(existing_files)

        all_codes = get_stock_list()
        remaining_codes = [code for code in all_codes if code not in existing_set]

        try:
            max_stocks_int = int(max_stocks)
        except Exception:
            max_stocks_int = 50

        if max_stocks_int <= 0:
            max_stocks_int = len(remaining_codes)

        stock_codes = remaining_codes[:max_stocks_int]
        download_status["total"] = len(stock_codes)

        if not stock_codes:
            download_status["running"] = False
            yield "No new stocks to download. Existing data is already complete for the current universe.\n"
            return

        yield (
            "Starting incremental download\n"
            f"Existing CSV files: {len(existing_files)}\n"
            f"Planned new stocks this run: {len(stock_codes)}\n"
            f"Start date: {start_date}\n\n"
        )

        api = TdxHq_API()
        server = "124.71.187.122"
        port = 7709

        if not api.connect(server, port):
            download_status["running"] = False
            yield "❌ 连接TDX服务器失败 / Failed to connect to TDX server\n"
            return

        yield f"✅ 已连接到TDX服务器 / Connected to TDX server: {server}:{port}\n\n"

        downloaded = 0
        failed = 0

        for idx, code in enumerate(stock_codes):
            download_status["progress"] = idx + 1
            download_status["current_stock"] = code

            progress((idx + 1) / len(stock_codes), desc=f"下载中 / Downloading: {code}")

            try:
                market = 1 if code.startswith('SH') else 0
                stock_code = code.replace('SH', '').replace('SZ', '')

                data = api.get_security_bars(9, market, stock_code, 0, 1600)

                if data and len(data) > 100:
                    df = api.to_df(data)
                    df = df[['datetime', 'open', 'high', 'low', 'close', 'vol']]
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                    df = df.sort_values('date')

                    if start_date:
                        df = df[df['date'] >= start_date]

                    csv_path = os.path.join(data_dir, f"{code}.csv")
                    df.to_csv(csv_path, index=False)

                    downloaded += 1
                    if code not in metadata["downloaded_stocks"]:
                        metadata["downloaded_stocks"].append(code)
                    metadata["total_records"] += len(df)

                    if downloaded % 10 == 0:
                        yield (
                            f"Downloaded: {downloaded}/{len(stock_codes)} | "
                            f"Failed: {failed}\nCurrent: {code} ({len(df)} records)\n"
                        )
                else:
                    failed += 1
                    if code not in metadata["failed_stocks"]:
                        metadata["failed_stocks"].append(code)

            except Exception as e:
                failed += 1
                metadata["failed_stocks"].append(code)
                if failed % 20 == 0:
                    yield f"⚠️ 失败 / Failed: {failed} | 最新错误 / Latest: {code}\n"

        api.disconnect()

        metadata["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        download_status["running"] = False

        summary = f"""
{'='*60}
✅ 下载完成 / Download Complete
{'='*60}
✅ 成功 / Success: {downloaded}
❌ 失败 / Failed: {failed}
📈 总记录数 / Total Records: {metadata['total_records']:,}
📁 保存位置 / Saved to: {data_dir}
🕐 完成时间 / Completed: {metadata['last_update']}
"""
        yield summary

    except Exception as e:
        download_status["running"] = False
        yield f"❌ 下载出错 / Download error: {str(e)}\n"



def inspect_download_plan(max_stocks, start_date):
    """Inspect existing data and planned incremental download without downloading."""
    data_dir = "full_stock_data/training_data"
    os.makedirs(data_dir, exist_ok=True)

    metadata_file = "full_stock_data/metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "downloaded_stocks": [],
            "failed_stocks": [],
            "total_records": 0,
            "last_update": "",
        }

    existing_files = [
        f.replace(".csv", "") for f in os.listdir(data_dir) if f.endswith(".csv")
    ]
    existing_set = set(existing_files)

    all_codes = get_stock_list()
    remaining_codes = [code for code in all_codes if code not in existing_set]

    try:
        max_stocks_int = int(max_stocks)
    except Exception:
        max_stocks_int = 50

    if max_stocks_int <= 0:
        planned_new = len(remaining_codes)
    else:
        planned_new = min(max_stocks_int, len(remaining_codes))

    sample_new = remaining_codes[: min(10, len(remaining_codes))]

    info_lines = []
    info_lines.append("A-Share Data Check")
    info_lines.append("=" * 60)
    info_lines.append(f"Existing CSV files: {len(existing_files)}")
    info_lines.append(
        f"Metadata downloaded stocks: {len(metadata.get('downloaded_stocks', []))}"
    )
    info_lines.append(
        f"Failed stocks recorded: {len(metadata.get('failed_stocks', []))}"
    )
    info_lines.append(f"Total records: {metadata.get('total_records', 0):,}")
    last_update = metadata.get("last_update") or "N/A"
    info_lines.append(f"Last update: {last_update}")
    info_lines.append("")
    info_lines.append(f"Universe size (get_stock_list): {len(all_codes)}")
    info_lines.append(
        f"Remaining stocks without CSV: {len(remaining_codes)}"
    )
    info_lines.append(
        f"Requested max new stocks this run: {max_stocks_int}"
    )
    info_lines.append(
        f"Planned new stocks if you start download: {planned_new}"
    )
    info_lines.append(f"Start date: {start_date}")

    if sample_new:
        info_lines.append("")
        info_lines.append("Sample of new stocks to download:")
        info_lines.append(", ".join(sample_new))
    else:
        info_lines.append("")
        info_lines.append(
            "No remaining stocks to download. Existing data is already complete for the current universe."
        )

    return "\n".join(info_lines)

def create_config(model_type, hidden_dim, num_layers, num_heads, batch_size, epochs, lr):
    """
    Create configuration from UI inputs
    从UI输入创建配置

    Args:
        model_type: Model architecture (Transformer/LSTM) / 模型架构
        hidden_dim: Hidden dimension size / 隐藏层维度
        num_layers: Number of layers / 层数
        num_heads: Number of attention heads (Transformer only) / 注意力头数
        batch_size: Training batch size / 批次大小
        epochs: Number of training epochs / 训练轮数
        lr: Learning rate / 学习率
    """
    config = Config()
    config.model.model_type = model_type.lower()
    config.model.hidden_dim = hidden_dim
    config.model.num_layers = num_layers
    config.model.num_heads = num_heads
    config.training.batch_size = batch_size
    config.training.num_epochs = epochs
    config.training.learning_rate = lr
    return config

def train_model(model_type, hidden_dim, num_layers, num_heads, batch_size, epochs, lr, stock_code):
    """
    Train model with real-time loss curve updates on real stock data
    """
    global global_trainer, training_status

    try:
        training_status["running"] = True
        training_status["epoch"] = 0

        config = create_config(model_type, hidden_dim, num_layers, num_heads, batch_size, epochs, lr)

        data_dir = "full_stock_data/training_data"
        stock_file = os.path.join(data_dir, f"{stock_code}.csv")

        if not os.path.exists(stock_file):
            error_fig = go.Figure()
            error_fig.update_layout(title=f'Error: Stock {stock_code} not found')
            yield f"Error: Stock data not found for {stock_code}\nFile: {stock_file}\n", error_fig
            return

        data = pd.read_csv(stock_file)

        if len(data) < 100:
            error_fig = go.Figure()
            error_fig.update_layout(title='Error: Not enough data')
            yield f"Error: Not enough data for {stock_code} (need at least 100 rows)\n", error_fig
            return

        train_loader, val_loader, test_loader, scaler = create_dataloaders(data, config)

        global_trainer = Trainer(config)

        model_params = sum(p.numel() for p in global_trainer.model.parameters())
        device_info = f"Device: {config.model.device}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"

        flash_status = check_flash_attention()

        init_log = f"""
Training Started
{'='*60}
{device_info}
{flash_status}
Stock: {stock_code}
Data Points: {len(data)}
Model: {model_type}
Parameters: {model_params:,}
Batch Size: {batch_size}
Learning Rate: {lr}

"""

        init_fig = go.Figure()
        init_fig.update_layout(
            title='Training will start...',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )

        yield init_log, init_fig

        for epoch in range(config.training.num_epochs):
            training_status["epoch"] = epoch + 1

            train_loss = global_trainer.train_epoch(train_loader)
            val_loss = global_trainer.validate(val_loader)

            training_status["train_loss"] = train_loss
            training_status["val_loss"] = val_loss

            global_trainer.train_losses.append(train_loss)
            global_trainer.val_losses.append(val_loss)

            if val_loss < global_trainer.best_val_loss:
                global_trainer.best_val_loss = val_loss
                global_trainer.patience_counter = 0
                global_trainer.save_checkpoint("best_model.pt")
                status = "NEW BEST!"
            else:
                global_trainer.patience_counter += 1
                status = ""

            log_text = init_log + f"\nEpoch {epoch+1}/{config.training.num_epochs}\nTrain Loss: {train_loss:.6f}\nVal Loss: {val_loss:.6f} {status}\n"

            fig = make_subplots(rows=1, cols=1)
            epochs_list = list(range(1, len(global_trainer.train_losses) + 1))

            fig.add_trace(go.Scatter(
                x=epochs_list,
                y=global_trainer.train_losses,
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=epochs_list,
                y=global_trainer.val_losses,
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title=f'Training Progress - Epoch {epoch+1}/{config.training.num_epochs}',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )

            yield log_text, fig

            if global_trainer.patience_counter >= config.training.early_stopping_patience:
                final_log = log_text + f"\nEarly stopping!\nBest Val Loss: {global_trainer.best_val_loss:.6f}\n"
                yield final_log, fig
                break

        training_status["running"] = False
        final_log = log_text + f"\nCompleted!\nBest Val Loss: {global_trainer.best_val_loss:.6f}\nSaved to: financial_model/checkpoints/best_model.pt\n"
        yield final_log, fig

    except Exception as e:
        training_status["running"] = False
        error_fig = go.Figure()
        error_fig.update_layout(title='Training Error')
        yield f"Error: {str(e)}\n", error_fig

def plot_training_history():
    """Plot training history"""
    global global_trainer

    if global_trainer is None or len(global_trainer.train_losses) == 0:
        return None

    fig = make_subplots(rows=1, cols=1)

    epochs = list(range(1, len(global_trainer.train_losses) + 1))

    fig.add_trace(go.Scatter(
        x=epochs, y=global_trainer.train_losses,
        mode='lines+markers', name='Train Loss',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=epochs, y=global_trainer.val_losses,
        mode='lines+markers', name='Val Loss',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified',
        template='plotly_white'
    )

    return fig


def get_stock_list_by_industry(industry, max_stocks, data_dir):
    """Get stock codes filtered by industry."""
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    all_codes = [f.replace(".csv", "") for f in all_files]

    try:
        max_stocks = int(max_stocks)
    except Exception:
        max_stocks = 50

    if industry is None or industry == "all":
        return all_codes[: max_stocks]

    filtered_codes = []
    for code in all_codes:
        try:
            industry_l1, _ = IndustryClassifier.get_industry(code)
            if industry_l1 == industry:
                filtered_codes.append(code)
        except Exception:
            continue

    return filtered_codes[: max_stocks]


def analyze_universe(max_stocks):
    """Analyze industry, style, and market regime based on downloaded data."""
    data_dir = "full_stock_data/training_data"
    if not os.path.exists(data_dir):
        return "Error: data directory 'full_stock_data/training_data' not found. Please run batch_download.py first.", None

    try:
        max_stocks = int(max_stocks)
    except Exception:
        max_stocks = 50

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not all_files:
        return "Error: no CSV files found in 'full_stock_data/training_data'.", None

    codes = [f.replace(".csv", "") for f in all_files][:max_stocks]

    industry_counts = {}
    volatility_counts = {"low_vol": 0, "medium_vol": 0, "high_vol": 0}

    for code in codes:
        try:
            industry_l1, industry_l2 = IndustryClassifier.get_industry(code)
            key = industry_l1
            industry_counts[key] = industry_counts.get(key, 0) + 1

            csv_path = os.path.join(data_dir, f"{code}.csv")
            df = pd.read_csv(csv_path)
            if "close" in df.columns and len(df) > 40:
                returns = df["close"].pct_change()
                vol_20d = returns.rolling(window=20).std().iloc[-1]
                if pd.isna(vol_20d):
                    continue
                if vol_20d < 0.01:
                    bucket = "low_vol"
                elif vol_20d < 0.02:
                    bucket = "medium_vol"
                else:
                    bucket = "high_vol"
                volatility_counts[bucket] += 1
        except Exception:
            continue

    # Market regime analysis using index data provider
    try:
        index_df = IndexDataProvider.get_index_data()
        detector = MarketRegimeDetector()
        current_regime = detector.detect_regime(index_df["close"])
        regime_series = detector.detect_regime_series(index_df["close"])
        regime_counts = regime_series.value_counts().to_dict()
    except Exception:
        current_regime = "unknown"
        regime_counts = {}

    lines = []
    lines.append(f"Total CSV files in data directory: {len(all_files)}")
    lines.append(f"Analyzed stocks: {len(codes)} (max_stocks={max_stocks})")
    lines.append("")
    lines.append("Industry distribution (level 1):")
    for name, count in sorted(industry_counts.items(), key=lambda x: x[0]):
        lines.append(f"  - {name}: {count}")
    lines.append("")
    lines.append("Volatility style (20 day standard deviation):")
    for name in ["low_vol", "medium_vol", "high_vol"]:
        lines.append(f"  - {name}: {volatility_counts[name]}")
    lines.append("")
    lines.append(f"Current market regime: {current_regime}")
    if regime_counts:
        lines.append("Historical market regimes:")
        for name, count in regime_counts.items():
            lines.append(f"  - {name}: {count}")

    summary_text = "\n".join(lines)

    # Create summary plot
    cols = 2
    fig = make_subplots(rows=1, cols=cols, subplot_titles=("Industry distribution", "Volatility styles"))

    if industry_counts:
        fig.add_trace(
            go.Bar(
                x=list(industry_counts.keys()),
                y=list(industry_counts.values()),
                name="Industry",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=["low_vol", "medium_vol", "high_vol"],
            y=[
                volatility_counts["low_vol"],
                volatility_counts["medium_vol"],
                volatility_counts["high_vol"],
            ],
            name="Volatility",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        title="Universe analysis (industry and volatility)",
        template="plotly_white",
    )

    return summary_text, fig


def check_cache_status():
    """Check current cache status"""
    cache_dir = "data_cache"
    db_path = os.path.join(cache_dir, "meta.sqlite")

    if not os.path.exists(db_path):
        return 0, "No cache found"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks")
        count = cursor.fetchone()[0]
        conn.close()
        return count, f"{count} stocks cached"
    except Exception as e:
        return 0, f"Error reading cache: {str(e)}"


def stop_multi_dim_training():
    """Stop multi-dimensional training immediately"""
    global training_status, global_trainer

    if training_status["running"] and global_trainer is not None:
        global_trainer.should_stop = True
        training_status["should_stop"] = True
        return "⏹️ Training stopped immediately!"
    else:
        return "⚠️ No training is currently running."


def build_cache_for_training(max_stocks=100, use_gpu=True):
    """Build cache files for fast training with progress updates"""
    try:
        import time
        import sys
        import importlib.util

        build_cache_path = os.path.join(os.path.dirname(__file__), 'build_cache.py')
        spec = importlib.util.spec_from_file_location("build_cache", build_cache_path)
        build_cache_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_cache_module)
        CacheDatabaseBuilder = build_cache_module.CacheDatabaseBuilder

        data_dir = "full_stock_data/training_data"
        cache_dir = "data_cache"

        cached_count, cache_status = check_cache_status()
        log = f"Current cache status: {cache_status}\n\n"

        if not os.path.exists(data_dir):
            error_msg = log + f"Error: Data directory not found: {data_dir}\n"
            error_fig = go.Figure()
            error_fig.update_layout(title='Error: Data directory not found')
            yield error_msg, error_fig
            return

        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not csv_files:
            error_msg = log + f"Error: No CSV files found in {data_dir}\n"
            error_fig = go.Figure()
            error_fig.update_layout(title='Error: No CSV files found')
            yield error_msg, error_fig
            return

        stock_codes = sorted([f.replace('.csv', '') for f in csv_files])[:max_stocks]

        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        device_info = f"GPU ({torch.cuda.get_device_name(0)})" if device == 'cuda' else "CPU"

        log = f"Building cache for {len(stock_codes)} stocks using {device_info}...\n"
        log += f"Data directory: {data_dir}\n"
        log += f"Cache directory: {cache_dir}\n\n"

        init_fig = go.Figure()
        init_fig.update_layout(title='Initializing cache builder...')
        yield log, init_fig

        builder = CacheDatabaseBuilder(
            data_dir=data_dir,
            cache_dir=cache_dir,
            device=device
        )

        start_time = time.perf_counter()
        success_count = 0
        error_count = 0

        for idx, code in enumerate(stock_codes):
            try:
                builder.build_for_stock(code)
                success_count += 1
                current = idx + 1

                if current % 5 == 0 or current == len(stock_codes):
                    elapsed = time.perf_counter() - start_time
                    speed = current / elapsed if elapsed > 0 else 0
                    eta = (len(stock_codes) - current) / speed if speed > 0 else 0

                    log += f"[{current}/{len(stock_codes)}] {code} - Success ({speed:.1f} stocks/sec, ETA: {eta:.1f}s)\n"

                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=['Success', 'Error'],
                        y=[success_count, error_count],
                        marker=dict(color=['#4CAF50', '#F44336']),
                        text=[success_count, error_count],
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title=f'Cache Building Progress: {current}/{len(stock_codes)} ({current*100//len(stock_codes)}%)',
                        yaxis=dict(title='Count'),
                        template='plotly_white',
                        height=400
                    )
                    yield log, fig

            except Exception as e:
                error_count += 1
                log += f"[{idx+1}/{len(stock_codes)}] {code} - Error: {str(e)}\n"

        total_time = time.perf_counter() - start_time
        avg_speed = len(stock_codes) / total_time if total_time > 0 else 0

        log += f"\n{'='*60}\n"
        log += f"Cache Building Complete!\n"
        log += f"{'='*60}\n"
        log += f"Total stocks: {len(stock_codes)}\n"
        log += f"Success: {success_count}\n"
        log += f"Errors: {error_count}\n"
        log += f"Total time: {total_time:.2f}s\n"
        log += f"Average speed: {avg_speed:.2f} stocks/sec\n"
        log += f"Cache directory: {cache_dir}\n"

        final_fig = go.Figure()
        final_fig.add_trace(go.Bar(
            x=['Success', 'Error'],
            y=[success_count, error_count],
            marker=dict(color=['#4CAF50', '#F44336']),
            text=[success_count, error_count],
            textposition='auto'
        ))
        final_fig.update_layout(
            title=f'Cache Building Complete: {success_count} success, {error_count} errors',
            yaxis=dict(title='Count'),
            template='plotly_white',
            height=400
        )
        yield log, final_fig

    except Exception as e:
        import traceback
        error_msg = f"Error building cache: {str(e)}\n{traceback.format_exc()}"
        error_fig = go.Figure()
        error_fig.update_layout(title='Error building cache')
        yield error_msg, error_fig


def train_multi_dim_model(mode, industry, max_stocks, batch_size, epochs, lr,
                          use_amp=True, loss_type="Trend Aware", save_interval=5, early_stopping_patience=50,
                          model_name="multi_dim_model", output_dir="checkpoints", use_cache=True, use_compile=False):
    """Train conditional multi dimensional model with real-time curve updates"""
    global global_trainer, training_status

    try:
        training_status["running"] = True
        training_status["epoch"] = 0
        training_status["should_stop"] = False

        config = Config()
        config.model.seq_length = 60
        config.model.pred_length = 1
        config.model.hidden_dim = 128
        config.model.num_layers = 2
        config.training.batch_size = int(batch_size)

        loss_type_map = {
            "MSE": "mse",
            "Trend Aware": "trend_aware",
            "Huber Trend": "huber_trend",
            "Directional": "directional"
        }
        config.training.loss_type = loss_type_map.get(loss_type, "trend_aware")
        config.training.num_epochs = int(epochs)
        config.training.learning_rate = float(lr)
        config.training.use_amp = bool(use_amp)
        config.training.early_stopping_patience = int(early_stopping_patience)

        save_interval = int(save_interval)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_with_timestamp = f"{model_name}_{timestamp}"

        os.makedirs(output_dir, exist_ok=True)

        data_dir = "full_stock_data/training_data"
        if not os.path.exists(data_dir):
            training_status["running"] = False
            error_fig = go.Figure()
            error_fig.update_layout(title='Error: Data directory not found')
            yield "Error: data directory not found\n", error_fig
            return

        try:
            max_stocks = int(max_stocks)
        except Exception:
            max_stocks = 100

        if mode.startswith("Single"):
            if not industry or industry == "all":
                training_status["running"] = False
                error_fig = go.Figure()
                error_fig.update_layout(title='Error: Select industry')
                yield "Error: select an industry\n", error_fig
                return
            stock_codes = get_stock_list_by_industry(industry, max_stocks, data_dir)
        else:
            stock_codes = get_stock_list_by_industry("all", max_stocks, data_dir)

        if not stock_codes:
            training_status["running"] = False
            error_fig = go.Figure()
            error_fig.update_layout(title='Error: No stocks available')
            yield "Error: no stocks available\n", error_fig
            return

        loading_log = f"Loading {len(stock_codes)} stocks from {data_dir}...\n"

        loading_fig = go.Figure()
        loading_fig.update_layout(
            title='Preparing to load stock data...',
            template='plotly_white'
        )
        yield loading_log, loading_fig

        train_loader = None
        val_loader = None
        test_loader = None

        try:
            if use_cache:
                cache_dir = "data_cache"
                cached_count, cache_status = check_cache_status()

                if cached_count == 0:
                    loading_log += f"\n⚠️ {cache_status}\n"
                    loading_log += f"⚠️ Falling back to CSV loading...\n"
                    loading_log += f"💡 Tip: Build cache first for 1.7x faster loading!\n\n"
                    use_cache = False
                else:
                    loading_log += f"\n✅ Cache found: {cache_status}\n"
                    loading_log += f"✅ Using pre-built cache (404x faster regime detection, 1.7x faster loading)\n\n"

            if use_cache:
                for status, current, total, msg, tl, vl, tel in create_multi_dim_dataloaders_from_cache(
                    stock_codes,
                    config,
                    cache_dir=cache_dir
                ):
                    if status == 'loading':
                        loading_log += f"{msg}\n"

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['Loaded'],
                            y=[current],
                            text=[f'{current}/{total}'],
                            textposition='auto',
                            marker=dict(color='#2196F3')
                        ))
                        fig.update_layout(
                            title=f'Loading from Cache: {current}/{total} stocks ({current*100//total}%)',
                            yaxis=dict(range=[0, total], title='Stocks'),
                            template='plotly_white',
                            showlegend=False,
                            height=400
                        )
                        yield loading_log, fig

                    elif status == 'complete':
                        train_loader = tl
                        val_loader = vl
                        test_loader = tel
                        loading_log += f"\n{msg}\nTrain batches: {len(train_loader)}\nVal batches: {len(val_loader)}\n"
                        yield loading_log, fig
            else:
                for status, current, total, msg, tl, vl, tel in create_multi_dim_dataloaders_with_progress(
                    stock_codes,
                    config,
                    data_dir=data_dir,
                    use_gpu_normalize=True
                ):
                    if status == 'loading':
                        loading_log += f"{msg}\n"

                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=['Loaded'],
                            y=[current],
                            text=[f'{current}/{total}'],
                            textposition='auto',
                            marker=dict(color='#4CAF50')
                        ))
                        fig.update_layout(
                            title=f'Loading Progress: {current}/{total} stocks ({current*100//total}%)',
                            yaxis=dict(range=[0, total], title='Stocks'),
                            template='plotly_white',
                            showlegend=False,
                            height=400
                        )
                        yield loading_log, fig

                    elif status == 'complete':
                        train_loader = tl
                        val_loader = vl
                        test_loader = tel
                        loading_log += f"\n{msg}\nTrain batches: {len(train_loader)}\nVal batches: {len(val_loader)}\n"
                        yield loading_log, fig

        except Exception as e:
            import traceback
            training_status["running"] = False
            error_fig = go.Figure()
            error_fig.update_layout(title='Error loading data')
            error_msg = f"Error loading data:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}\n"
            yield error_msg, error_fig
            return

        if train_loader is None or len(train_loader) == 0:
            training_status["running"] = False
            error_fig = go.Figure()
            error_fig.update_layout(title='Error: Empty dataloader')
            yield "Error: empty dataloader\n", error_fig
            return

        global_trainer = Trainer(config, use_conditional=True)

        if use_compile and hasattr(torch, 'compile'):
            loading_log += f"\n🚀 Compiling model with torch.compile...\n"

            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = False
            inductor_config.coordinate_descent_tuning = True
            inductor_config.epilogue_fusion = True
            inductor_config.max_autotune = False

            device_before = str(next(global_trainer.model.parameters()).device)
            loading_log += f"Model device before compile: {device_before}\n"

            global_trainer.model = torch.compile(
                global_trainer.model,
                mode="default",
                fullgraph=False,
                dynamic=False
            )

            global_trainer.model = global_trainer.model.to(global_trainer.device)

            device_after = str(next(global_trainer.model.parameters()).device)
            loading_log += f"Model device after compile: {device_after}\n"
            loading_log += f"✅ Model compiled! First epoch will be slow (compiling kernels)\n"
            loading_log += f"   Subsequent epochs will be ~1.24x faster\n"

        model_params = sum(p.numel() for p in global_trainer.model.parameters())
        model_device = str(next(global_trainer.model.parameters()).device)

        device_info = f"Config Device: {config.model.device}"
        if torch.cuda.is_available():
            device_info += f"\nGPU: {torch.cuda.get_device_name(0)}"
            device_info += f"\nGPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            device_info += f"\nModel Device: {model_device}"

        flash_status = check_flash_attention()

        if mode.startswith("Single"):
            mode_desc = f"Single industry: {industry}"
        else:
            mode_desc = "Unified model (all industries)"

        data_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        header = (
            f"Multi-Dimensional Training\n"
            f"{'='*60}\n"
            f"{device_info}\n"
            f"{flash_status}\n"
            f"Mode: {mode_desc}\n"
            f"Model Name: {model_name_with_timestamp}\n"
            f"Output Dir: {output_dir}\n"
            f"Save Interval: Every {save_interval} epochs\n"
            f"Early Stopping Patience: {config.training.early_stopping_patience} epochs\n"
            f"Loss Function: {loss_type} ({config.training.loss_type})\n"
            f"Data dir: {data_dir} (CSV files: {len(data_files)})\n"
            f"Stocks used this run: {len(stock_codes)}\n"
            f"Train batches: {len(train_loader)}\n"
            f"Val batches: {len(val_loader)}\n"
            f"Parameters: {model_params:,}\n"
            f"Use AMP (Mixed Precision): {config.training.use_amp}\n"
            f"Batch Size: {config.training.batch_size}\n"
            f"Learning Rate: {config.training.learning_rate}\n"
            f"Epochs: {config.training.num_epochs}\n\n"
        )

        init_fig = go.Figure()
        init_fig.update_layout(
            title='Training will start...',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )

        yield header, init_fig

        for epoch in range(config.training.num_epochs):
            training_status["epoch"] = epoch + 1

            try:
                train_loss = global_trainer.train_epoch(train_loader)
                val_loss = global_trainer.validate(val_loader)
            except KeyboardInterrupt:
                stop_log = header + f"\n\n⏹️ Training stopped by user at epoch {epoch+1}/{config.training.num_epochs}\n"
                if len(global_trainer.train_losses) > 0:
                    stop_log += f"Last Train Loss: {global_trainer.train_losses[-1]:.6f}\n"
                if len(global_trainer.val_losses) > 0:
                    stop_log += f"Last Val Loss: {global_trainer.val_losses[-1]:.6f}\n"

                final_model_path = os.path.join(output_dir, f"{model_name_with_timestamp}_stopped_epoch{epoch+1}.pt")
                global_trainer.save_checkpoint(final_model_path)
                stop_log += f"\nModel saved to: {final_model_path}\n"

                training_status["running"] = False
                training_status["should_stop"] = False

                fig = make_subplots(rows=1, cols=1)
                if len(global_trainer.train_losses) > 0:
                    epochs_list = list(range(1, len(global_trainer.train_losses) + 1))
                    fig.add_trace(go.Scatter(
                        x=epochs_list, y=global_trainer.train_losses,
                        mode='lines+markers', name='Train Loss',
                        line=dict(color='blue', width=2), marker=dict(size=8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=epochs_list, y=global_trainer.val_losses,
                        mode='lines+markers', name='Val Loss',
                        line=dict(color='red', width=2), marker=dict(size=8)
                    ))
                    fig.update_layout(
                        title=f'Training Stopped at Epoch {epoch+1}',
                        xaxis_title='Epoch', yaxis_title='Loss',
                        template='plotly_white', showlegend=True
                    )

                yield stop_log, fig
                return

            training_status["train_loss"] = train_loss
            training_status["val_loss"] = val_loss

            global_trainer.train_losses.append(train_loss)
            global_trainer.val_losses.append(val_loss)

            save_info = ""

            if val_loss < global_trainer.best_val_loss:
                global_trainer.best_val_loss = val_loss
                global_trainer.patience_counter = 0
                best_model_path = os.path.join(output_dir, f"{model_name_with_timestamp}_best.pt")
                global_trainer.save_checkpoint(best_model_path)
                status = "NEW BEST"
                save_info = f" -> Saved to {best_model_path}"
            else:
                global_trainer.patience_counter += 1
                status = ""

            if (epoch + 1) % save_interval == 0:
                interval_model_path = os.path.join(output_dir, f"{model_name_with_timestamp}_epoch{epoch+1}.pt")
                global_trainer.save_checkpoint(interval_model_path)
                save_info += f" -> Checkpoint saved to {interval_model_path}"

            log_text = header + f"\nEpoch {epoch + 1}/{config.training.num_epochs}\nTrain Loss: {train_loss:.6f}\nVal Loss: {val_loss:.6f} {status}{save_info}\n"

            fig = make_subplots(rows=1, cols=1)
            epochs_list = list(range(1, len(global_trainer.train_losses) + 1))

            fig.add_trace(go.Scatter(
                x=epochs_list,
                y=global_trainer.train_losses,
                mode='lines+markers',
                name='Train Loss',
                line=dict(color='blue', width=2),
                marker=dict(size=8)
            ))

            fig.add_trace(go.Scatter(
                x=epochs_list,
                y=global_trainer.val_losses,
                mode='lines+markers',
                name='Val Loss',
                line=dict(color='red', width=2),
                marker=dict(size=8)
            ))

            fig.update_layout(
                title=f'Multi-Dim Training - Epoch {epoch+1}/{config.training.num_epochs}',
                xaxis_title='Epoch',
                yaxis_title='Loss',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True
            )

            yield log_text, fig

            if global_trainer.patience_counter >= config.training.early_stopping_patience:
                final_model_path = os.path.join(output_dir, f"{model_name_with_timestamp}_final.pt")
                global_trainer.save_checkpoint(final_model_path)
                best_model_path = os.path.join(output_dir, f"{model_name_with_timestamp}_best.pt")
                final_log = (
                    log_text +
                    f"\nEarly stopping!\n"
                    f"Best Val Loss: {global_trainer.best_val_loss:.6f}\n"
                    f"Best model: {best_model_path}\n"
                    f"Final model: {final_model_path}\n"
                )
                yield final_log, fig
                break

        training_status["running"] = False
        training_status["should_stop"] = False

        final_model_path = os.path.join(output_dir, f"{model_name_with_timestamp}_final.pt")
        global_trainer.save_checkpoint(final_model_path)
        best_model_path = os.path.join(output_dir, f"{model_name_with_timestamp}_best.pt")

        final_log = (
            log_text +
            f"\nTraining Completed!\n"
            f"Best Val Loss: {global_trainer.best_val_loss:.6f}\n"
            f"Best model: {best_model_path}\n"
            f"Final model: {final_model_path}\n"
            f"Output directory: {output_dir}\n"
        )
        yield final_log, fig

    except Exception as e:
        import traceback
        training_status["running"] = False
        training_status["should_stop"] = False
        error_fig = go.Figure()
        error_fig.update_layout(title='Training Error')
        error_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}\n"
        yield error_msg, error_fig


def run_inference(checkpoint_path, num_samples):
    """Run inference on test data"""
    try:
        if not os.path.exists(checkpoint_path):
            return "Error: Checkpoint not found. Please train a model first.", None

        predictor = Predictor(checkpoint_path)

        test_data = create_sample_data(num_samples=num_samples)

        predictions = predictor.predict_from_dataframe(test_data)

        actual_values = test_data['close'].values[-len(predictions):]

        fig = make_subplots(rows=1, cols=1)

        indices = list(range(len(predictions)))

        fig.add_trace(go.Scatter(
            x=indices, y=actual_values,
            mode='lines', name='Actual',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=indices, y=predictions,
            mode='lines', name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Predictions vs Actual',
            xaxis_title='Sample',
            yaxis_title='Price',
            hovermode='x unified',
            template='plotly_white'
        )

        mse = np.mean((predictions - actual_values) ** 2)
        mae = np.mean(np.abs(predictions - actual_values))

        result = f"""
Inference Results:
==================
Samples: {len(predictions)}
MSE: {mse:.6f}
MAE: {mae:.6f}

Sample Predictions:
{pd.DataFrame({
    'Actual': actual_values[:10],
    'Predicted': predictions[:10],
    'Error': np.abs(actual_values[:10] - predictions[:10])
}).to_string()}
"""

        return result, fig

    except Exception as e:
        return f"Error during inference: {str(e)}", None

def upload_data_and_predict(file, checkpoint_path):
    """Upload CSV and run predictions"""
    try:
        if file is None:
            return "Please upload a CSV file", None

        if not os.path.exists(checkpoint_path):
            return "Error: Checkpoint not found. Please train a model first.", None

        data = pd.read_csv(file.name)

        predictor = Predictor(checkpoint_path)
        predictions = predictor.predict_from_dataframe(data)

        result_df = pd.DataFrame({
            'Prediction': predictions
        })

        output_path = "predictions.csv"
        result_df.to_csv(output_path, index=False)

        return f"Predictions saved to {output_path}\n\nFirst 10 predictions:\n{result_df.head(10).to_string()}", output_path

    except Exception as e:
        return f"Error: {str(e)}", None

def analyze_industry_trends(max_stocks_per_industry, prediction_days, checkpoint_path, progress=gr.Progress()):
    """
    Analyze and compare industry trends - historical vs predicted

    Returns industry performance comparison with visualization
    """
    try:
        if not os.path.exists(checkpoint_path):
            return "Error: Checkpoint not found. Please train a model first.", None, None

        data_dir = "full_stock_data/training_data"
        if not os.path.exists(data_dir):
            return "Error: Data directory not found.", None, None

        all_stocks = get_available_stocks(randomize=True)
        if not all_stocks:
            return "Error: No stocks available.", None, None

        max_stocks_per_industry = int(max_stocks_per_industry)
        prediction_days = int(prediction_days)

        predictor = ConditionalPredictor(checkpoint_path)

        industry_data = {}
        total_processed = 0

        for stock_code in all_stocks:
            try:
                stock_file = os.path.join(data_dir, f"{stock_code}.csv")
                data = pd.read_csv(stock_file)

                if len(data) < 60:
                    continue

                predictions, metadata = predictor.predict_stock(
                    stock_code=stock_code,
                    recent_data=data,
                    future_steps=prediction_days
                )

                industry = metadata['industry_l1']

                if industry not in industry_data:
                    industry_data[industry] = {
                        'stocks': [],
                        'current_prices': [],
                        'predicted_prices': [],
                        'increase_pcts': [],
                        'historical_returns': []
                    }

                if len(industry_data[industry]['stocks']) >= max_stocks_per_industry:
                    continue

                current_price = metadata['last_price']
                predicted_price = predictions[-1]
                increase_pct = ((predicted_price - current_price) / current_price) * 100

                recent_30d = data['close'].tail(30)
                historical_return = ((recent_30d.iloc[-1] - recent_30d.iloc[0]) / recent_30d.iloc[0]) * 100

                industry_data[industry]['stocks'].append(stock_code)
                industry_data[industry]['current_prices'].append(current_price)
                industry_data[industry]['predicted_prices'].append(predicted_price)
                industry_data[industry]['increase_pcts'].append(increase_pct)
                industry_data[industry]['historical_returns'].append(historical_return)

                total_processed += 1
                progress(total_processed / (max_stocks_per_industry * 12), desc=f"Analyzing {industry}")

                if all(len(v['stocks']) >= max_stocks_per_industry for v in industry_data.values() if len(v['stocks']) > 0):
                    if len(industry_data) >= 10:
                        break

            except Exception as e:
                continue

        if not industry_data:
            return "No data collected. Try increasing stocks per industry.", None, None

        industry_summary = []
        for industry, data_dict in industry_data.items():
            if len(data_dict['stocks']) == 0:
                continue

            avg_historical = np.mean(data_dict['historical_returns'])
            avg_predicted = np.mean(data_dict['increase_pcts'])
            stock_count = len(data_dict['stocks'])

            industry_summary.append({
                'industry': industry,
                'stock_count': stock_count,
                'avg_historical_30d': avg_historical,
                'avg_predicted': avg_predicted,
                'momentum_shift': avg_predicted - avg_historical,
                'stocks': data_dict['stocks']
            })

        summary_df = pd.DataFrame(industry_summary)
        summary_df = summary_df.sort_values('avg_predicted', ascending=False)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Industry Performance Comparison',
                'Momentum Shift (Predicted - Historical)',
                'Historical 30-Day Returns',
                'Predicted Returns Heatmap'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}]
            ]
        )

        industries = summary_df['industry'].tolist()
        historical = summary_df['avg_historical_30d'].tolist()
        predicted = summary_df['avg_predicted'].tolist()
        momentum = summary_df['momentum_shift'].tolist()

        fig.add_trace(go.Bar(
            name='Historical (30d)',
            x=industries,
            y=historical,
            marker_color='lightblue',
            text=[f"{v:.2f}%" for v in historical],
            textposition='outside'
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            name=f'Predicted ({prediction_days}d)',
            x=industries,
            y=predicted,
            marker_color='orange',
            text=[f"{v:.2f}%" for v in predicted],
            textposition='outside'
        ), row=1, col=1)

        momentum_colors = ['green' if m > 0 else 'red' for m in momentum]
        fig.add_trace(go.Bar(
            x=industries,
            y=momentum,
            marker_color=momentum_colors,
            text=[f"{v:+.2f}%" for v in momentum],
            textposition='outside',
            showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=industries,
            y=historical,
            marker_color='steelblue',
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=historical,
            y=predicted,
            mode='markers+text',
            text=industries,
            textposition='top center',
            marker=dict(size=15, color=momentum, colorscale='RdYlGn', showscale=True),
            showlegend=False
        ), row=2, col=2)

        fig.add_shape(
            type="line",
            x0=min(historical), y0=min(historical),
            x1=max(historical), y1=max(historical),
            line=dict(color="gray", width=2, dash="dash"),
            row=2, col=2
        )

        fig.update_layout(
            title=f'Industry Trend Analysis - {len(industry_data)} Industries',
            height=900,
            showlegend=True
        )

        fig.update_xaxes(tickangle=-45)
        fig.update_xaxes(title_text="Historical Return %", row=2, col=2)
        fig.update_yaxes(title_text="Predicted Return %", row=2, col=2)

        result_text = f"""
Industry Trend Analysis
{'='*80}
Total Industries Analyzed: {len(industry_data)}
Stocks per Industry: {max_stocks_per_industry}
Historical Period: Last 30 days
Prediction Horizon: {prediction_days} days

{'='*80}
HOTTEST INDUSTRIES (Predicted Performance):
{'='*80}
"""

        for idx, row in summary_df.head(10).iterrows():
            trend = "📈 HEATING UP" if row['momentum_shift'] > 0 else "📉 COOLING DOWN"
            result_text += f"""
{row['industry'].upper()}:
  Historical (30d): {row['avg_historical_30d']:+.2f}%
  Predicted ({prediction_days}d): {row['avg_predicted']:+.2f}%
  Momentum Shift: {row['momentum_shift']:+.2f}% {trend}
  Sample Stocks: {', '.join(row['stocks'][:5])}
  Total Stocks: {row['stock_count']}
"""

        result_text += f"\n{'='*80}\n"
        result_text += "Interpretation:\n"
        result_text += "- Green bars in Momentum Shift = Industry gaining strength\n"
        result_text += "- Red bars = Industry losing momentum\n"
        result_text += "- Points above diagonal line (bottom right) = Stronger predicted performance\n"

        return result_text, fig, summary_df

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None, None


def scan_stock_pool(max_stocks, prediction_days, checkpoint_path, min_increase_pct, progress=gr.Progress()):
    """
    Scan stock pool and rank by upward probability

    Returns stocks ranked by predicted increase percentage
    """
    try:
        if not os.path.exists(checkpoint_path):
            return "Error: Checkpoint not found. Please train a model first.", None

        data_dir = "full_stock_data/training_data"
        if not os.path.exists(data_dir):
            return "Error: Data directory not found.", None

        all_stocks = get_available_stocks(randomize=True)
        if not all_stocks:
            return "Error: No stocks available.", None, None

        max_stocks = int(max_stocks)
        prediction_days = int(prediction_days)
        min_increase_pct = float(min_increase_pct)

        if max_stocks >= len(all_stocks):
            stocks_to_scan = all_stocks
            max_stocks = len(all_stocks)
        else:
            stocks_to_scan = all_stocks[:max_stocks]

        predictor = ConditionalPredictor(checkpoint_path)

        results = []

        for idx, stock_code in enumerate(stocks_to_scan):
            progress((idx + 1) / len(stocks_to_scan), desc=f"Scanning {stock_code}")

            try:
                stock_file = os.path.join(data_dir, f"{stock_code}.csv")
                data = pd.read_csv(stock_file)

                if len(data) < 60:
                    continue

                predictions, metadata = predictor.predict_stock(
                    stock_code=stock_code,
                    recent_data=data,
                    future_steps=prediction_days
                )

                current_price = metadata['last_price']
                predicted_price = predictions[-1]
                increase_pct = ((predicted_price - current_price) / current_price) * 100

                if increase_pct >= min_increase_pct:
                    avg_daily_increase = increase_pct / prediction_days

                    upward_days = sum(1 for i in range(len(predictions)) if predictions[i] > (current_price if i == 0 else predictions[i-1]))
                    upward_probability = (upward_days / len(predictions)) * 100

                    results.append({
                        'rank': 0,
                        'stock_code': stock_code,
                        'industry_l1': metadata['industry_l1'],
                        'industry_l2': metadata['industry_l2'],
                        'style': metadata['style'],
                        'regime': metadata['regime'],
                        'current_price': current_price,
                        'predicted_price': predicted_price,
                        'increase_pct': increase_pct,
                        'avg_daily_increase': avg_daily_increase,
                        'upward_probability': upward_probability,
                        'volatility': metadata['volatility']
                    })

            except Exception as e:
                continue

        if not results:
            return "No stocks meet the criteria. Try lowering the minimum increase percentage.", None

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('increase_pct', ascending=False)
        results_df['rank'] = range(1, len(results_df) + 1)

        results_df = results_df[[
            'rank', 'stock_code', 'industry_l1', 'industry_l2', 'style', 'regime',
            'current_price', 'predicted_price', 'increase_pct', 'avg_daily_increase',
            'upward_probability', 'volatility'
        ]]

        summary_text = f"""
Stock Pool Screening Results
{'='*80}
Total Scanned: {len(stocks_to_scan)} stocks
Qualified Stocks: {len(results_df)} stocks
Prediction Horizon: {prediction_days} days
Min Increase Filter: {min_increase_pct}%

Top 10 Recommendations:
{'='*80}
"""

        for idx, row in results_df.head(10).iterrows():
            summary_text += f"""
Rank {row['rank']}: {row['stock_code']}
  Industry: {row['industry_l1']} / {row['industry_l2']}
  Style: {row['style']} | Regime: {row['regime']}
  Current Price: {row['current_price']:.2f}
  Predicted Price ({prediction_days}d): {row['predicted_price']:.2f}
  Expected Gain: {row['increase_pct']:.2f}% (avg {row['avg_daily_increase']:.2f}%/day)
  Upward Probability: {row['upward_probability']:.1f}%
  Volatility: {row['volatility']:.4f}
"""

        summary_text += f"\n{'='*80}\n"
        summary_text += "Full results table displayed below.\n"

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Top 20 by Expected Gain',
                'Industry Distribution',
                'Style Distribution',
                'Market Regime Distribution'
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "pie"}, {"type": "pie"}]
            ]
        )

        top_20 = results_df.head(20)

        fig.add_trace(go.Bar(
            x=top_20['stock_code'],
            y=top_20['increase_pct'],
            name='Expected Gain %',
            marker_color='green',
            text=top_20['increase_pct'].round(2),
            textposition='outside'
        ), row=1, col=1)

        industry_counts = results_df['industry_l1'].value_counts()
        fig.add_trace(go.Pie(
            labels=industry_counts.index,
            values=industry_counts.values,
            name='Industry'
        ), row=1, col=2)

        style_counts = results_df['style'].value_counts()
        fig.add_trace(go.Pie(
            labels=style_counts.index,
            values=style_counts.values,
            name='Style'
        ), row=2, col=1)

        regime_counts = results_df['regime'].value_counts()
        fig.add_trace(go.Pie(
            labels=regime_counts.index,
            values=regime_counts.values,
            name='Regime'
        ), row=2, col=2)

        fig.update_layout(
            title=f'Stock Pool Analysis - {len(results_df)} Qualified Stocks',
            showlegend=False,
            height=800
        )

        fig.update_xaxes(tickangle=-45, row=1, col=1)

        return summary_text, fig, results_df

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None, None


def backtest_model(stock_code, test_days, checkpoint_path):
    """Backtest model on historical data"""
    try:
        stock_code = normalize_stock_code(stock_code)

        if not os.path.exists(checkpoint_path):
            return "Error: Checkpoint not found. Please train a model first.", None

        data_dir = "full_stock_data/training_data"
        stock_file = os.path.join(data_dir, f"{stock_code}.csv")

        if not os.path.exists(stock_file):
            return f"Error: Stock data not found for {stock_code}\n\nTip: Enter 6-digit code only (e.g., 600017 or 000001)", None

        data = pd.read_csv(stock_file)

        if len(data) < 120:
            return f"Error: Not enough data for {stock_code} (need at least 120 rows)", None

        predictor = ConditionalPredictor(checkpoint_path)

        test_days = int(test_days)
        test_days = min(test_days, len(data) - 60)

        backtest_data = data.iloc[-(60 + test_days):].copy()

        predictions = []
        actuals = []
        dates = []

        for i in range(test_days):
            historical_window = backtest_data.iloc[:60 + i]

            pred, metadata = predictor.predict_stock(
                stock_code=stock_code,
                recent_data=historical_window,
                future_steps=1
            )

            actual_price = backtest_data.iloc[60 + i]['close']

            predictions.append(pred[0])
            actuals.append(actual_price)
            dates.append(i + 1)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        mape = np.mean(np.abs((predictions - actuals) / actuals)) * 100

        direction_actual = np.diff(actuals) > 0
        direction_pred = np.diff(predictions) > 0
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=(f'{stock_code} Backtest: Predicted vs Actual', 'Prediction Error')
        )

        fig.add_trace(go.Scatter(
            x=dates,
            y=actuals,
            mode='lines+markers',
            name='Actual Price',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='red', width=2, dash='dash'),
            marker=dict(size=6)
        ), row=1, col=1)

        errors = predictions - actuals
        colors = ['red' if e > 0 else 'green' for e in errors]

        fig.add_trace(go.Bar(
            x=dates,
            y=errors,
            name='Prediction Error',
            marker_color=colors,
            showlegend=False
        ), row=2, col=1)

        fig.update_layout(
            title=f'{stock_code} Backtesting Results ({test_days} days)',
            xaxis2_title='Day',
            yaxis_title='Price',
            yaxis2_title='Error',
            hovermode='x unified',
            template='plotly_white',
            height=800
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        result = f"""
Backtesting Results
{'='*60}
Stock Code: {stock_code}
Test Period: {test_days} days
Model: {checkpoint_path}

Performance Metrics:
{'='*60}
RMSE (Root Mean Square Error): {rmse:.4f}
MAE (Mean Absolute Error): {mae:.4f}
MAPE (Mean Absolute Percentage Error): {mape:.2f}%
Direction Accuracy: {direction_accuracy:.2f}%

Price Statistics:
{'='*60}
Actual Price Range: {actuals.min():.2f} - {actuals.max():.2f}
Predicted Price Range: {predictions.min():.2f} - {predictions.max():.2f}
Mean Actual Price: {actuals.mean():.2f}
Mean Predicted Price: {predictions.mean():.2f}

Error Statistics:
{'='*60}
Max Overestimation: {errors.max():.4f} ({errors.max()/actuals.mean()*100:.2f}%)
Max Underestimation: {errors.min():.4f} ({errors.min()/actuals.mean()*100:.2f}%)
Mean Error: {errors.mean():.4f}

Interpretation:
{'='*60}
- RMSE closer to 0 is better (measures overall error magnitude)
- MAE shows average absolute error in price units
- MAPE shows percentage error (< 5% is good, < 10% is acceptable)
- Direction Accuracy > 50% means better than random guess
"""

        return result, fig

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None


def predict_single_stock(stock_code, future_days, checkpoint_path):
    """Predict future prices for a single stock with K-line chart and volume"""
    try:
        stock_code = normalize_stock_code(stock_code)

        if not os.path.exists(checkpoint_path):
            return "Error: Checkpoint not found. Please train a model first.", None

        data_dir = "full_stock_data/training_data"
        stock_file = os.path.join(data_dir, f"{stock_code}.csv")

        if not os.path.exists(stock_file):
            return f"Error: Stock data not found for {stock_code}\n\nTip: Enter 6-digit code only (e.g., 600017 or 000001)", None

        data = pd.read_csv(stock_file)

        if len(data) < 60:
            return f"Error: Not enough data for {stock_code} (need at least 60 rows)", None

        predictor = ConditionalPredictor(checkpoint_path)

        predictions, metadata = predictor.predict_stock(
            stock_code=stock_code,
            recent_data=data,
            future_steps=int(future_days)
        )

        historical_data = data.tail(60).copy()
        historical_data['date_index'] = list(range(-len(historical_data), 0))

        predicted_ohlcv = metadata['predicted_ohlcv']
        future_dates = list(range(1, len(predictions) + 1))
        predicted_ohlcv['date_index'] = future_dates

        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.03,
            subplot_titles=(f'{stock_code} K-Line Chart with Predictions', 'Volume'),
            shared_xaxes=True
        )

        fig.add_trace(go.Candlestick(
            x=historical_data['date_index'],
            open=historical_data['open'],
            high=historical_data['high'],
            low=historical_data['low'],
            close=historical_data['close'],
            name='Historical K-Line',
            increasing_line_color='red',
            decreasing_line_color='green',
            showlegend=True
        ), row=1, col=1)

        fig.add_trace(go.Candlestick(
            x=predicted_ohlcv['date_index'],
            open=predicted_ohlcv['open'],
            high=predicted_ohlcv['high'],
            low=predicted_ohlcv['low'],
            close=predicted_ohlcv['close'],
            name='Predicted K-Line',
            increasing_line_color='rgba(255, 140, 0, 0.6)',
            decreasing_line_color='rgba(0, 200, 100, 0.6)',
            showlegend=True
        ), row=1, col=1)

        last_close = historical_data['close'].iloc[-1]
        fig.add_trace(go.Scatter(
            x=[0] + future_dates,
            y=[last_close] + list(predictions),
            mode='lines+markers',
            name='Predicted Close Line',
            line=dict(color='orange', width=2, dash='dot'),
            marker=dict(size=8, symbol='star'),
            showlegend=True
        ), row=1, col=1)

        hist_colors = ['red' if historical_data['close'].iloc[i] >= historical_data['open'].iloc[i]
                       else 'green' for i in range(len(historical_data))]

        fig.add_trace(go.Bar(
            x=historical_data['date_index'],
            y=historical_data['volume'],
            name='Historical Volume',
            marker_color=hist_colors,
            showlegend=False,
            opacity=0.7
        ), row=2, col=1)

        pred_colors = ['rgba(255, 140, 0, 0.5)' if predicted_ohlcv['close'].iloc[i] >= predicted_ohlcv['open'].iloc[i]
                       else 'rgba(0, 200, 100, 0.5)' for i in range(len(predicted_ohlcv))]

        fig.add_trace(go.Bar(
            x=predicted_ohlcv['date_index'],
            y=predicted_ohlcv['volume'],
            name='Predicted Volume',
            marker_color=pred_colors,
            showlegend=False,
            opacity=0.7
        ), row=2, col=1)

        fig.update_layout(
            title=f'{stock_code} Price Prediction with K-Line',
            xaxis2_title='Days (0=Today)',
            yaxis_title='Price',
            yaxis2_title='Volume',
            hovermode='x unified',
            template='plotly_white',
            height=800,
            xaxis_rangeslider_visible=False
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

        result = f"""
Stock Prediction Results
========================
Stock Code: {metadata['stock_code']}
Industry: {metadata['industry_l1']} / {metadata['industry_l2']}
Style: {metadata['style']} (Volatility: {metadata['volatility']:.4f})
Market Regime: {metadata['regime']}

Current Price: {metadata['last_price']:.2f}

Future Predictions:
"""
        for i, pred in enumerate(predictions, 1):
            change = ((pred - metadata['last_price']) / metadata['last_price']) * 100
            result += f"  Day {i}: {pred:.2f} ({change:+.2f}%)\n"

        return result, fig

    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n\n{traceback.format_exc()}", None

with gr.Blocks(title="A-Share Multi-Dimensional Financial Model", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 A股多维度金融模型训练系统 / A-Share Multi-Dimensional Financial Model")
    gr.Markdown("### 基于 PyTorch + Flash Attention 2.8.2 + CUDA 12.8 / Powered by PyTorch + Flash Attention 2.8.2 + CUDA 12.8")

    with gr.Row():
        flash_status_box = gr.Textbox(value=check_flash_attention(), label="Flash Attention 状态 / Status", interactive=False)
        data_status_box = gr.Textbox(value=check_data_status(), label="数据集状态 / Dataset Status", interactive=False, lines=8)

    with gr.Tabs():
        with gr.Tab("📥 数据下载 / Data Download"):
            gr.Markdown("""
            ## 📥 A股历史数据下载 / A-Share Historical Data Download

            **数据源 / Data Source:** 通达信TDX服务器 / TDX Server (124.71.187.122:7709)

            **数据格式 / Data Format:**
            - 日期 / Date
            - 开盘价 / Open
            - 最高价 / High
            - 最低价 / Low
            - 收盘价 / Close
            - 成交量 / Volume

            **覆盖范围 / Coverage:**
            - 上海A股 / Shanghai A-Share: 600xxx, 601xxx, 603xxx, 605xxx, 688xxx
            - 深圳A股 / Shenzhen A-Share: 000xxx, 001xxx, 002xxx, 003xxx, 300xxx, 301xxx
            """)

            with gr.Row():
                with gr.Column():
                    download_max_stocks = gr.Slider(
                        10, 5000, value=100, step=10,
                        label="下载股票数量 / Number of Stocks",
                        info="建议先下载100只测试 / Recommend 100 for testing"
                    )
                    download_start_date = gr.Textbox(
                        value="1999-01-01",
                        label="起始日期 / Start Date",
                        info="格式 / Format: YYYY-MM-DD"
                    )
                    inspect_download_btn = gr.Button(
                        "Check Existing Data",
                        variant="secondary",
                    )
                    download_btn = gr.Button("🚀 开始下载 / Start Download", variant="primary", size="lg")
                    refresh_status_btn = gr.Button("🔄 刷新状态 / Refresh Status", variant="secondary")

                with gr.Column():
                    download_output = gr.Textbox(
                        label="下载日志 / Download Log",
                        lines=20,
                        max_lines=25
                    )

            inspect_download_btn.click(
                fn=inspect_download_plan,
                inputs=[download_max_stocks, download_start_date],
                outputs=download_output,
            )

            download_btn.click(
                fn=download_stock_data,
                inputs=[download_max_stocks, download_start_date],
                outputs=download_output
            )

            refresh_status_btn.click(
                fn=check_data_status,
                outputs=data_status_box
            )

        with gr.Tab("🎯 Basic Training"):
            gr.Markdown("""
            ## 🎯 Single Stock Model Training

            **Train a basic model on real stock data**

            **Features:**
            - Uses real stock data from `full_stock_data/training_data/`
            - Train on single stock OHLCV data
            - Real-time training curve visualization
            - Supports Transformer (with Flash Attention) or LSTM

            **Training Process:**
            1. Load real stock data (OHLCV)
            2. Normalize data (StandardScaler)
            3. Create sequences (60 days → predict 1 day)
            4. Train with AdamW optimizer
            5. Early stopping (patience=10)

            **Model Architecture:**
            - **Transformer**: Multi-head attention with Flash Attention acceleration
            - **LSTM**: Traditional recurrent network, faster training

            **Data Split:**
            - Train: 70% | Validation: 15% | Test: 15%
            """)

            with gr.Row():
                with gr.Column():
                    model_type = gr.Radio(
                        ["Transformer", "LSTM"],
                        value="Transformer",
                        label="模型类型 / Model Type",
                        info="Transformer使用Flash Attention / Transformer uses Flash Attention"
                    )
                    hidden_dim = gr.Slider(
                        64, 512, value=256, step=64,
                        label="隐藏层维度 / Hidden Dimension",
                        info="更大=更强表达能力 / Larger = More capacity"
                    )
                    num_layers = gr.Slider(
                        1, 8, value=4, step=1,
                        label="网络层数 / Number of Layers",
                        info="更深=更复杂模式 / Deeper = More complex patterns"
                    )
                    num_heads = gr.Slider(
                        2, 16, value=8, step=2,
                        label="注意力头数 / Attention Heads",
                        info="仅Transformer / Transformer only"
                    )

                with gr.Column():
                    batch_size = gr.Slider(
                        8, 128, value=32, step=8,
                        label="批次大小 / Batch Size",
                        info="RTX 5090建议64-128 / RTX 5090: 64-128"
                    )
                    epochs = gr.Slider(
                        1, 100, value=10, step=1,
                        label="训练轮数 / Epochs",
                        info="通常10-50轮足够 / Usually 10-50 is enough"
                    )
                    lr = gr.Number(
                        value=0.001,
                        label="Learning Rate",
                        info="Default 0.001, fine-tune 0.0001"
                    )

                    available_stocks = get_available_stocks()
                    if not available_stocks:
                        available_stocks = ["No stocks available - Please download data first"]

                    stock_code_dropdown = gr.Dropdown(
                        choices=available_stocks,
                        value=available_stocks[0] if available_stocks else None,
                        label="Stock Code",
                        info=f"Select from {len(available_stocks)} available stocks",
                        allow_custom_value=False
                    )

                    stock_info_display = gr.Textbox(
                        label="Stock Data Info",
                        lines=10,
                        interactive=False,
                        value=get_stock_info(available_stocks[0]) if available_stocks else ""
                    )

            stock_code_dropdown.change(
                fn=get_stock_info,
                inputs=[stock_code_dropdown],
                outputs=[stock_info_display]
            )

            train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

            with gr.Row():
                with gr.Column():
                    train_output = gr.Textbox(label="Training Log", lines=15, max_lines=20)
                with gr.Column():
                    plot_output = gr.Plot(label="Real-time Training Curve")

            train_btn.click(
                fn=train_model,
                inputs=[model_type, hidden_dim, num_layers, num_heads, batch_size, epochs, lr, stock_code_dropdown],
                outputs=[train_output, plot_output]
            )

        with gr.Tab("🔮 Inference"):
            gr.Markdown("## Model Inference")

            with gr.Row():
                with gr.Column():
                    available_checkpoints_infer = get_available_checkpoints()
                    default_ckpt_infer = "financial_model/checkpoints/best_model.pt"
                    if default_ckpt_infer not in available_checkpoints_infer:
                        available_checkpoints_infer = [
                            default_ckpt_infer,
                            *[p for p in available_checkpoints_infer if p != default_ckpt_infer],
                        ]

                    checkpoint_path = gr.Dropdown(
                        choices=available_checkpoints_infer,
                        value=available_checkpoints_infer[0] if available_checkpoints_infer else default_ckpt_infer,
                        label="Checkpoint Path",
                        info="Select an existing checkpoint or type a custom path",
                        allow_custom_value=True,
                    )

                    refresh_ckpt_infer_btn = gr.Button("🔄 Refresh Checkpoint List", variant="secondary", size="sm")

                with gr.Column():
                    test_samples = gr.Slider(100, 1000, value=200, step=50, label="Test Samples")

            inference_btn = gr.Button("🔮 Run Inference", variant="primary", size="lg")

            with gr.Row():
                inference_output = gr.Textbox(label="Inference Results", lines=15)

            with gr.Row():
                inference_plot = gr.Plot(label="Predictions vs Actual")

            refresh_ckpt_infer_btn.click(
                fn=lambda: gr.update(choices=get_available_checkpoints()),
                outputs=checkpoint_path
            )

            inference_btn.click(
                fn=run_inference,
                inputs=[checkpoint_path, test_samples],
                outputs=[inference_output, inference_plot]
            )

        with gr.Tab("📈 Single Stock Prediction"):
            gr.Markdown("""
            ## 📈 Single Stock Future Prediction

            **Predict future prices for a specific stock using the trained conditional model**

            Features:
            - Multi-day future prediction (1-30 days)
            - Industry, style, and market regime aware
            - Visual comparison with historical prices
            - Percentage change analysis
            """)

            with gr.Row():
                with gr.Column():
                    available_stocks_predict = get_available_stocks()
                    if not available_stocks_predict:
                        available_stocks_predict = ["No stocks available"]

                    with gr.Row():
                        stock_code_predict_dropdown = gr.Textbox(
                            value=available_stocks_predict[0] if available_stocks_predict else "",
                            label="Stock Code (Smart Input)",
                            info=f"Just type 6 digits + Enter / 只输入6位数字+回车 (e.g., 600017, 000001) | {len(available_stocks_predict)} stocks",
                            placeholder="600017, 000001, 300001...",
                            scale=3
                        )

                        stock_code_selector = gr.Dropdown(
                            choices=available_stocks_predict,
                            value=available_stocks_predict[0] if available_stocks_predict else None,
                            label="Or Select",
                            info="Quick select",
                            scale=2,
                            allow_custom_value=False
                        )

                    stock_info_predict_display = gr.Textbox(
                        label="Stock Data Info",
                        lines=8,
                        interactive=False,
                        value=get_stock_info(available_stocks_predict[0]) if available_stocks_predict else ""
                    )

                    future_days_input = gr.Number(
                        value=5,
                        minimum=1,
                        maximum=30,
                        step=1,
                        label="Future Days to Predict",
                        info="Press Enter to predict / 按回车键快速推理"
                    )

                    available_checkpoints_single = get_available_checkpoints()
                    default_ckpt_single = "financial_model/checkpoints/conditional_best_model.pt"
                    if default_ckpt_single not in available_checkpoints_single:
                        available_checkpoints_single = [
                            default_ckpt_single,
                            *[p for p in available_checkpoints_single if p != default_ckpt_single],
                        ]

                    checkpoint_path_single = gr.Dropdown(
                        choices=available_checkpoints_single,
                        value=available_checkpoints_single[0] if available_checkpoints_single else default_ckpt_single,
                        label="Checkpoint Path",
                        info="Path to trained conditional model (select or type)",
                        allow_custom_value=True,
                    )

                    refresh_ckpt_single_btn = gr.Button("🔄 Refresh Checkpoint List", variant="secondary", size="sm")

                with gr.Column():
                    predict_single_btn = gr.Button("🔮 Predict Future", variant="primary", size="lg")

            stock_code_predict_dropdown.change(
                fn=get_stock_info,
                inputs=[stock_code_predict_dropdown],
                outputs=[stock_info_predict_display]
            )

            stock_code_selector.change(
                fn=lambda x: (x, get_stock_info(x)),
                inputs=[stock_code_selector],
                outputs=[stock_code_predict_dropdown, stock_info_predict_display]
            )

            refresh_ckpt_single_btn.click(
                fn=lambda: gr.update(choices=get_available_checkpoints()),
                outputs=checkpoint_path_single
            )

            with gr.Row():
                single_stock_plot = gr.Plot(label="K-Line Chart with Predictions")

            with gr.Row():
                single_stock_output = gr.Textbox(label="Prediction Results", lines=15)

            predict_single_btn.click(
                fn=predict_single_stock,
                inputs=[stock_code_predict_dropdown, future_days_input, checkpoint_path_single],
                outputs=[single_stock_output, single_stock_plot]
            )

            stock_code_predict_dropdown.submit(
                fn=predict_single_stock,
                inputs=[stock_code_predict_dropdown, future_days_input, checkpoint_path_single],
                outputs=[single_stock_output, single_stock_plot]
            )

            future_days_input.submit(
                fn=predict_single_stock,
                inputs=[stock_code_predict_dropdown, future_days_input, checkpoint_path_single],
                outputs=[single_stock_output, single_stock_plot]
            )

        with gr.Tab("🎯 Stock Pool / 智能选股"):
            gr.Markdown("""
            ## 🎯 Intelligent Stock Screening / 智能股票池

            **Automatically scan and rank stocks by upward potential**

            **功能说明：**
            - 批量扫描股票池，自动预测每只股票的未来走势
            - 按预期涨幅排序，筛选出最具潜力的股票
            - 显示行业、风格、市场环境等多维度信息
            - 生成可视化分析报告

            **How it works:**
            1. Scan specified number of stocks from your pool
            2. Predict future price for each stock (configurable horizon)
            3. Calculate expected gain percentage
            4. Rank stocks by expected return
            5. Filter by minimum gain threshold

            **Use Cases:**
            - **Daily Stock Selection**: Find best opportunities each morning
            - **Portfolio Construction**: Build diversified portfolio
            - **Sector Rotation**: Identify hot sectors
            - **Risk Management**: Avoid stocks with predicted decline
            """)

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        scan_max_stocks = gr.Number(
                            value=100,
                            minimum=10,
                            maximum=5000,
                            step=10,
                            label="Max Stocks to Scan / 扫描股票数",
                            info="More stocks = better coverage but slower (Recommended: 100-500, Max: 5000)",
                            scale=3
                        )

                        scan_all_btn = gr.Button("📊 Scan ALL Stocks", variant="secondary", size="sm", scale=1)

                    scan_prediction_days = gr.Number(
                        value=5,
                        minimum=1,
                        maximum=20,
                        step=1,
                        label="Prediction Horizon (days) / 预测周期",
                        info="How many days ahead to predict"
                    )

                    scan_min_increase = gr.Number(
                        value=2.0,
                        minimum=0.0,
                        maximum=20.0,
                        step=0.5,
                        label="Min Expected Gain (%) / 最低预期涨幅",
                        info="Filter out stocks below this threshold"
                    )

                    available_checkpoints_pool = get_available_checkpoints()
                    default_ckpt_pool = "checkpoints/multi_dim_model_20251115_014201_best.pt"
                    if default_ckpt_pool not in available_checkpoints_pool:
                        available_checkpoints_pool = [
                            default_ckpt_pool,
                            *[p for p in available_checkpoints_pool if p != default_ckpt_pool],
                        ]

                    checkpoint_path_pool = gr.Dropdown(
                        choices=available_checkpoints_pool,
                        value=available_checkpoints_pool[0] if available_checkpoints_pool else default_ckpt_pool,
                        label="Checkpoint Path",
                        info="Path to trained conditional model",
                        allow_custom_value=True,
                    )

                    refresh_ckpt_pool_btn = gr.Button("🔄 Refresh Checkpoint List", variant="secondary", size="sm")

                with gr.Column():
                    scan_btn = gr.Button("🎯 Scan Stock Pool", variant="primary", size="lg")
                    gr.Markdown("""
                    **Tips:**
                    - Start with 100 stocks for quick results
                    - Use 5-day horizon for short-term trading
                    - Set min gain to 2-5% to filter noise
                    - Larger scans take longer but find more opportunities
                    """)

            with gr.Row():
                pool_plot = gr.Plot(label="Stock Pool Analysis")

            with gr.Row():
                pool_output = gr.Textbox(label="Screening Results", lines=20)

            with gr.Row():
                pool_table = gr.Dataframe(
                    label="Full Results Table (Sortable)",
                    headers=[
                        'Rank', 'Stock Code', 'Industry L1', 'Industry L2', 'Style', 'Regime',
                        'Current Price', 'Predicted Price', 'Increase %', 'Avg Daily %',
                        'Upward Prob %', 'Volatility'
                    ],
                    interactive=False,
                    wrap=True
                )

            refresh_ckpt_pool_btn.click(
                fn=lambda: gr.update(choices=get_available_checkpoints()),
                outputs=checkpoint_path_pool
            )

            scan_all_btn.click(
                fn=lambda pred_days, ckpt, min_inc: scan_stock_pool(
                    len(get_available_stocks()),
                    pred_days,
                    ckpt,
                    min_inc
                ),
                inputs=[scan_prediction_days, checkpoint_path_pool, scan_min_increase],
                outputs=[pool_output, pool_plot, pool_table]
            )

            scan_btn.click(
                fn=scan_stock_pool,
                inputs=[scan_max_stocks, scan_prediction_days, checkpoint_path_pool, scan_min_increase],
                outputs=[pool_output, pool_plot, pool_table]
            )

        with gr.Tab("📊 Industry Analysis / 行业分析"):
            gr.Markdown("""
            ## 📊 Industry Trend Analysis / 行业趋势分析

            **Compare industry performance - Historical vs Predicted**

            **功能说明：**
            - 对比各行业的历史表现和未来预测
            - 识别当前热门行业和未来潜力行业
            - 发现行业轮动机会
            - 支持行业配置决策

            **How it works:**
            1. Sample stocks from each industry
            2. Calculate average historical return (last 30 days)
            3. Predict average future return (configurable horizon)
            4. Compare momentum shift between historical and predicted

            **Key Metrics:**
            - **Historical Return**: Past 30-day average performance
            - **Predicted Return**: Future N-day average performance
            - **Momentum Shift**: Predicted - Historical (positive = heating up)

            **Use Cases:**
            - **Sector Rotation**: Switch from cooling sectors to heating sectors
            - **Portfolio Rebalancing**: Overweight hot industries
            - **Risk Management**: Avoid industries with negative momentum
            - **Market Timing**: Understand overall market sentiment by industries
            """)

            with gr.Row():
                with gr.Column():
                    industry_stocks_per = gr.Number(
                        value=10,
                        minimum=5,
                        maximum=50,
                        step=5,
                        label="Stocks per Industry / 每行业股票数",
                        info="More stocks = more accurate but slower (5-20 recommended)"
                    )

                    industry_pred_days = gr.Number(
                        value=7,
                        minimum=1,
                        maximum=20,
                        step=1,
                        label="Prediction Horizon (days) / 预测周期",
                        info="How many days ahead to predict"
                    )

                    available_checkpoints_industry = get_available_checkpoints()
                    default_ckpt_industry = "checkpoints/multi_dim_model_20251115_014201_best.pt"
                    if default_ckpt_industry not in available_checkpoints_industry:
                        available_checkpoints_industry = [
                            default_ckpt_industry,
                            *[p for p in available_checkpoints_industry if p != default_ckpt_industry],
                        ]

                    checkpoint_path_industry = gr.Dropdown(
                        choices=available_checkpoints_industry,
                        value=available_checkpoints_industry[0] if available_checkpoints_industry else default_ckpt_industry,
                        label="Checkpoint Path",
                        info="Path to trained conditional model",
                        allow_custom_value=True,
                    )

                    refresh_ckpt_industry_btn = gr.Button("🔄 Refresh Checkpoint List", variant="secondary", size="sm")

                with gr.Column():
                    industry_analyze_btn = gr.Button("📊 Analyze Industries", variant="primary", size="lg")
                    gr.Markdown("""
                    **Tips:**
                    - 10 stocks per industry is usually sufficient
                    - 7-day prediction balances accuracy and relevance
                    - Analysis covers all major industries
                    - Look for green bars in Momentum Shift chart
                    """)

            refresh_ckpt_industry_btn.click(
                fn=lambda: gr.update(choices=get_available_checkpoints()),
                outputs=checkpoint_path_industry
            )

            with gr.Row():
                industry_plot = gr.Plot(label="Industry Trend Comparison")

            with gr.Row():
                industry_output = gr.Textbox(label="Analysis Results", lines=20)

            with gr.Row():
                industry_table = gr.Dataframe(
                    label="Industry Performance Table",
                    headers=['Industry', 'Stock Count', 'Historical 30d %', 'Predicted %', 'Momentum Shift %', 'Sample Stocks'],
                    interactive=False,
                    wrap=True
                )

            industry_analyze_btn.click(
                fn=analyze_industry_trends,
                inputs=[industry_stocks_per, industry_pred_days, checkpoint_path_industry],
                outputs=[industry_output, industry_plot, industry_table]
            )

        with gr.Tab("📊 Backtesting"):
            gr.Markdown("""
            ## 📊 Historical Backtesting / 历史回测

            **Test your model's performance on historical data**

            **功能说明：**
            - 在历史数据上验证模型准确性
            - 使用滑动窗口逐日预测，模拟真实交易场景
            - 计算多种性能指标评估模型质量

            **How it works:**
            1. Select a stock and test period (e.g., last 30 days)
            2. Model predicts each day using only data available up to that point
            3. Compare predictions with actual prices
            4. Calculate performance metrics

            **Metrics Explained:**
            - **RMSE**: Overall prediction error (lower is better)
            - **MAE**: Average absolute error in price units
            - **MAPE**: Percentage error (< 5% excellent, < 10% good)
            - **Direction Accuracy**: % of correct up/down predictions (> 50% beats random)
            """)

            with gr.Row():
                with gr.Column():
                    available_stocks_backtest = get_available_stocks()
                    if not available_stocks_backtest:
                        available_stocks_backtest = ["No stocks available"]

                    with gr.Row():
                        stock_code_backtest = gr.Textbox(
                            value=available_stocks_backtest[0] if available_stocks_backtest else "",
                            label="Stock Code (Smart Input)",
                            info=f"Type 6 digits + Enter / 输入6位数字+回车 | {len(available_stocks_backtest)} stocks",
                            placeholder="600017, 000001, 300001...",
                            scale=3
                        )

                        stock_code_backtest_selector = gr.Dropdown(
                            choices=available_stocks_backtest,
                            value=available_stocks_backtest[0] if available_stocks_backtest else None,
                            label="Or Select",
                            info="Quick select",
                            scale=2,
                            allow_custom_value=False
                        )

                    test_days_input = gr.Number(
                        value=30,
                        minimum=5,
                        maximum=100,
                        step=1,
                        label="Test Days / 回测天数",
                        info="Press Enter to run backtest / 按回车开始回测"
                    )

                    available_checkpoints_backtest = get_available_checkpoints()
                    default_ckpt_backtest = "checkpoints/multi_dim_model_20251115_014201_best.pt"
                    if default_ckpt_backtest not in available_checkpoints_backtest:
                        available_checkpoints_backtest = [
                            default_ckpt_backtest,
                            *[p for p in available_checkpoints_backtest if p != default_ckpt_backtest],
                        ]

                    checkpoint_path_backtest = gr.Dropdown(
                        choices=available_checkpoints_backtest,
                        value=available_checkpoints_backtest[0] if available_checkpoints_backtest else default_ckpt_backtest,
                        label="Checkpoint Path",
                        info="Path to trained conditional model",
                        allow_custom_value=True,
                    )

                    refresh_ckpt_backtest_btn = gr.Button("🔄 Refresh Checkpoint List", variant="secondary", size="sm")

                with gr.Column():
                    backtest_btn = gr.Button("📊 Run Backtest", variant="primary", size="lg")

            stock_code_backtest_selector.change(
                fn=lambda x: x,
                inputs=[stock_code_backtest_selector],
                outputs=[stock_code_backtest]
            )

            refresh_ckpt_backtest_btn.click(
                fn=lambda: gr.update(choices=get_available_checkpoints()),
                outputs=checkpoint_path_backtest
            )

            with gr.Row():
                backtest_plot = gr.Plot(label="Backtest Results: Predicted vs Actual")

            with gr.Row():
                backtest_output = gr.Textbox(label="Performance Metrics", lines=25)

            backtest_btn.click(
                fn=backtest_model,
                inputs=[stock_code_backtest, test_days_input, checkpoint_path_backtest],
                outputs=[backtest_output, backtest_plot]
            )

            stock_code_backtest.submit(
                fn=backtest_model,
                inputs=[stock_code_backtest, test_days_input, checkpoint_path_backtest],
                outputs=[backtest_output, backtest_plot]
            )

            test_days_input.submit(
                fn=backtest_model,
                inputs=[stock_code_backtest, test_days_input, checkpoint_path_backtest],
                outputs=[backtest_output, backtest_plot]
            )

        with gr.Tab("📁 Custom Data"):
            gr.Markdown("## Upload Your Data for Prediction")
            gr.Markdown("CSV format: `date,open,high,low,close,volume`")

            with gr.Row():
                with gr.Column():
                    file_input = gr.File(label="Upload CSV File", file_types=[".csv"])

                with gr.Column():
                    available_checkpoints_custom = get_available_checkpoints()
                    default_ckpt_custom = "financial_model/checkpoints/best_model.pt"
                    if default_ckpt_custom not in available_checkpoints_custom:
                        available_checkpoints_custom = [
                            default_ckpt_custom,
                            *[p for p in available_checkpoints_custom if p != default_ckpt_custom],
                        ]

                    checkpoint_path_custom = gr.Dropdown(
                        choices=available_checkpoints_custom,
                        value=available_checkpoints_custom[0] if available_checkpoints_custom else default_ckpt_custom,
                        label="Checkpoint Path",
                        info="Select an existing checkpoint or type a custom path",
                        allow_custom_value=True,
                    )

                    refresh_ckpt_custom_btn = gr.Button("🔄 Refresh Checkpoint List", variant="secondary", size="sm")

            predict_btn = gr.Button("📊 Predict", variant="primary", size="lg")

            with gr.Row():
                custom_output = gr.Textbox(label="Results", lines=10)
                download_output = gr.File(label="Download Predictions")

            refresh_ckpt_custom_btn.click(
                fn=lambda: gr.update(choices=get_available_checkpoints()),
                outputs=checkpoint_path_custom
            )

            predict_btn.click(
                fn=upload_data_and_predict,
                inputs=[file_input, checkpoint_path_custom],
                outputs=[custom_output, download_output]
            )

        with gr.Tab("🎨 多维度训练 / Multi-Dimensional Training"):
            gr.Markdown("""
            ## 🎨 多维度条件化模型训练 / Multi-Dimensional Conditional Model Training

            **核心思想 / Core Concept:**
            不同行业、不同风格、不同市场环境下的股票，其价格驱动逻辑完全不同。
            Stocks in different industries, styles, and market regimes have completely different price dynamics.

            **三大维度 / Three Dimensions:**

            1. **行业板块 / Industry Sector (11类 / 11 categories)**
               - 金融 / Finance: 银行、保险、券商 / Banks, Insurance, Securities
               - 消费 / Consumer: 白酒、食品、零售 / Liquor, Food, Retail
               - 科技 / Technology: 芯片、软件、通信 / Chips, Software, Telecom
               - 医药 / Healthcare: 制药、医疗器械 / Pharma, Medical Devices
               - 工业 / Industrial: 机械、制造 / Machinery, Manufacturing
               - 材料 / Materials: 化工、建材 / Chemicals, Construction Materials
               - 能源 / Energy: 煤炭、石油 / Coal, Oil
               - 公用事业 / Utilities: 电力、水务 / Power, Water
               - 房地产 / Real Estate: 地产、物业 / Property, Property Management
               - 电信 / Telecom: 运营商 / Carriers
               - 其他 / Other

            2. **风格因子 / Style Factors (5类 / 5 categories)**
               - 市值 / Market Cap: 超大/大/中/小/微 / Mega/Large/Mid/Small/Micro
               - 估值 / Valuation: 深度价值/价值/平衡/成长 / Deep Value/Value/Balanced/Growth
               - 波动率 / Volatility: 低/中/高 / Low/Medium/High
               - 动量 / Momentum: 强势/正/中性/负/反转 / Strong/Positive/Neutral/Negative/Reversal

            3. **市场环境 / Market Regime (4种 / 4 states)**
               - 牛市 / Bull: 趋势向上 / Uptrend
               - 熊市 / Bear: 趋势向下 / Downtrend
               - 震荡 / Sideways: 横盘整理 / Consolidation
               - 高波动 / Volatile: 剧烈波动 / High volatility

            **模型架构 / Model Architecture:**
            ```
            输入序列 (OHLCV) / Input Sequence
                ↓
            [共享CNN骨干网 / Shared CNN Backbone] → 提取通用技术形态 / Extract patterns
                ↓
            [共享LSTM骨干网 / Shared LSTM Backbone] → 学习时序依赖 / Learn temporal dependencies
                ↓
            [骨干特征 / Backbone Features] ────────┐
                                                  │
            [行业嵌入 / Industry Embedding (32维)] ────┤
            [风格嵌入 / Style Embedding (16维)] ────────┤→ [融合层 / Fusion] → [预测头 / Prediction Head]
            [环境嵌入 / Regime Embedding (16维)] ───────┘
            ```

            **训练模式 / Training Modes:**

            1. **统一条件化模型 / Unified Conditional Model (推荐 / RECOMMENDED)**
               - 一个模型学习所有行业 / One model learns all industries
               - 通过条件输入区分不同类型股票 / Distinguishes stock types via conditional inputs
               - 优势 / Advantages:
                 * 效率高 / High efficiency: 只需训练一个模型 / Train only one model
                 * 性能好 / Better performance: 学习跨行业共性 / Learns cross-industry patterns
                 * 泛化强 / Strong generalization: 自动建立维度间关联 / Auto-learns dimension correlations

            2. **单行业模型 / Single Industry Model**
               - 针对特定行业训练专门模型 / Train specialized model for specific industry
               - 适用场景 / Use cases:
                 * 行业特定策略 / Industry-specific strategies
                 * 深度行业研究 / Deep industry research

            **数据要求 / Data Requirements:**
            - 数据源 / Source: `full_stock_data/training_data/`
            - 格式 / Format: CSV (date, open, high, low, close, volume)
            - 最少股票数 / Min stocks: 10 (测试 / testing)
            - 推荐股票数 / Recommended: 100-500 (生产 / production)
            """)

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 🚀 缓存构建 / Cache Builder")
                    gr.Markdown("""
                    **为什么需要缓存？/ Why Cache?**
                    - ⚡ 404x faster regime detection (vectorized)
                    - 🚀 1.7x faster data loading
                    - 💾 Pre-computed normalization
                    - ✅ Consistent preprocessing

                    **使用流程 / Workflow:**
                    1. 构建缓存 / Build cache (一次性 / one-time)
                    2. 训练时勾选"使用缓存" / Check "Use Cache" when training
                    """)

                    cache_max_stocks = gr.Slider(
                        10, 5000, value=100, step=10,
                        label="缓存股票数 / Stocks to Cache",
                        info="建议与训练股票数一致 / Should match training stocks"
                    )

                    cache_use_gpu = gr.Checkbox(
                        value=True,
                        label="使用 GPU 加速 / Use GPU Acceleration",
                        info="GPU 快 10-20x / GPU is 10-20x faster"
                    )

                    build_cache_btn = gr.Button("🚀 构建缓存 / Build Cache", variant="primary", size="lg")

                    with gr.Row():
                        cache_output = gr.Textbox(label="缓存构建日志 / Cache Build Log", lines=15)

                    with gr.Row():
                        cache_plot = gr.Plot(label="Cache Building Progress")

                with gr.Column():
                    gr.Markdown("### 📊 股票池分析 / Universe Analysis")
                    analyze_max_stocks = gr.Slider(
                        10, 5000, value=100, step=10,
                        label="分析股票数 / Max Stocks to Analyze",
                        info="分析行业分布和风格特征 / Analyze industry distribution and style factors"
                    )
                    analyze_btn = gr.Button("📊 分析股票池 / Analyze Universe", variant="secondary", size="lg")

                    with gr.Row():
                        analysis_output = gr.Textbox(label="分析结果 / Analysis Results", lines=15)

                    with gr.Row():
                        analysis_plot = gr.Plot(label="Industry & Style Distribution")

                with gr.Column():
                    gr.Markdown("### ⚙️ 训练配置 / Training Configuration")

                    training_mode = gr.Radio(
                        ["Unified Conditional Model", "Single Industry Model"],
                        value="Unified Conditional Model",
                        label="训练模式 / Training Mode",
                        info="统一模型(推荐) vs 单行业模型 / Unified (Recommended) vs Single Industry"
                    )

                    industry_filter = gr.Dropdown(
                        ["all", "finance", "consumer", "technology", "healthcare", "industrial",
                         "materials", "energy", "utilities", "real_estate", "telecom", "other"],
                        value="all",
                        label="行业筛选 / Industry Filter",
                        info="仅单行业模式有效 / Only for Single Industry mode"
                    )

                    multi_max_stocks = gr.Slider(
                        10, 5000, value=100, step=10,
                        label="最大股票数 / Max Stocks",
                        info="测试:10-50, 生产:100-500, 大规模:500-5000 / Test:10-50, Prod:100-500, Large:500-5000"
                    )
                    multi_batch_size = gr.Slider(
                        8, 128, value=64, step=8,
                        label="批次大小 / Batch Size",
                        info="RTX 5090推荐64-128 / RTX 5090: 64-128"
                    )
                    multi_epochs = gr.Slider(
                        1, 100, value=20, step=1,
                        label="训练轮数 / Epochs",
                        info="通常20-50轮 / Usually 20-50"
                    )
                    multi_lr = gr.Number(
                        value=0.001,
                        label="学习率 / Learning Rate",
                        info="默认0.001 / Default 0.001"
                    )

                    multi_use_cache = gr.Checkbox(
                        value=True,
                        label="使用预构建缓存 / Use Pre-built Cache (FAST)",
                        info="404x faster regime detection! Run build_cache.py first"
                    )

                    multi_use_compile = gr.Checkbox(
                        value=False,
                        label="使用 torch.compile 加速 / Use torch.compile (Experimental)",
                        info="1.24x speedup with batch size 128+. May be slow on first run."
                    )

                    multi_use_amp = gr.Checkbox(
                        value=True,
                        label="混合精度训练 / Mixed Precision (AMP)",
                        info="使用FP16加速训练，降低显存占用 / Use FP16 for faster training and lower VRAM"
                    )

                    multi_loss_type = gr.Dropdown(
                        ["MSE", "Trend Aware", "Huber Trend", "Directional"],
                        value="Trend Aware",
                        label="损失函数 / Loss Function",
                        info="MSE:标准 / Trend Aware:趋势感知(推荐) / Huber:鲁棒 / Directional:方向"
                    )

                    multi_save_interval = gr.Slider(
                        1, 20, value=5, step=1,
                        label="保存间隔 / Save Interval (epochs)",
                        info="每N个epoch保存一次模型 / Save model every N epochs"
                    )

                    multi_early_stopping_patience = gr.Slider(
                        5, 100, value=50, step=5,
                        label="Early Stopping Patience (epochs)",
                        info="验证损失未改善的等待轮数，设为100可禁用 / Wait epochs before stopping, set to 100 to disable"
                    )

                    multi_model_name = gr.Textbox(
                        value="multi_dim_model",
                        label="模型名称 / Model Name",
                        info="自动添加时间戳 / Timestamp will be added automatically"
                    )

                    multi_output_dir = gr.Textbox(
                        value="checkpoints",
                        label="输出路径 / Output Directory",
                        info="模型保存目录 / Directory to save models"
                    )

                    with gr.Row():
                        multi_train_btn = gr.Button("🚀 开始多维度训练 / Start Multi-Dim Training", variant="primary", size="lg")
                        multi_stop_btn = gr.Button("⏹️ 停止训练 / Stop Training", variant="stop", size="lg")

            with gr.Row():
                with gr.Column():
                    multi_train_output = gr.Textbox(label="Training Log", lines=15, max_lines=20)
                with gr.Column():
                    multi_plot_output = gr.Plot(label="Real-time Training Curve")

            build_cache_btn.click(
                fn=build_cache_for_training,
                inputs=[cache_max_stocks, cache_use_gpu],
                outputs=[cache_output, cache_plot]
            )

            analyze_btn.click(
                fn=analyze_universe,
                inputs=[analyze_max_stocks],
                outputs=[analysis_output, analysis_plot]
            )

            multi_train_btn.click(
                fn=train_multi_dim_model,
                inputs=[
                    training_mode, industry_filter, multi_max_stocks, multi_batch_size,
                    multi_epochs, multi_lr, multi_use_amp, multi_loss_type, multi_save_interval,
                    multi_early_stopping_patience, multi_model_name, multi_output_dir, multi_use_cache, multi_use_compile
                ],
                outputs=[multi_train_output, multi_plot_output]
            )

            multi_stop_btn.click(
                fn=stop_multi_dim_training,
                inputs=[],
                outputs=[multi_train_output]
            )

    gr.Markdown("""
    ---
    ## 💡 使用提示 / Tips

    ### 📥 数据下载 / Data Download
    - **首次使用 / First Time**: 先下载100只股票测试 / Download 100 stocks for testing
    - **生产环境 / Production**: 下载500-5000只股票 / Download 500-5000 stocks
    - **数据更新 / Update**: 定期重新下载获取最新数据 / Re-download periodically for latest data
    - **存储位置 / Storage**: `full_stock_data/training_data/` 目录 / directory

    ### 🎯 基础训练 / Basic Training
    - **Transformer**: 复杂模式，使用Flash Attention加速 / Complex patterns, uses Flash Attention
    - **LSTM**: 简单快速，适合基础序列 / Simple and fast, good for basic sequences
    - **批次大小 / Batch Size**: 越大越快(GPU内存允许) / Larger = faster (if GPU allows)
    - **学习率 / Learning Rate**: 0.001默认，0.0001微调 / 0.001 default, 0.0001 for fine-tuning
    - **合成数据 / Synthetic Data**: 仅用于快速测试模型架构 / Only for quick architecture testing

    ### 🎨 多维度训练 / Multi-Dimensional Training
    - **统一条件化模型 / Unified Conditional Model**:
      * 一个模型学习所有行业 / One model learns all industries
      * 通过条件输入区分股票类型 / Distinguishes via conditional inputs
      * **推荐使用 / RECOMMENDED** - 效率高、性能好、泛化强 / High efficiency, better performance, strong generalization

    - **单行业模型 / Single Industry Model**:
      * 针对特定行业训练 / Train for specific industry
      * 适合行业特定策略 / Good for industry-specific strategies

    - **行业分类 / Industry Classification**: 11类 / 11 categories
      * 金融、消费、科技、医药、工业、材料、能源、公用事业、房地产、电信、其他
      * Finance, Consumer, Technology, Healthcare, Industrial, Materials, Energy, Utilities, Real Estate, Telecom, Other

    - **风格因子 / Style Factors**: 5类 / 5 categories
      * 市值、估值、波动率、动量 / Market cap, Valuation, Volatility, Momentum

    - **市场环境 / Market Regime**: 4种 / 4 states
      * 牛市、熊市、震荡、高波动 / Bull, Bear, Sideways, Volatile

    - **数据要求 / Data Requirements**:
      * 最少10只股票(测试) / Min 10 stocks (testing)
      * 推荐100-500只(生产) / Recommended 100-500 (production)
      * 使用真实A股数据 / Uses real A-share data

    ### 🎯 Flash Attention 优势 / Benefits
    - ⚡ 训练速度提升2-4倍 / 2-4x faster training
    - 💾 内存占用更低 / Lower memory usage
    - 📈 长序列扩展性更好 / Better scaling for long sequences
    - 🚀 RTX 5090完美支持 / Perfect support on RTX 5090

    ### 📊 训练建议 / Training Recommendations
    1. **数据准备 / Data Preparation**: 先下载数据 → 分析股票池 / Download data → Analyze universe
    2. **快速测试 / Quick Test**: 10-50只股票，10轮训练 / 10-50 stocks, 10 epochs
    3. **生产训练 / Production**: 100-500只股票，20-50轮训练 / 100-500 stocks, 20-50 epochs
    4. **模型选择 / Model Selection**: 多维度训练优于基础训练 / Multi-dim better than basic
    5. **性能监控 / Performance**: 关注验证损失，避免过拟合 / Watch validation loss, avoid overfitting

    ### 🔧 硬件配置 / Hardware Configuration
    - **GPU**: NVIDIA RTX 5090 (32GB VRAM)
    - **CUDA**: 12.8
    - **推荐批次大小 / Recommended Batch Size**: 64-128
    - **训练速度 / Training Speed**: ~88 it/s

    ---
    **作者 / Author**: eddy | **版本 / Version**: 2.0 | **更新 / Updated**: 2025-11-14
    """)

if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_api=False,
        quiet=False,
        inbrowser=False
    )

