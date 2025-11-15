<img width="1090" height="326" alt="image" src="https://github.com/user-attachments/assets/d10e5e5b-fcc6-4153-b3ab-6355e1787005" />
<img width="1376" height="571" alt="image" src="https://github.com/user-attachments/assets/bca461fb-877e-42ff-b3df-854b9a8fde06" />
<img width="1409" height="404" alt="image" src="https://github.com/user-attachments/assets/47ce2c7f-10f6-45b7-b188-82f79df21a65" />
<img width="1393" height="601" alt="image" src="https://github.com/user-attachments/assets/e1bd2eaf-323b-4d83-9312-4677ba5bcdfa" />
<img width="1413" height="604" alt="image" src="https://github.com/user-attachments/assets/8225ece5-b956-48e9-b56a-a67fdeba307e" />






# A-Share Multi-Dimensional Financial Model

A complete financial time series prediction system with multi-dimensional conditional training for A-share stocks.

## Author
eddy

## Repository
https://github.com/eddyhhlure1Eddy/Hermes

## Version
2.0 (Updated: 2025-11-15)

## Core Features

- **Bilingual Interface**: Chinese and English support
- **Integrated Data Download**: Download A-share data directly in Gradio interface
- **Multi-Dimensional Training**: Industry, style factors, and market regime awareness
- **Flash Attention 2.8.2**: GPU-accelerated training on RTX 5090
- **Real A-Share Data**: Historical data from 1999 to present (3500+ stocks)
- **Web Interface**: Gradio-based training and inference panel
- **Conditional Model**: Shared backbone with context-aware predictions
- **Improved Loss Functions**: Trend-aware, directional, and Huber loss options
- **GPU-Accelerated Data Processing**: 2-5x faster data normalization
- **Real-time Training Progress**: Live loss curves and progress updates

## Quick Start

### Method 1: Use Start Script

**Windows:**
```bash
start.bat
```

### Method 2: Manual Start

```bash
python gradio_app.py
```

Open browser: http://127.0.0.1:7860

## Complete Workflow

### Step 1: Download Data

1. Open Gradio interface
2. Go to "Data Download" tab
3. Set number of stocks (recommend 100 for testing)
4. Set start date (default: 1999-01-01)
5. Click "Start Download"
6. Wait for completion
7. Click "Refresh Status" to verify

**Data will be saved to:** `full_stock_data/training_data/`

### Step 2: Analyze Universe (Optional)

1. Go to "Multi-Dimensional Training" tab
2. Set "Max Stocks to Analyze"
3. Click "Analyze Universe"
4. Review industry distribution and style factors

### Step 3: Train Model

**Option A: Basic Training (Quick Test)**
1. Go to "Basic Training" tab
2. Select model type (Transformer recommended)
3. Configure parameters
4. Click "Start Training"

**Option B: Multi-Dimensional Training (Recommended)**
1. Go to "Multi-Dimensional Training" tab
2. Select "Unified Conditional Model"
3. Set max stocks (100-500)
4. Configure parameters:
   - Batch Size: 64 (for RTX 5090)
   - Epochs: 30-50
   - Learning Rate: 0.001
   - Loss Function: Trend Aware (recommended)
   - Save Interval: 5 epochs
   - Model Name: custom name with auto timestamp
   - Output Directory: relative path (e.g., "output" or "checkpoints")
5. Click "Start Multi-Dim Training"

### Step 4: Model Inference

1. Go to "Inference" tab
2. Select checkpoint path
3. Set test samples
4. Click "Run Inference"
5. Review predictions

## Project Structure

```
fork/
├── gradio_app.py                    # Main Gradio web interface
├── batch_download.py                # A-share data downloader
├── financial_model/                 # Core model implementation
│   ├── src/
│   │   ├── config.py               # Configuration
│   │   ├── model.py                # Base models (Transformer/LSTM)
│   │   ├── conditional_model.py    # Multi-dimensional conditional model
│   │   ├── dataset.py              # Basic dataset
│   │   ├── multi_dim_dataset.py    # Multi-dimensional dataset
│   │   ├── stock_metadata.py       # Industry & style classification
│   │   ├── market_regime.py        # Market regime detection
│   │   ├── train.py                # Trainer
│   │   └── inference.py            # Predictor
│   └── checkpoints/                # Saved models
├── full_stock_data/                # Downloaded stock data
│   ├── training_data/              # CSV files (3500+ stocks)
│   └── metadata.json               # Download progress
└── old/                            # Archived files

```

## Multi-Dimensional Training

### Three Core Dimensions

1. **Industry Classification** (11 categories)
   - finance, consumer, technology, healthcare, industrial
   - materials, energy, utilities, real_estate, telecom, other

2. **Style Factors** (5 categories)
   - Market Cap: mega/large/mid/small/micro
   - Value/Growth: deep_value/value/balanced/growth
   - Volatility: low/medium/high
   - Momentum: strong/positive/neutral/negative/reversal

3. **Market Regime** (4 states)
   - bull, bear, sideways, volatile

### Loss Functions

1. **MSE Loss** (Standard)
   - Mean Squared Error
   - May produce flat predictions

2. **Trend Aware Loss** (Recommended)
   - Combines MSE, directional loss, and magnitude loss
   - Encourages trend prediction
   - Best for avoiding flat line predictions

3. **Huber Trend Loss**
   - Robust to outliers
   - Combines Huber loss with directional awareness

4. **Directional Loss**
   - Penalizes incorrect trend direction
   - Focus on getting direction right

### Model Architecture

```
Input (OHLCV) → CNN Backbone → LSTM Backbone → Backbone Features
                                                        ↓
Industry Embedding (32 dim) ────────────────────────→ Fusion Layer → Prediction Head
Style Embedding (16 dim) ───────────────────────────→
Regime Embedding (16 dim) ──────────────────────────→
```

### Training Modes

1. **Unified Conditional Model** (RECOMMENDED)
   - One model learns all industries with conditional inputs
   - Best performance and efficiency
   - Model size: ~50MB

2. **Single Industry Model**
   - Train on specific industry for specialized predictions
   - Useful for industry-specific strategies

## Data Format

CSV files with columns:
- `date`: Trading date
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

## Performance

- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **Training Speed**: ~88 it/s
- **Model Parameters**: 2.5M
- **Training Time**: ~3-4 hours (100 stocks, 20 epochs)
- **Data Size**: 3500+ stocks, ~2.4GB

## Key Files

### Main Scripts
- `gradio_app.py` - Web interface with 4 tabs (Training, Inference, Custom Data, Multi-Dim Training)
- `batch_download.py` - Download A-share data from TDX server

### Model Files
- `financial_model/src/conditional_model.py` - Multi-dimensional conditional model
- `financial_model/src/multi_dim_dataset.py` - Dataset with automatic labeling
- `financial_model/src/stock_metadata.py` - Industry and style classification
- `financial_model/src/market_regime.py` - Market regime detection
- `financial_model/src/improved_loss.py` - Advanced loss functions for better predictions

## Requirements

- Python 3.12.8
- PyTorch 2.9.1+cu128
- Flash Attention 2.8.2
- Gradio 5.7.1
- pytdx 1.72
- pandas, numpy, plotly

## Usage Examples

### Analyze Stock Universe

```python
# In Gradio interface:
# 1. Go to "Multi-Dimensional Training" tab
# 2. Set "Max Stocks to Analyze" = 100
# 3. Click "Analyze Universe"
#
# Output:
# - Industry distribution
# - Volatility styles
# - Current market regime
```

### Train Unified Model with Recommended Settings

```python
# In Gradio interface:
# 1. Select "Unified Conditional Model"
# 2. Set Max Stocks = 100
# 3. Set Batch Size = 64
# 4. Set Epochs = 30
# 5. Set Learning Rate = 0.001
# 6. Set Loss Function = "Trend Aware"
# 7. Set Save Interval = 5
# 8. Set Model Name = "stock_model_100"
# 9. Set Output Directory = "output"
# 10. Click "Start Multi-Dim Training"
#
# Models saved to:
# - output/stock_model_100_YYYYMMDD_HHMMSS_best.pt (best validation loss)
# - output/stock_model_100_YYYYMMDD_HHMMSS_epoch5.pt (every 5 epochs)
# - output/stock_model_100_YYYYMMDD_HHMMSS_final.pt (training completion)
```

## Tips

- Start with 100 stocks for testing
- Use batch size 64 for RTX 5090
- Unified Conditional Model is recommended for best results
- Use "Trend Aware" loss function to avoid flat predictions
- Monitor training loss - should decrease to < 0.01
- Check validation loss to avoid overfitting
- Use relative paths for output directory (e.g., "output" not "C:\Users\...")
- Model names automatically include timestamps
- Best model is saved when validation loss improves

## Troubleshooting

### Flat Line Predictions
- **Problem**: Model predicts constant values
- **Solution**: Use "Trend Aware" loss function instead of MSE
- **Retrain**: Train new model with improved loss function

### Path Errors
- **Problem**: "Parent directory does not exist"
- **Solution**: Use relative paths like "output" or "checkpoints"
- **Avoid**: Absolute paths like "C:\Users\..."

### Port Already in Use
- **Problem**: "Cannot find empty port in range: 7860-7860"
- **Solution**: Close previous Gradio instance or use different port

## Next Steps

1. Wait for batch download to complete (3500+ stocks)
2. Analyze full universe
3. Train on 500+ stocks for production model
4. Evaluate performance by industry and market regime
5. Fine-tune parameters based on results

## License

AGPL-3.0 license


