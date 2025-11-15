"""
Financial Model Configuration

Author: eddy
"""

import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""

    model_type: str = "transformer"

    input_dim: int = 5
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    seq_length: int = 60
    pred_length: int = 1

    use_flash_attention: bool = True

    industry_embed_dim: int = 128
    style_embed_dim: int = 64
    regime_embed_dim: int = 64

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingConfig:
    """Training configuration"""

    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0001

    early_stopping_patience: int = 10

    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    checkpoint_dir: str = "financial_model/checkpoints"
    log_dir: str = "financial_model/logs"

    save_every_n_epochs: int = 5

    use_amp: bool = True

    gradient_clip: float = 1.0

    loss_type: str = "mse"

@dataclass
class DataConfig:
    """Data configuration"""
    
    data_path: Optional[str] = None
    
    features: list = None
    target: str = "close"
    
    normalize: bool = True
    
    add_technical_indicators: bool = True
    
    def __post_init__(self):
        if self.features is None:
            self.features = ["open", "high", "low", "close", "volume"]

class Config:
    """Main configuration"""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
    
    def __repr__(self):
        return f"""
Financial Model Configuration:
=============================
Model:
  Type: {self.model.model_type}
  Input Dim: {self.model.input_dim}
  Hidden Dim: {self.model.hidden_dim}
  Layers: {self.model.num_layers}
  Heads: {self.model.num_heads}
  Flash Attention: {self.model.use_flash_attention}
  Device: {self.model.device}

Training:
  Batch Size: {self.training.batch_size}
  Epochs: {self.training.num_epochs}
  Learning Rate: {self.training.learning_rate}
  Early Stopping: {self.training.early_stopping_patience} epochs
  Mixed Precision: {self.training.use_amp}

Data:
  Features: {self.data.features}
  Target: {self.data.target}
  Normalize: {self.data.normalize}
  Technical Indicators: {self.data.add_technical_indicators}
"""

if __name__ == "__main__":
    config = Config()
    print(config)

