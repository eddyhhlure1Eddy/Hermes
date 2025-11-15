"""
Training Script for Financial Model

Author: eddy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
import os
from tqdm import tqdm
import numpy as np

from .config import Config
from .model import create_model
from .dataset import create_sample_data, create_dataloaders
from .conditional_model import create_conditional_model
from .multi_dim_dataset import create_multi_dim_dataloaders
from .improved_loss import TrendAwareLoss, HuberTrendLoss, DirectionalLoss

class Trainer:
    """Model trainer"""

    def __init__(self, config: Config, use_conditional: bool = False):
        self.config = config
        self.device = config.model.device
        self.use_conditional = use_conditional
        self.should_stop = False

        if use_conditional:
            self.model = create_conditional_model(config)
        else:
            self.model = create_model(config)

        loss_type = config.training.loss_type.lower()
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "trend_aware":
            self.criterion = TrendAwareLoss()
        elif loss_type == "huber_trend":
            self.criterion = HuberTrendLoss()
        elif loss_type == "directional":
            self.criterion = DirectionalLoss()
        else:
            print(f"Warning: Unknown loss type '{loss_type}', using MSE")
            self.criterion = nn.MSELoss()

        print(f"Using loss function: {loss_type} -> {type(self.criterion).__name__}")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )

        self.scaler = GradScaler('cuda') if config.training.use_amp and torch.cuda.is_available() else None

        self.best_val_loss = float('inf')
        self.patience_counter = 0

        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        os.makedirs(config.training.log_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader) -> float:
        """
        Train one epoch with GPU optimizations

        Optimizations:
        - non_blocking=True: Asynchronous GPU transfer
        - AMP (Automatic Mixed Precision): Faster training with FP16
        - Flash Attention: GPU-accelerated attention mechanism
        """
        self.model.train()
        total_loss = 0

        for batch_data in tqdm(train_loader, desc="Training"):
            if self.should_stop:
                raise KeyboardInterrupt("Training stopped by user")
            if self.use_conditional:
                batch_x, batch_y, industry_idx, style_idx, regime_idx = batch_data
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                industry_idx = industry_idx.to(self.device, non_blocking=True)
                style_idx = style_idx.to(self.device, non_blocking=True)
                regime_idx = regime_idx.to(self.device, non_blocking=True)
            else:
                batch_x, batch_y = batch_data
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.config.training.use_amp:
                with autocast(device_type='cuda'):
                    if self.use_conditional:
                        outputs = self.model(batch_x, industry_idx, style_idx, regime_idx)
                    else:
                        outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)

                self.scaler.scale(loss).backward()

                if self.config.training.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.use_conditional:
                    outputs = self.model(batch_x, industry_idx, style_idx, regime_idx)
                else:
                    outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()

                if self.config.training.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip
                    )

                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)
    
    def validate(self, val_loader) -> float:
        """
        Validate model with GPU optimizations

        Optimizations:
        - non_blocking=True: Asynchronous GPU transfer
        - torch.no_grad(): Disable gradient computation for faster inference
        """
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_data in val_loader:
                if self.should_stop:
                    raise KeyboardInterrupt("Training stopped by user")
                if self.use_conditional:
                    batch_x, batch_y, industry_idx, style_idx, regime_idx = batch_data
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    industry_idx = industry_idx.to(self.device, non_blocking=True)
                    style_idx = style_idx.to(self.device, non_blocking=True)
                    regime_idx = regime_idx.to(self.device, non_blocking=True)

                    outputs = self.model(batch_x, industry_idx, style_idx, regime_idx)
                else:
                    batch_x, batch_y = batch_data
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    outputs = self.model(batch_x)

                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader):
        """Full training loop"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.training.num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best_model.pt")
                print(f"  New best model saved!")
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
            
            if self.patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        if os.path.isabs(filename) or os.path.dirname(filename):
            checkpoint_path = filename
        else:
            checkpoint_path = os.path.join(self.config.training.checkpoint_dir, filename)

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        state_dict = self.model.state_dict()

        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {key.replace('_orig_mod.', ''): value for key, value in state_dict.items()}

        torch.save({
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, checkpoint_path)

if __name__ == "__main__":
    config = Config()
    print(config)
    
    print("\nCreating sample data...")
    data = create_sample_data(num_samples=2000)
    print(f"Data shape: {data.shape}")
    print(data.head())
    
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, scaler = create_dataloaders(data, config)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    print("\nStarting training...")
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)

