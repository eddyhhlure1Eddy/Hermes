"""Optimized trainer with torch.compile and advanced GPU techniques.

Author: eddy
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional
import time


class CompiledTrainer:
    """High-performance trainer with torch.compile and GPU optimizations"""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        use_compile: bool = True,
        compile_backend: str = "inductor",
        compile_mode: str = "max-autotune",
    ):
        self.config = config
        self.device = torch.device(config.model.device if hasattr(config.model, 'device') else 'cuda')
        
        self.model = model.to(self.device)
        
        if use_compile and hasattr(torch, 'compile'):
            print(f"Compiling model with backend={compile_backend}, mode={compile_mode}...")
            self.model = torch.compile(
                self.model,
                backend=compile_backend,
                mode=compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            print("Model compiled successfully!")
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay if hasattr(config.training, 'weight_decay') else 0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=True,
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.training.learning_rate,
            total_steps=1000,
            pct_start=0.1,
            anneal_strategy='cos',
        )
        
        from financial_model.src.loss import create_loss_function
        self.criterion = create_loss_function(
            config.training.loss_type if hasattr(config.training, 'loss_type') else 'mse'
        )
        
        self.scaler = GradScaler(enabled=config.training.use_amp if hasattr(config.training, 'use_amp') else True)
        
        self.train_losses = []
        self.val_losses = []
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    def train_epoch(self, train_loader, use_conditional: bool = True) -> float:
        """Optimized training epoch with minimal CPU-GPU synchronization"""
        self.model.train()
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        for batch_data in train_loader:
            if use_conditional:
                batch_x, batch_y, industry_idx, style_idx, regime_idx = batch_data
                industry_idx = industry_idx.to(self.device, non_blocking=True)
                style_idx = style_idx.to(self.device, non_blocking=True)
                regime_idx = regime_idx.to(self.device, non_blocking=True)
            else:
                batch_x, batch_y = batch_data
            
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda', enabled=self.scaler.is_enabled()):
                if use_conditional:
                    outputs = self.model(batch_x, industry_idx, style_idx, regime_idx)
                else:
                    outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.detach()
            num_batches += 1
        
        avg_loss = (total_loss / num_batches).item()
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader, use_conditional: bool = True) -> float:
        """Optimized validation with no gradient computation"""
        self.model.eval()
        
        total_loss = torch.tensor(0.0, device=self.device)
        num_batches = 0
        
        for batch_data in val_loader:
            if use_conditional:
                batch_x, batch_y, industry_idx, style_idx, regime_idx = batch_data
                industry_idx = industry_idx.to(self.device, non_blocking=True)
                style_idx = style_idx.to(self.device, non_blocking=True)
                regime_idx = regime_idx.to(self.device, non_blocking=True)
            else:
                batch_x, batch_y = batch_data
            
            batch_x = batch_x.to(self.device, non_blocking=True)
            batch_y = batch_y.to(self.device, non_blocking=True)
            
            with autocast(device_type='cuda', enabled=self.scaler.is_enabled()):
                if use_conditional:
                    outputs = self.model(batch_x, industry_idx, style_idx, regime_idx)
                else:
                    outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
            
            total_loss += loss
            num_batches += 1
        
        avg_loss = (total_loss / num_batches).item()
        return avg_loss

