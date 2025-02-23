"""Main trainer class for model training and evaluation."""
from typing import Dict, List, Optional, Union, Any, Callable
import os
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torch.distributed as dist

from .optimizer import AdamW, ShardedAdam
from .callbacks import (
    CheckpointCallback,
    EarlyStopping,
    LRSchedulerCallback
)
from .distributed import (
    initialize_tpu,
    create_xla_model,
    xla_data_loader
)

class Trainer:
    """Main trainer class.
    
    Handles training loop, optimization, checkpointing, and metrics.
    Supports distributed training and expert parallelism.
    
    Args:
        model: Model to train
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        optimizer: Optimizer instance
        device: Device to train on
        max_steps: Maximum training steps
        max_epochs: Maximum epochs
        accumulation_steps: Gradient accumulation steps
        callbacks: Optional training callbacks
        use_tpu: Whether to use TPU
        use_amp: Whether to use automatic mixed precision
        expert_parallel: Whether using expert parallelism
        log_every: How often to log metrics
        save_every: How often to save checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        device: Optional[torch.device] = None,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        accumulation_steps: int = 1,
        callbacks: Optional[List[Any]] = None,
        use_tpu: bool = False,
        use_amp: bool = False,
        expert_parallel: bool = False,
        log_every: int = 100,
        save_every: Optional[int] = None
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer or self._create_optimizer()
        self.device = device or self._setup_device(use_tpu)
        self.max_steps = max_steps
        self.max_epochs = max_epochs
        self.accumulation_steps = accumulation_steps
        self.callbacks = callbacks or []
        self.use_tpu = use_tpu
        self.use_amp = use_amp
        self.expert_parallel = expert_parallel
        self.log_every = log_every
        self.save_every = save_every
        
        # Initialize state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf')
        
        # Move model to device
        if self.use_tpu:
            self.model = create_xla_model(self.model, self.device)
        else:
            self.model = self.model.to(self.device)
            
        # Setup AMP if enabled
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
    def _create_optimizer(self) -> Optimizer:
        """Create default optimizer."""
        if self.expert_parallel:
            return ShardedAdam(
                self.model.parameters(),
                lr=1e-3,
                expert_parallel=True
            )
        return AdamW(self.model.parameters(), lr=1e-3)
        
    def _setup_device(self, use_tpu: bool) -> torch.device:
        """Setup training device."""
        if use_tpu:
            return initialize_tpu()
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _run_batch(
        self,
        batch: Dict[str, torch.Tensor],
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Process single batch.
        
        Args:
            batch: Input batch
            training: Whether in training mode
            
        Returns:
            Dict with loss and metrics
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast() if self.use_amp else nullcontext():
            outputs = self.model(**batch)
            
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        if training:
            loss = loss / self.accumulation_steps
            
        # Backward pass
        if training and self.use_amp:
            self.scaler.scale(loss).backward()
        elif training:
            loss.backward()
            
        return {
            'loss': loss.item() * self.accumulation_steps,
            **{k: v.item() for k, v in outputs.items() if k != 'loss'}
        }
        
    def _run_epoch(self, training: bool = True) -> Dict[str, float]:
        """Run single epoch.
        
        Args:
            training: Whether in training mode
            
        Returns:
            Dict of metrics
        """
        dataloader = self.train_dataloader if training else self.val_dataloader
        
        if training:
            self.model.train()
        else:
            self.model.eval()
            
        metrics = []
        for batch_idx, batch in enumerate(dataloader):
            # Process batch
            if training:
                batch_metrics = self._run_batch(batch, training=True)
                
                # Optimizer step
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update step counter
                    self.global_step += 1
                    
                    # Run callbacks
                    for callback in self.callbacks:
                        callback(
                            epoch=self.current_epoch,
                            step=self.global_step,
                            metrics=batch_metrics
                        )
                        
                    # Check max steps
                    if self.max_steps and self.global_step >= self.max_steps:
                        return self._aggregate_metrics(metrics)
                        
            else:
                with torch.no_grad():
                    batch_metrics = self._run_batch(batch, training=False)
                    
            metrics.append(batch_metrics)
            
            # Log metrics
            if (batch_idx + 1) % self.log_every == 0:
                self._log_metrics(self._aggregate_metrics(metrics), training)
                
            # Save checkpoint
            if self.save_every and (batch_idx + 1) % self.save_every == 0:
                self._save_checkpoint()
                
        return self._aggregate_metrics(metrics)
        
    def _aggregate_metrics(self, metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate metrics across batches.
        
        Args:
            metrics: List of metric dicts
            
        Returns:
            Aggregated metrics
        """
        if not metrics:
            return {}
            
        agg_metrics = {}
        for k in metrics[0].keys():
            agg_metrics[k] = sum(m[k] for m in metrics) / len(metrics)
            
        return agg_metrics
        
    def _log_metrics(
        self,
        metrics: Dict[str, float],
        training: bool = True
    ) -> None:
        """Log metrics.
        
        Args:
            metrics: Metrics dict
            training: Whether in training mode
        """
        prefix = 'train' if training else 'val'
        metrics_str = ' '.join(
            f"{prefix}_{k}: {v:.4f}" for k, v in metrics.items()
        )
        print(
            f"Epoch {self.current_epoch} Step {self.global_step}: {metrics_str}"
        )
        
    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        save_dict = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_metric': self.best_metric
        }
        
        # Add callback states
        save_dict['callbacks'] = {
            f"callback_{i}": cb.state_dict()
            for i, cb in enumerate(self.callbacks)
            if hasattr(cb, 'state_dict')
        }
        
        path = f"checkpoint-{self.current_epoch:03d}-{self.global_step:06d}.pt"
        torch.save(save_dict, path)
        
    def _load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scaler and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        # Load callback states
        for i, cb in enumerate(self.callbacks):
            if hasattr(cb, 'load_state_dict'):
                state_key = f"callback_{i}"
                if state_key in checkpoint['callbacks']:
                    cb.load_state_dict(checkpoint['callbacks'][state_key])
                    
    def train(
        self,
        resume_from: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """Run training loop.
        
        Args:
            resume_from: Optional checkpoint to resume from
            
        Returns:
            Dict of training history
        """
        if resume_from:
            self._load_checkpoint(resume_from)
            
        history = {'train': [], 'val': []}
        while True:
            # Check max epochs
            if self.max_epochs and self.current_epoch >= self.max_epochs:
                break
                
            # Training epoch
            train_metrics = self._run_epoch(training=True)
            history['train'].append(train_metrics)
            
            # Validation epoch
            if self.val_dataloader is not None:
                val_metrics = self._run_epoch(training=False)
                history['val'].append(val_metrics)
                
                # Update best metric
                if val_metrics['loss'] < self.best_metric:
                    self.best_metric = val_metrics['loss']
                    
            # Increment epoch counter
            self.current_epoch += 1
            
            # Check max steps
            if self.max_steps and self.global_step >= self.max_steps:
                break
                
        return history

class nullcontext:
    """Context manager that does nothing."""
    def __enter__(self): return None
    def __exit__(self, *excinfo): return False
