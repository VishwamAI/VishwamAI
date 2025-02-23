"""Checkpoint callbacks for model saving and loading."""
from typing import Any, Dict, Optional, List, Union
import os
import torch
from torch.nn import Module
from torch.optim import Optimizer

class CheckpointCallback:
    """Base class for checkpoint callbacks."""
    
    def __init__(
        self,
        dirpath: str,
        filename: str = 'checkpoint-{epoch:03d}-{step:06d}',
        monitor: str = 'val_loss',
        save_top_k: int = 1,
        mode: str = 'min',
        save_last: bool = True,
        save_weights_only: bool = False,
        every_n_epochs: Optional[int] = None,
        every_n_steps: Optional[int] = None
    ):
        """Initialize checkpoint callback.
        
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename template
            monitor: Metric to monitor for best checkpoints
            save_top_k: Number of best checkpoints to keep
            mode: One of ['min', 'max'] for metric monitoring
            save_last: Whether to save last checkpoint
            save_weights_only: Whether to only save weights
            every_n_epochs: Save every N epochs
            every_n_steps: Save every N steps
        """
        self.dirpath = dirpath
        self.filename = filename
        self.monitor = monitor
        self.save_top_k = save_top_k
        self.mode = mode
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.every_n_epochs = every_n_epochs
        self.every_n_steps = every_n_steps
        
        # Initialize tracking
        self.best_k_models: Dict[str, float] = {}
        self.best_k_paths: List[str] = []
        self.current_score = float('-inf') if mode == 'max' else float('inf')
        
        os.makedirs(dirpath, exist_ok=True)
        
    def check_monitor_top_k(self, current: float) -> bool:
        """Check if current score is among top k.
        
        Args:
            current: Current metric value
            
        Returns:
            Whether current value is in top k
        """
        if not self.best_k_models or len(self.best_k_models) < self.save_top_k:
            return True
            
        if self.mode == 'min':
            return current <= max(self.best_k_models.values())
        return current >= min(self.best_k_models.values())
        
    def format_checkpoint_name(
        self,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """Format checkpoint filename.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dict
            
        Returns:
            Formatted checkpoint filename
        """
        # Add metrics to format dict
        metrics = metrics or {}
        format_dict = dict(epoch=epoch, step=step, **metrics)
        
        filename = self.filename.format(**format_dict)
        path = os.path.join(self.dirpath, filename)
        return path
        
    def save_checkpoint(
        self,
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dict
        """
        path = self.format_checkpoint_name(epoch, step, metrics)
        
        # Create save dict
        save_dict = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict() if not self.save_weights_only else 
                               {k: v for k, v in model.state_dict().items() if 'weight' in k},
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {}
        }
        
        # Save checkpoint
        torch.save(save_dict, path)
        
        # Update best k tracking
        if metrics and self.monitor in metrics:
            current = metrics[self.monitor]
            if self.check_monitor_top_k(current):
                if len(self.best_k_paths) == self.save_top_k:
                    self.cleanup_checkpoint(self.best_k_paths[0])
                    self.best_k_paths = self.best_k_paths[1:]
                self.best_k_paths.append(path)
                self.best_k_models[path] = current
                self.current_score = current
                
        # Save last if enabled
        if self.save_last:
            last_path = os.path.join(self.dirpath, 'last.ckpt')
            torch.save(save_dict, last_path)
            
    def cleanup_checkpoint(self, path: str) -> None:
        """Remove checkpoint file.
        
        Args:
            path: Path to checkpoint
        """
        if os.path.exists(path):
            os.remove(path)
            if path in self.best_k_models:
                del self.best_k_models[path]

class ModelCheckpoint(CheckpointCallback):
    """Model checkpoint callback for standard training."""
    
    def __call__(
        self,
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Save checkpoint if conditions are met.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dict
        """
        # Check save conditions
        should_save = (
            (self.every_n_epochs and epoch % self.every_n_epochs == 0) or
            (self.every_n_steps and step % self.every_n_steps == 0) or
            (metrics and self.monitor in metrics and
             self.check_monitor_top_k(metrics[self.monitor]))
        )
        
        if should_save:
            self.save_checkpoint(model, optimizer, epoch, step, metrics)

class ExpertCheckpoint(CheckpointCallback):
    """Checkpoint callback for expert-parallel training.
    
    Handles sharded expert state correctly.
    """
    
    def save_checkpoint(
        self,
        model: Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Save checkpoint with expert handling.
        
        Args:
            model: Expert parallel model
            optimizer: Sharded optimizer
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dict
        """
        path = self.format_checkpoint_name(epoch, step, metrics)
        
        # Get consolidated optimizer state
        if hasattr(optimizer, 'consolidate_state_dict'):
            optim_state = optimizer.consolidate_state_dict()
        else:
            optim_state = optimizer.state_dict()
            
        # Create save dict
        save_dict = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.module.state_dict() 
                               if hasattr(model, 'module') 
                               else model.state_dict(),
            'optimizer_state_dict': optim_state,
            'metrics': metrics or {}
        }
        
        # Save only on rank 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            torch.save(save_dict, path)
            
            if metrics and self.monitor in metrics:
                current = metrics[self.monitor]
                if self.check_monitor_top_k(current):
                    if len(self.best_k_paths) == self.save_top_k:
                        self.cleanup_checkpoint(self.best_k_paths[0])
                        self.best_k_paths = self.best_k_paths[1:]
                    self.best_k_paths.append(path)
                    self.best_k_models[path] = current
                    self.current_score = current
                    
            if self.save_last:
                last_path = os.path.join(self.dirpath, 'last.ckpt')
                torch.save(save_dict, last_path)

class BestModelCheckpoint(ModelCheckpoint):
    """Checkpoint callback that only saves best models."""
    
    def __init__(
        self,
        dirpath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        filename: Optional[str] = None
    ):
        if filename is None:
            filename = f'best-{{epoch:03d}}-{{{monitor}:.4f}}'
        """Initialize best model checkpoint.
        
        Args:
            dirpath: Directory to save checkpoints
            monitor: Metric to monitor
            mode: One of ['min', 'max']
            filename: Checkpoint filename template
        """
        super().__init__(
            dirpath=dirpath,
            filename=filename,
            monitor=monitor,
            save_top_k=1,
            mode=mode,
            save_last=False
        )
