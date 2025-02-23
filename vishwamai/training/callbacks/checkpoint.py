"""Model checkpoint callback for saving model states."""

import os
from typing import Dict, Optional
import torch
import torch_xla.core.xla_model as xm
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing.
    
    Attributes:
        dirpath (str): Directory path for saving checkpoints
        filename (str): Base filename for checkpoints
        save_top_k (int): Number of best models to save
        monitor (str): Metric to monitor
        mode (str): One of {min, max} for monitoring
        save_last (bool): Whether to save last model
        save_weights_only (bool): Whether to save only weights
    """
    dirpath: str
    filename: str = "checkpoint"
    save_top_k: int = 3
    monitor: str = "val_loss"
    mode: str = "min"
    save_last: bool = True
    save_weights_only: bool = False

class ModelCheckpoint:
    """Callback for saving model checkpoints during training.
    
    This callback saves model checkpoints based on monitoring metrics,
    with options for saving best models and latest checkpoints.
    
    Args:
        config (CheckpointConfig): Checkpoint configuration
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.best_score = float('inf') if config.mode == "min" else float('-inf')
        self.best_path = None
        self.last_path = None
        self.saved_paths = []
        
        # Create checkpoint directory
        os.makedirs(config.dirpath, exist_ok=True)
        
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score.
        
        Args:
            current (float): Current score
            best (float): Best score so far
            
        Returns:
            bool: True if current is better
        """
        return (
            (self.config.mode == "min" and current < best) or
            (self.config.mode == "max" and current > best)
        )
        
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        filename: str
    ) -> str:
        """Save model checkpoint.
        
        Args:
            model (Module): Model to save
            optimizer (Optimizer): Optimizer to save
            scheduler (Optional[_LRScheduler]): Scheduler to save
            epoch (int): Current epoch
            step (int): Current step
            metrics (Dict[str, float]): Training metrics
            filename (str): Checkpoint filename
            
        Returns:
            str: Path to saved checkpoint
        """
        path = os.path.join(self.config.dirpath, filename)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }
        
        if self.config.save_weights_only:
            checkpoint['model_state_dict'] = model.state_dict()
        else:
            checkpoint['model'] = model
            
        # Save checkpoint
        xm.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
        return path
        
    def on_validation_end(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        metrics: Dict[str, float],
        epoch: int,
        step: int
    ) -> None:
        """Called after validation phase.
        
        Args:
            model (Module): Current model
            optimizer (Optimizer): Current optimizer
            scheduler (Optional[_LRScheduler]): Current scheduler
            metrics (Dict[str, float]): Validation metrics
            epoch (int): Current epoch
            step (int): Current step
        """
        monitor_val = metrics.get(self.config.monitor)
        if monitor_val is None:
            logger.warning(f"Metric {self.config.monitor} not found in metrics")
            return
            
        # Check if current model is best
        if self._is_better(monitor_val, self.best_score):
            self.best_score = monitor_val
            
            # Save best checkpoint
            filename = f"{self.config.filename}-best.pt"
            new_path = self._save_checkpoint(
                model, optimizer, scheduler, epoch, step, metrics, filename
            )
            
            # Update best path
            if self.best_path and self.best_path != new_path:
                if os.path.exists(self.best_path):
                    os.remove(self.best_path)
            self.best_path = new_path
            
            # Maintain top-k checkpoints
            self.saved_paths.append(new_path)
            if len(self.saved_paths) > self.config.save_top_k:
                oldest_path = self.saved_paths.pop(0)
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
                    
        # Save last checkpoint if configured
        if self.config.save_last:
            filename = f"{self.config.filename}-last.pt"
            new_path = self._save_checkpoint(
                model, optimizer, scheduler, epoch, step, metrics, filename
            )
            
            # Update last path
            if self.last_path and self.last_path != new_path:
                if os.path.exists(self.last_path):
                    os.remove(self.last_path)
            self.last_path = new_path
            
    def on_train_end(self) -> None:
        """Called at end of training."""
        logger.info(
            f"Best {self.config.monitor}: {self.best_score:.4f} "
            f"(saved to {self.best_path})"
        )
