"""Learning rate scheduler callback for training."""

from typing import Dict, List, Optional, Union
import torch
from torch.optim.lr_scheduler import _LRScheduler
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduler callback.
    
    Attributes:
        scheduler_type (str): Type of scheduler (e.g., "cosine", "linear")
        monitor (str): Metric to monitor for plateau scheduler
        interval (str): Scheduler step interval ("step" or "epoch")
        frequency (int): Number of steps/epochs between scheduler updates
        reduce_on_plateau (bool): Whether to reduce LR on plateau
    """
    scheduler_type: str = "cosine"
    monitor: str = "val_loss"
    interval: str = "epoch"
    frequency: int = 1
    reduce_on_plateau: bool = False

class LRSchedulerCallback:
    """Callback for managing learning rate scheduling during training.
    
    This callback handles learning rate scheduling based on various strategies,
    including step-based, epoch-based, and metric-based scheduling.
    
    Args:
        scheduler (_LRScheduler): Learning rate scheduler
        config (LRSchedulerConfig): Scheduler configuration
    """
    
    def __init__(
        self,
        scheduler: _LRScheduler,
        config: LRSchedulerConfig
    ):
        self.scheduler = scheduler
        self.config = config
        self.last_step = -1
        self.last_epoch = -1
        self.lr_history: List[float] = []
        
        if config.interval not in ["step", "epoch"]:
            raise ValueError(
                f"interval must be 'step' or 'epoch', got {config.interval}"
            )
            
        if config.frequency < 1:
            raise ValueError(
                f"frequency must be >= 1, got {config.frequency}"
            )
            
    def on_train_batch_end(
        self,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Called after training batch.
        
        Args:
            step (int): Current global step
            metrics (Optional[Dict[str, float]], optional): Training metrics. Defaults to None.
        """
        if self.config.interval == "step":
            if step % self.config.frequency == 0 and step > self.last_step:
                if self.config.reduce_on_plateau and metrics:
                    monitor_val = metrics.get(self.config.monitor)
                    if monitor_val is not None:
                        self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()
                    
                self.last_step = step
                self._record_lr()
                
    def on_epoch_end(
        self,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Called after epoch.
        
        Args:
            epoch (int): Current epoch
            metrics (Optional[Dict[str, float]], optional): Epoch metrics. Defaults to None.
        """
        if self.config.interval == "epoch":
            if epoch % self.config.frequency == 0 and epoch > self.last_epoch:
                if self.config.reduce_on_plateau and metrics:
                    monitor_val = metrics.get(self.config.monitor)
                    if monitor_val is not None:
                        self.scheduler.step(monitor_val)
                else:
                    self.scheduler.step()
                    
                self.last_epoch = epoch
                self._record_lr()
                
    def _record_lr(self) -> None:
        """Record current learning rate."""
        current_lr = self.get_last_lr()
        self.lr_history.append(current_lr)
        
    def get_last_lr(self) -> float:
        """Get most recent learning rate.
        
        Returns:
            float: Last learning rate
        """
        try:
            return self.scheduler.get_last_lr()[0]
        except (AttributeError, IndexError):
            param_groups = self.scheduler.optimizer.param_groups
            return param_groups[0]['lr'] if param_groups else 0.0
            
    def state_dict(self) -> Dict:
        """Get current state for checkpointing.
        
        Returns:
            Dict: State dictionary
        """
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'last_step': self.last_step,
            'last_epoch': self.last_epoch,
            'lr_history': self.lr_history
        }
        
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint.
        
        Args:
            state_dict (Dict): State dictionary
        """
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.last_step = state_dict['last_step']
        self.last_epoch = state_dict['last_epoch']
        self.lr_history = state_dict['lr_history']
        
    def get_lr_history(self) -> List[float]:
        """Get history of learning rates.
        
        Returns:
            List[float]: Learning rate history
        """
        return self.lr_history
        
    def get_current_lr(self) -> float:
        """Get current learning rate.
        
        Returns:
            float: Current learning rate
        """
        return self.get_last_lr()
