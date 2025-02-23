"""Linear warmup scheduler for learning rate."""

from typing import List, Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging

logger = logging.getLogger(__name__)

class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup learning rate scheduler.
    
    This scheduler implements a linear warmup schedule where the learning rate
    increases linearly from an initial value to a target value over a specified
    number of steps. After warmup, the learning rate remains constant unless
    modified by another scheduler.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer
        warmup_steps (int): Number of warmup steps
        max_lr (float): Target learning rate after warmup
        start_lr (float, optional): Initial learning rate. Defaults to 0.0
        last_epoch (int, optional): The index of last epoch. Defaults to -1
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        max_lr: float,
        start_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.start_lr = start_lr
        
        # Validate parameters
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be > 0, got {warmup_steps}")
        if not start_lr <= max_lr:
            raise ValueError(
                f"start_lr must be <= max_lr, got {start_lr} > {max_lr}"
            )
        if start_lr < 0:
            raise ValueError(f"start_lr must be >= 0, got {start_lr}")
            
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Compute learning rates for current step.
        
        Returns:
            List[float]: List of learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            logger.warning(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`."
            )
            
        step = self.last_epoch
        
        if step >= self.warmup_steps:
            return [self.max_lr for _ in self.base_lrs]
            
        # Linear warmup
        alpha = step / self.warmup_steps
        return [
            self.start_lr + alpha * (self.max_lr - self.start_lr)
            for _ in self.base_lrs
        ]
        
    def get_warmup_progress(self) -> float:
        """Get current warmup progress.
        
        Returns:
            float: Progress between 0.0 and 1.0
        """
        return min(1.0, self.last_epoch / self.warmup_steps)
        
    def is_warmup_complete(self) -> bool:
        """Check if warmup phase is complete.
        
        Returns:
            bool: True if warmup is complete
        """
        return self.last_epoch >= self.warmup_steps
        
    def get_warmup_lr(self, step: int) -> List[float]:
        """Get learning rates at a specific warmup step.
        
        Args:
            step (int): Step to compute warmup learning rates for
            
        Returns:
            List[float]: Learning rates at specified step
        """
        if step >= self.warmup_steps:
            return [self.max_lr for _ in self.base_lrs]
            
        alpha = step / self.warmup_steps
        return [
            self.start_lr + alpha * (self.max_lr - self.start_lr)
            for _ in self.base_lrs
        ]
        
    def state_dict(self) -> dict:
        """Returns scheduler state for checkpointing.
        
        Returns:
            dict: State dictionary
        """
        return {
            key: value for key, value in self.__dict__.items()
            if key not in ['optimizer']
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads scheduler state from checkpoint.
        
        Args:
            state_dict (dict): State dictionary
        """
        self.__dict__.update(state_dict)
