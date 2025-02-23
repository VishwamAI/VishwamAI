"""Cosine annealing learning rate scheduler with warmup."""

import math
from typing import List, Optional
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import logging

logger = logging.getLogger(__name__)

class CosineAnnealingWarmupScheduler(_LRScheduler):
    """Cosine annealing scheduler with warmup.
    
    This scheduler implements a cosine annealing schedule with an optional warmup
    period. The learning rate follows these phases:
    1. Linear warmup from warmup_start_lr to max_lr over warmup_steps
    2. Cosine decay from max_lr to min_lr over remaining steps
    
    Args:
        optimizer (Optimizer): Wrapped optimizer
        warmup_steps (int): Number of warmup steps
        total_steps (int): Total number of training steps
        max_lr (float): Maximum learning rate after warmup
        min_lr (float, optional): Minimum learning rate at end. Defaults to 0.0
        warmup_start_lr (float, optional): Initial warmup learning rate. Defaults to 0.0
        last_epoch (int, optional): The index of last epoch. Defaults to -1
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        max_lr: float,
        min_lr: float = 0.0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        
        # Validate parameters
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < warmup_steps:
            raise ValueError(
                f"total_steps must be >= warmup_steps, got {total_steps} < {warmup_steps}"
            )
        if max_lr < min_lr:
            raise ValueError(
                f"max_lr must be >= min_lr, got {max_lr} < {min_lr}"
            )
        if not 0.0 <= min_lr <= max_lr:
            raise ValueError(
                f"Invalid learning rates: min_lr={min_lr}, max_lr={max_lr}"
            )
        if warmup_start_lr < 0:
            raise ValueError(
                f"warmup_start_lr must be >= 0, got {warmup_start_lr}"
            )
            
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
        
        # Warmup phase
        if step < self.warmup_steps:
            alpha = step / self.warmup_steps
            return [
                self.warmup_start_lr + alpha * (self.max_lr - self.warmup_start_lr)
                for _ in self.base_lrs
            ]
            
        # Cosine annealing phase
        step = step - self.warmup_steps
        total_cosine_steps = self.total_steps - self.warmup_steps
        
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / total_cosine_steps))
        return [
            self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            for _ in self.base_lrs
        ]
        
    def get_warmup_lr(self, step: int) -> List[float]:
        """Get learning rates during warmup phase.
        
        Args:
            step (int): Current step in warmup phase
            
        Returns:
            List[float]: Warmup learning rates
        """
        alpha = step / self.warmup_steps
        return [
            self.warmup_start_lr + alpha * (self.max_lr - self.warmup_start_lr)
            for _ in self.base_lrs
        ]
        
    def get_cosine_lr(self, step: int) -> List[float]:
        """Get learning rates during cosine annealing phase.
        
        Args:
            step (int): Current step in cosine phase
            
        Returns:
            List[float]: Cosine annealed learning rates
        """
        total_cosine_steps = self.total_steps - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step / total_cosine_steps))
        return [
            self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            for _ in self.base_lrs
        ]
        
    def state_dict(self) -> dict:
        """Returns scheduler state for checkpointing.
        
        Returns:
            dict: State dictionary
        """
        state_dict = {
            key: value for key, value in self.__dict__.items()
            if key not in ['optimizer']
        }
        return state_dict
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads scheduler state from checkpoint.
        
        Args:
            state_dict (dict): State dictionary
        """
        self.__dict__.update(state_dict)
        
    def get_final_lr(self) -> List[float]:
        """Get final learning rates after scheduling.
        
        Returns:
            List[float]: Final learning rates
        """
        return [self.min_lr for _ in self.base_lrs]
