"""Learning rate warmup implementations."""
from typing import Optional
import math
import torch
from torch.optim import Optimizer

class WarmupScheduler:
    """Base class for warmup schedulers."""
    
    def __init__(self, optimizer: Optimizer, warmup_steps: int):
        """Initialize warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        """Update learning rates based on current step."""
        if self.current_step < self.warmup_steps:
            lr_mult = self._get_lr_multiplier()
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * lr_mult
        self.current_step += 1
        
    def _get_lr_multiplier(self) -> float:
        """Get learning rate multiplier for current step."""
        raise NotImplementedError
        
    def state_dict(self) -> dict:
        """Return state dict for checkpointing."""
        return {
            'current_step': self.current_step,
            'base_lrs': self.base_lrs,
            'warmup_steps': self.warmup_steps
        }
        
    def load_state_dict(self, state_dict: dict):
        """Load state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.base_lrs = state_dict['base_lrs']
        self.warmup_steps = state_dict['warmup_steps']

class LinearWarmup(WarmupScheduler):
    """Linear learning rate warmup.
    
    Gradually increases learning rate linearly from 0 to base_lr.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate during warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        min_lr: float = 0.0
    ):
        super().__init__(optimizer, warmup_steps)
        self.min_lr = min_lr
        
    def _get_lr_multiplier(self) -> float:
        """Get linear warmup multiplier."""
        if self.current_step >= self.warmup_steps:
            return 1.0
            
        progress = float(self.current_step) / float(max(1, self.warmup_steps))
        return self.min_lr + (1.0 - self.min_lr) * progress

class ExponentialWarmup(WarmupScheduler):
    """Exponential learning rate warmup.
    
    Gradually increases learning rate exponentially from min_lr to base_lr.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate during warmup
        gamma: Exponential growth rate
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        min_lr: float = 0.0,
        gamma: float = 5.0
    ):
        super().__init__(optimizer, warmup_steps)
        self.min_lr = min_lr
        self.gamma = gamma
        
    def _get_lr_multiplier(self) -> float:
        """Get exponential warmup multiplier."""
        if self.current_step >= self.warmup_steps:
            return 1.0
            
        progress = float(self.current_step) / float(max(1, self.warmup_steps))
        return self.min_lr + (1.0 - self.min_lr) * math.exp(self.gamma * progress - self.gamma)

class CosineWarmup(WarmupScheduler):
    """Cosine learning rate warmup.
    
    Gradually increases learning rate following a cosine curve from min_lr to base_lr.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        min_lr: Minimum learning rate during warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        min_lr: float = 0.0
    ):
        super().__init__(optimizer, warmup_steps)
        self.min_lr = min_lr
        
    def _get_lr_multiplier(self) -> float:
        """Get cosine warmup multiplier."""
        if self.current_step >= self.warmup_steps:
            return 1.0
            
        progress = float(self.current_step) / float(max(1, self.warmup_steps))
        return self.min_lr + (1.0 - self.min_lr) * (1 - math.cos(math.pi * progress)) / 2

class CustomWarmup(WarmupScheduler):
    """Custom learning rate warmup.
    
    Implements warmup with a custom multiplier function.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        warmup_fn: Custom warmup multiplier function
        min_lr: Minimum learning rate during warmup
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_fn,
        min_lr: float = 0.0
    ):
        super().__init__(optimizer, warmup_steps)
        self.warmup_fn = warmup_fn
        self.min_lr = min_lr
        
    def _get_lr_multiplier(self) -> float:
        """Get custom warmup multiplier."""
        if self.current_step >= self.warmup_steps:
            return 1.0
            
        progress = float(self.current_step) / float(max(1, self.warmup_steps))
        return self.min_lr + (1.0 - self.min_lr) * self.warmup_fn(progress)
