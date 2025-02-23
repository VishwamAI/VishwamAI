"""Learning rate scheduler implementations."""
from typing import List, Optional, Dict, Any
import math
import warnings
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingWarmupLR(_LRScheduler):
    """Cosine annealing scheduler with warmup.
    
    Combines cosine annealing with linear warmup for better training stability.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0)
        last_epoch: Last epoch (default: -1)
        verbose: Whether to print scheduler changes (default: False)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups.
        
        Returns:
            Learning rates for each parameter group
        """
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
            
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_mult = float(step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay after warmup
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            lr_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return [
            self.min_lr + (base_lr - self.min_lr) * lr_mult
            for base_lr in self.base_lrs
        ]

class LinearWarmupLR(_LRScheduler):
    """Linear learning rate scheduler with warmup.
    
    Implements linear learning rate decay with initial warmup period.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0)
        last_epoch: Last epoch (default: -1)
        verbose: Whether to print scheduler changes (default: False)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
            
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_mult = float(step) / float(max(1, self.warmup_steps))
        else:
            # Linear decay
            lr_mult = max(0.0, float(self.total_steps - step) / 
                         float(max(1, self.total_steps - self.warmup_steps)))
            
        return [
            self.min_lr + (base_lr - self.min_lr) * lr_mult
            for base_lr in self.base_lrs
        ]

class PolynomialDecayLR(_LRScheduler):
    """Polynomial learning rate decay scheduler.
    
    Implements polynomial learning rate decay with optional warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate (default: 0)
        power: Power for polynomial decay (default: 1.0)
        last_epoch: Last epoch (default: -1)
        verbose: Whether to print scheduler changes (default: False)  
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        power: float = 1.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.power = power
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> List[float]:
        """Get learning rates for all parameter groups."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.")
            
        step = self.last_epoch
        
        if step < self.warmup_steps:
            # Linear warmup
            lr_mult = float(step) / float(max(1, self.warmup_steps))
        else:
            # Polynomial decay
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            lr_mult = math.pow(1.0 - progress, self.power)
            
        return [
            self.min_lr + (base_lr - self.min_lr) * lr_mult
            for base_lr in self.base_lrs
        ]

class ExpertWiseLR(_LRScheduler):
    """Expert-specific learning rate scheduler.
    
    Implements separate learning rate schedules for expert and shared parameters.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        expert_lr_mult: Multiplier for expert learning rates (default: 1.0)
        min_lr: Minimum learning rate (default: 0)
        schedule_type: Type of schedule ['cosine', 'linear', 'polynomial']
        last_epoch: Last epoch (default: -1)
        verbose: Whether to print scheduler changes (default: False)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        expert_lr_mult: float = 1.0,
        min_lr: float = 0.0,
        schedule_type: str = 'cosine',
        last_epoch: int = -1,
        verbose: bool = False
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.expert_lr_mult = expert_lr_mult
        self.min_lr = min_lr
        self.schedule_type = schedule_type
        
        # Create separate schedulers for expert and shared params
        if schedule_type == 'cosine':
            self.shared_scheduler = CosineAnnealingWarmupLR(
                optimizer, warmup_steps, total_steps, min_lr
            )
        elif schedule_type == 'linear':
            self.shared_scheduler = LinearWarmupLR(
                optimizer, warmup_steps, total_steps, min_lr
            )
        elif schedule_type == 'polynomial':
            self.shared_scheduler = PolynomialDecayLR(
                optimizer, warmup_steps, total_steps, min_lr
            )
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        super().__init__(optimizer, last_epoch, verbose)
        
    def get_lr(self) -> List[float]:
        """Get learning rates for expert and shared parameters."""
        shared_lrs = self.shared_scheduler.get_lr()
        
        return [
            lr * (self.expert_lr_mult if 'expert' in group.get('name', '')
                 else 1.0)
            for lr, group in zip(shared_lrs, self.optimizer.param_groups)
        ]
