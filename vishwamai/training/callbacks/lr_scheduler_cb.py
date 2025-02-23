"""Learning rate scheduler callbacks."""
from typing import Optional, Dict, Union, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class LRSchedulerCallback:
    """Base learning rate scheduler callback."""
    
    def __init__(
        self,
        scheduler: _LRScheduler,
        monitor: Optional[str] = None,
        interval: str = 'epoch',
        frequency: int = 1,
        strict: bool = True
    ):
        """Initialize scheduler callback.
        
        Args:
            scheduler: Learning rate scheduler
            monitor: Optional metric to monitor
            interval: When to update ['epoch', 'step']
            frequency: How often to update
            strict: Whether to enforce monitoring
        """
        self.scheduler = scheduler
        self.monitor = monitor
        self.interval = interval
        self.frequency = frequency
        self.strict = strict
        
        self.last_step = -1
        
    def __call__(
        self,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update learning rate.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dict
        """
        if self.interval == 'epoch':
            if epoch % self.frequency == 0:
                if self.monitor is not None:
                    if metrics is None or self.monitor not in metrics:
                        if self.strict:
                            raise ValueError(
                                f'Monitored metric {self.monitor} not found in metrics'
                            )
                        return
                    self.scheduler.step(metrics[self.monitor])
                else:
                    self.scheduler.step()
                    
        elif self.interval == 'step':
            if step % self.frequency == 0 and step != self.last_step:
                self.scheduler.step()
                self.last_step = step
                
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            'scheduler_state': self.scheduler.state_dict(),
            'last_step': self.last_step
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from checkpoint."""
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.last_step = state_dict['last_step']

class WarmupCallback(LRSchedulerCallback):
    """Callback for warmup scheduling."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        min_lr: float = 0.0,
        warmup_type: str = 'linear',
        interval: str = 'step',
        frequency: int = 1
    ):
        """Initialize warmup callback.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            min_lr: Minimum learning rate
            warmup_type: Type of warmup ['linear', 'exponential']
            interval: When to update ['epoch', 'step']
            frequency: How often to update
        """
        from ..scheduling.warmup import (
            LinearWarmup,
            ExponentialWarmup
        )
        
        warmup_cls = {
            'linear': LinearWarmup,
            'exponential': ExponentialWarmup
        }[warmup_type]
        
        scheduler = warmup_cls(
            optimizer,
            warmup_steps=warmup_steps,
            min_lr=min_lr
        )
        
        super().__init__(
            scheduler=scheduler,
            interval=interval,
            frequency=frequency
        )

class CosineAnnealingCallback(LRSchedulerCallback):
    """Callback for cosine annealing scheduling."""
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
        interval: str = 'step',
        frequency: int = 1
    ):
        """Initialize cosine annealing callback.
        
        Args:
            optimizer: Wrapped optimizer
            total_steps: Total number of steps
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            interval: When to update ['epoch', 'step']
            frequency: How often to update
        """
        from ..scheduling.lr_scheduler import CosineAnnealingWarmupLR
        
        scheduler = CosineAnnealingWarmupLR(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr
        )
        
        super().__init__(
            scheduler=scheduler,
            interval=interval,
            frequency=frequency
        )

class ExpertSchedulerCallback(LRSchedulerCallback):
    """Learning rate scheduler for expert-parallel training."""
    
    def __init__(
        self,
        scheduler: _LRScheduler,
        expert_lr_mult: float = 1.0,
        interval: str = 'epoch',
        frequency: int = 1
    ):
        """Initialize expert scheduler.
        
        Args:
            scheduler: Learning rate scheduler
            expert_lr_mult: Expert learning rate multiplier
            interval: When to update ['epoch', 'step']
            frequency: How often to update
        """
        super().__init__(
            scheduler=scheduler,
            interval=interval,
            frequency=frequency
        )
        self.expert_lr_mult = expert_lr_mult
        
    def __call__(
        self,
        epoch: int,
        step: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Update learning rates with expert handling.
        
        Args:
            epoch: Current epoch
            step: Current step
            metrics: Optional metrics dict
        """
        super().__call__(epoch, step, metrics)
        
        # Apply expert multiplier
        for param_group in self.scheduler.optimizer.param_groups:
            if 'expert' in param_group.get('name', ''):
                param_group['lr'] *= self.expert_lr_mult
