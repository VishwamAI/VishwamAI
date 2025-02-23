"""FairScale sharded optimizer implementation for memory-efficient training."""

from typing import Dict, List, Optional, Any
import torch
from torch.optim import Optimizer
from fairscale.optim.oss import OSS
import logging

logger = logging.getLogger(__name__)

class ShardedOptimizer:
    """Wrapper for FairScale's OSS (Optionally Sharded Optimizer) with additional features.
    
    This implementation uses FairScale's optimizer sharding to reduce memory usage during
    training by partitioning optimizer states across multiple devices. It includes additional
    features like gradient accumulation and automatic bucket size tuning.
    
    Args:
        optimizer_cls: Base optimizer class (e.g. AdamW)
        model_params (List): List of model parameter groups
        optimizer_kwargs (Dict): Arguments for base optimizer
        bucket_cap_mb (int, optional): Maximum size of gradient buckets in MB. Defaults to 100.
        auto_tune_buckets (bool, optional): Whether to auto-tune bucket sizes. Defaults to True.
        grad_accum_steps (int, optional): Number of gradient accumulation steps. Defaults to 1.
    """
    
    def __init__(
        self,
        optimizer_cls: Any,
        model_params: List,
        optimizer_kwargs: Dict,
        bucket_cap_mb: int = 100,
        auto_tune_buckets: bool = True,
        grad_accum_steps: int = 1
    ):
        self.grad_accum_steps = grad_accum_steps
        self.current_step = 0
        
        # Initialize base optimizer
        base_optimizer = optimizer_cls(model_params, **optimizer_kwargs)
        
        # Wrap with FairScale OSS
        self.optimizer = OSS(
            params=base_optimizer.param_groups,
            optim=base_optimizer,
            bucket_cap_mb=bucket_cap_mb,
            auto_tune_bucket_size=auto_tune_buckets
        )
        
        logger.info(
            f"Initialized sharded optimizer with {grad_accum_steps} "
            f"gradient accumulation steps"
        )
        
    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zeros gradients if at the start of accumulation cycle.
        
        Args:
            set_to_none (bool, optional): Whether to set grads to None. Defaults to True.
        """
        if self.current_step % self.grad_accum_steps == 0:
            self.optimizer.zero_grad(set_to_none=set_to_none)
            
    def step(self) -> None:
        """Performs optimization step if at end of accumulation cycle."""
        self.current_step += 1
        
        if self.current_step % self.grad_accum_steps == 0:
            # Scale gradients by accumulation steps
            if self.grad_accum_steps > 1:
                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.div_(self.grad_accum_steps)
                            
            self.optimizer.step()
            
    def state_dict(self) -> Dict:
        """Gets optimizer state for checkpointing.
        
        Returns:
            Dict: Optimizer state dictionary
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'current_step': self.current_step
        }
        
    def load_state_dict(self, state_dict: Dict) -> None:
        """Loads optimizer state from checkpoint.
        
        Args:
            state_dict (Dict): Optimizer state dictionary
        """
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.current_step = state_dict['current_step']
        
    @property
    def param_groups(self):
        """Access to underlying optimizer param groups."""
        return self.optimizer.param_groups
        
    def consolidate(self) -> None:
        """Consolidates sharded optimizer states for checkpointing."""
        self.optimizer.consolidate_state_dict()
        
    def broadcast_params(self) -> None:
        """Broadcasts consolidated parameters across devices."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    torch.distributed.broadcast(p.data, src=0)
                    
    def partition_parameters(self) -> None:
        """Re-partitions parameters after loading from checkpoint."""
        self.optimizer.partition_parameters()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Gets optimizer memory usage statistics.
        
        Returns:
            Dict[str, float]: Memory usage in GB for different components
        """
        usage = {}
        
        # State memory
        state_mem = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state:
                    state = self.optimizer.state[p]
                    for v in state.values():
                        if torch.is_tensor(v):
                            state_mem += v.numel() * v.element_size()
                            
        usage['optimizer_state_gb'] = state_mem / (1024 ** 3)
        
        # Parameter memory
        param_mem = 0
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_mem += p.numel() * p.element_size()
                if p.grad is not None:
                    param_mem += p.grad.numel() * p.grad.element_size()
                    
        usage['parameter_memory_gb'] = param_mem / (1024 ** 3)
        usage['total_memory_gb'] = usage['optimizer_state_gb'] + usage['parameter_memory_gb']
        
        return usage
