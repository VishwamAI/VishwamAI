"""FairScale sharded optimizer implementations."""
from typing import Dict, List, Optional, Any
import torch
from torch.optim import Optimizer
import torch.distributed as dist
from fairscale.optim import OSS

class ShardedOptimizer(OSS):
    """Base class for sharded optimizers.
    
    This implementation extends FairScale's OSS (Optimizer State Sharding)
    with additional features for MoE training.
    
    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups
        optim: Optimizer class to shard (e.g. torch.optim.Adam)
        group: Process group for sharding
        process_group: Process group for grad reduction
        broadcast_buffer_size: Size of buffer for broadcasting parameters
        force_broadcast: Whether to force parameter broadcast
    """
    
    def __init__(
        self,
        params,
        optim: type,
        group: Optional[Any] = None,
        process_group: Optional[Any] = None,
        broadcast_buffer_size: int = 2**23,  # 8MB
        force_broadcast: bool = False,
        **defaults
    ):
        super().__init__(
            params,
            optim,
            group=group,
            process_group=process_group,
            broadcast_buffer_size=broadcast_buffer_size,
            force_broadcast=force_broadcast,
            **defaults
        )
        
    def _broadcast_params(self) -> None:
        """Broadcast parameters to all processes in group."""
        for param_group in self.param_groups:
            for p in param_group['params']:
                if not self._is_root and p.device != self.device:
                    p.data = p.data.to(self.device)
                    
                dist.broadcast(p.data, src=self._root_rank, group=self.group)
                
    def consolidate_state_dict(self) -> Dict[str, Any]:
        """Consolidate a state_dict from sharded model state.
        
        Returns:
            Consolidated state dict
        """
        if not dist.is_initialized():
            return self.state_dict()
            
        state_dict = self.state_dict()
        consolidated = {}
        world_size = dist.get_world_size(self.group)
        
        # Gather state from all ranks
        for k, v in state_dict.items():
            if torch.is_tensor(v):
                consolidated[k] = [torch.zeros_like(v) for _ in range(world_size)]
                dist.all_gather(consolidated[k], v, group=self.group)
                consolidated[k] = torch.stack(consolidated[k]).mean(0)
            else:
                consolidated[k] = v
                
        return consolidated

class ShardedAdam(ShardedOptimizer):
    """Sharded Adam optimizer with weight decay fix.
    
    This implements the Adam algorithm in a sharded fashion,
    with proper weight decay and expert parameter handling.
    
    Args:
        params: Iterable of parameters
        lr: Learning rate
        betas: Coefficients for moving averages
        eps: Term added for numerical stability
        weight_decay: Weight decay factor
        amsgrad: Whether to use AMSGrad variant
        group: Process group for sharding
        process_group: Process group for grad reduction
        broadcast_buffer_size: Buffer size for broadcasting
        expert_parallel: Whether params are expert parallel
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        group: Optional[Any] = None,
        process_group: Optional[Any] = None,
        broadcast_buffer_size: int = 2**23,
        expert_parallel: bool = False,
        **kwargs
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad
        )
        
        super().__init__(
            params,
            torch.optim.Adam,
            group=group,
            process_group=process_group,
            broadcast_buffer_size=broadcast_buffer_size,
            **defaults
        )
        
        self.expert_parallel = expert_parallel
        
    @torch.no_grad()
    def step(self, closure=None):
        """Perform optimization step.
        
        Args:
            closure: Closure that reevaluates model and returns loss
            
        Returns:
            Loss from closure if provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # Update parameters in groups
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                # Handle expert parallel params specially
                if self.expert_parallel:
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("Sparse gradients not supported for expert params")
                else:
                    # Reduce gradients across processes
                    if dist.is_initialized():
                        dist.all_reduce(p.grad, group=self.process_group)
                        p.grad.div_(dist.get_world_size(self.process_group))
                        
        # Perform shard-local optimizer step
        loss = super().step(closure=None)
        
        # Broadcast updated parameters
        if not self.expert_parallel:
            self._broadcast_params()
            
        return loss
