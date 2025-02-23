"""Expert parallel training utilities."""
from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

class ExpertParallel(DistributedDataParallel):
    """Expert parallel wrapper for MoE models.
    
    Extends DistributedDataParallel to handle expert parallelism.
    Only expert parameters are sharded across devices, while
    shared parameters are replicated.
    
    Args:
        module: Model containing experts
        device_ids: List of device IDs
        output_device: Output device ID
        dim: Dimension to shard on
        broadcast_buffers: Whether to sync buffers
    """
    
    def __init__(
        self,
        module: nn.Module,
        device_ids: Optional[List[int]] = None,
        output_device: Optional[int] = None,
        dim: int = 0,
        broadcast_buffers: bool = True,
        **kwargs
    ):
        super().__init__(
            module,
            device_ids=device_ids,
            output_device=output_device,
            broadcast_buffers=broadcast_buffers,
            **kwargs
        )
        self.dim = dim
        
    def forward(self, *inputs, **kwargs):
        """Forward pass with expert sharding.
        
        Handles expert parameter sharding and gradient collection.
        """
        if not self.device_ids:
            return self.module(*inputs, **kwargs)
            
        # Shard experts before forward pass
        shard_expert_params(self.module)
        
        # Regular DDP forward
        outputs = super().forward(*inputs, **kwargs)
        
        # Gather expert grads after backward
        if self.training:
            gather_expert_grads(self.module)
            
        return outputs

def shard_expert_params(
    model: nn.Module,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> None:
    """Shard expert parameters across devices.
    
    Args:
        model: Model containing experts
        world_size: Number of devices
        rank: Device rank
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
        
    # Find expert parameters
    expert_params = []
    for name, param in model.named_parameters():
        if 'expert' in name:
            expert_params.append(param)
            
    # Compute shard sizes
    for param in expert_params:
        shard_size = param.size(0) // world_size
        start_idx = rank * shard_size
        end_idx = start_idx + shard_size if rank != world_size - 1 else param.size(0)
        
        # Keep only local shard
        param.data = param.data[start_idx:end_idx]
        if param.grad is not None:
            param.grad.data = param.grad.data[start_idx:end_idx]

def gather_expert_grads(
    model: nn.Module,
    world_size: Optional[int] = None,
    rank: Optional[int] = None
) -> None:
    """Gather expert gradients from all devices.
    
    Args:
        model: Model containing experts
        world_size: Number of devices
        rank: Device rank
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()
        
    expert_grads = []
    for name, param in model.named_parameters():
        if 'expert' in name and param.requires_grad:
            if param.grad is None:
                continue
            expert_grads.append(param.grad.data)
            
    # All-gather gradients
    for grad in expert_grads:
        grad_list = [torch.zeros_like(grad) for _ in range(world_size)]
        dist.all_gather(grad_list, grad)
        grad.data = torch.cat(grad_list, dim=0)

def get_expert_mapping(
    num_experts: int,
    world_size: int,
    strategy: str = 'uniform'
) -> Dict[int, List[int]]:
    """Get mapping of experts to devices.
    
    Args:
        num_experts: Total number of experts
        world_size: Number of devices
        strategy: Sharding strategy ['uniform', 'balanced', 'custom']
        
    Returns:
        Dict mapping device rank to list of expert indices
    """
    if strategy == 'uniform':
        # Uniformly distribute experts
        experts_per_device = num_experts // world_size
        mapping = {}
        for rank in range(world_size):
            start_idx = rank * experts_per_device
            end_idx = start_idx + experts_per_device
            if rank == world_size - 1:
                end_idx = num_experts
            mapping[rank] = list(range(start_idx, end_idx))
            
    elif strategy == 'balanced':
        # Balance expert computation
        mapping = {rank: [] for rank in range(world_size)}
        for expert_idx in range(num_experts):
            target_rank = expert_idx % world_size
            mapping[target_rank].append(expert_idx)
            
    else:
        raise ValueError(f"Unknown expert mapping strategy: {strategy}")
        
    return mapping

def rearrange_expert_outputs(
    outputs: torch.Tensor,
    dispatch_ids: torch.Tensor,
    gather_ids: torch.Tensor,
    world_size: Optional[int] = None
) -> torch.Tensor:
    """Rearrange expert outputs after all-to-all.
    
    Args:
        outputs: Expert outputs [tokens, hidden]
        dispatch_ids: Token to expert mapping
        gather_ids: Original token order
        world_size: Number of devices
        
    Returns:
        Reordered outputs in original token order
    """
    if world_size is None:
        world_size = dist.get_world_size()
        
    # Get output sizes for each device
    token_counts = [
        torch.scalar_tensor(outputs.size(0), device=outputs.device)
        for _ in range(world_size)
    ]
    dist.all_gather(token_counts, token_counts[0])
    
    # Pre-allocate output tensor
    total_tokens = sum(token_counts)
    hidden_size = outputs.size(1)
    gathered = outputs.new_zeros(total_tokens, hidden_size)
    
    # All-gather outputs
    outputs_list = [outputs.new_zeros(count, hidden_size) 
                   for count in token_counts]
    dist.all_gather(outputs_list, outputs)
    
    # Reorder based on gather_ids
    outputs = torch.cat(outputs_list, dim=0)
    return outputs[gather_ids]
