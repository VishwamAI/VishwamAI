"""Communication operations for distributed training."""
from typing import List, Optional, Tuple, Union
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group

def all_to_all(
    input_tensor: torch.Tensor,
    output_splits: Optional[List[int]] = None,
    input_splits: Optional[List[int]] = None,
    group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """All-to-all communication primitive.
    
    Args:
        input_tensor: Input tensor to scatter
        output_splits: Optional list of sizes to scatter
        input_splits: Optional list of sizes to gather
        group: Process group for communication
        
    Returns:
        Gathered tensor
    """
    if group is None:
        group = _get_default_group()
        
    world_size = dist.get_world_size(group)
    
    # Default to equal splits
    if output_splits is None:
        output_splits = [input_tensor.size(0) // world_size] * world_size
    if input_splits is None:
        input_splits = [input_tensor.size(0) // world_size] * world_size
        
    # Allocate output tensor
    total_size = sum(output_splits)
    output = input_tensor.new_empty(total_size, *input_tensor.size()[1:])
    
    # Perform all-to-all
    dist.all_to_all_single(
        output,
        input_tensor,
        output_splits,
        input_splits,
        group=group
    )
    
    return output

def all_gather(
    tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None
) -> List[torch.Tensor]:
    """All-gather communication primitive.
    
    Args:
        tensor: Input tensor to gather
        group: Process group for communication
        
    Returns:
        List of gathered tensors
    """
    if group is None:
        group = _get_default_group()
        
    world_size = dist.get_world_size(group)
    output = [torch.empty_like(tensor) for _ in range(world_size)]
    
    dist.all_gather(output, tensor, group=group)
    return output

def all_reduce(
    tensor: torch.Tensor,
    op: str = 'sum',
    group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """All-reduce communication primitive.
    
    Args:
        tensor: Input tensor to reduce
        op: Reduction operation ['sum', 'avg', 'product', 'min', 'max']
        group: Process group for communication
        
    Returns:
        Reduced tensor
    """
    if group is None:
        group = _get_default_group()
        
    # Map string ops to torch.distributed.ReduceOp
    op_map = {
        'sum': dist.ReduceOp.SUM,
        'avg': dist.ReduceOp.AVG,
        'product': dist.ReduceOp.PRODUCT,
        'min': dist.ReduceOp.MIN,
        'max': dist.ReduceOp.MAX
    }
    reduce_op = op_map.get(op, dist.ReduceOp.SUM)
    
    dist.all_reduce(tensor, op=reduce_op, group=group)
    return tensor

def reduce_scatter(
    input_list: List[torch.Tensor],
    op: str = 'sum',
    group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """Reduce-scatter communication primitive.
    
    Args:
        input_list: List of input tensors to reduce and scatter
        op: Reduction operation ['sum', 'avg', 'product', 'min', 'max']
        group: Process group for communication
        
    Returns:
        Local shard of reduced tensor
    """
    if group is None:
        group = _get_default_group()
        
    # Map string ops to torch.distributed.ReduceOp
    op_map = {
        'sum': dist.ReduceOp.SUM,
        'avg': dist.ReduceOp.AVG,
        'product': dist.ReduceOp.PRODUCT,
        'min': dist.ReduceOp.MIN,
        'max': dist.ReduceOp.MAX
    }
    reduce_op = op_map.get(op, dist.ReduceOp.SUM)
    
    # Pre-allocate output tensor
    output = torch.empty_like(input_list[0])
    
    dist.reduce_scatter(output, input_list, op=reduce_op, group=group)
    return output

def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None
) -> torch.Tensor:
    """Broadcast communication primitive.
    
    Args:
        tensor: Tensor to broadcast (only needed on source rank)
        src: Source rank for broadcast
        group: Process group for communication
        
    Returns:
        Broadcasted tensor
    """
    if group is None:
        group = _get_default_group()
        
    dist.broadcast(tensor, src=src, group=group)
    return tensor

def expert_to_expert(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    expert_capacity: int,
    group: Optional[dist.ProcessGroup] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expert-to-expert token routing communication.
    
    Args:
        tokens: Input tokens [batch, seq_len, hidden]
        scores: Expert scores [batch, seq_len, num_experts]
        expert_capacity: Maximum tokens per expert
        group: Process group for communication
        
    Returns:
        Tuple of:
        - Dispatched tokens
        - Assignment matrix
        - Combine weights
    """
    if group is None:
        group = _get_default_group()
        
    world_size = dist.get_world_size(group)
    
    # Get expert assignments
    num_experts = scores.size(-1) // world_size
    expert_mask = torch.zeros_like(scores)
    expert_mask[...,:num_experts] = 1
    
    # Sort scores and mask out overflow
    scores = scores * expert_mask
    _, indices = scores.sort(dim=-1, descending=True)
    indices = indices[...,:expert_capacity]
    
    # Create dispatch weights
    weights = torch.zeros_like(scores)
    weights.scatter_(-1, indices, scores.gather(-1, indices))
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-9)
    
    # Dispatch tokens
    token_splits = [tokens.size(1) // world_size] * world_size
    weight_splits = [num_experts] * world_size
    
    dispatched_tokens = all_to_all(tokens, token_splits, token_splits, group)
    dispatched_weights = all_to_all(weights, weight_splits, weight_splits, group)
    
    return dispatched_tokens, expert_mask, dispatched_weights
