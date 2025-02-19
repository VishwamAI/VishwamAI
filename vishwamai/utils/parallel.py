"""
Utilities for model parallel training and inference.

This module provides functions and utilities to support model parallelism
across multiple devices/GPUs.
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple, Union, Any, Callable
import functools

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1
) -> Tuple[torch.device, ...]:
    """
    Initialize model parallel execution environment.
    
    Args:
        tensor_model_parallel_size: Number of GPUs to split tensors across
        pipeline_model_parallel_size: Number of GPUs for pipeline parallelism
        
    Returns:
        Tuple of devices allocated for model parallel execution
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
        
    world_size = dist.get_world_size()
    assert world_size == tensor_model_parallel_size * pipeline_model_parallel_size, \
        "World size must equal tensor_parallel_size * pipeline_parallel_size"
        
    # Set up process groups
    for i in range(pipeline_model_parallel_size):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        pg = dist.new_group(ranks)
        if i == dist.get_rank() // tensor_model_parallel_size:
            tensor_model_parallel_group = pg
            
    # Assign devices
    devices = []
    for i in range(pipeline_model_parallel_size):
        for j in range(tensor_model_parallel_size):
            devices.append(torch.device(f'cuda:{i * tensor_model_parallel_size + j}'))
            
    return tuple(devices)

def get_model_parallel_rank() -> int:
    """Get rank of current process in model parallel group."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_model_parallel_world_size() -> int:
    """Get world size of model parallel group."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def model_parallel_forward(
    module: torch.nn.Module,
    *args,
    **kwargs
) -> Any:
    """
    Execute forward pass with model parallelism.
    
    Args:
        module: PyTorch module to execute
        *args: Positional arguments for module forward
        **kwargs: Keyword arguments for module forward
        
    Returns:
        Output of module's forward pass
    """
    if not dist.is_initialized():
        return module(*args, **kwargs)
        
    # Get device placement
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    device = torch.device(f'cuda:{rank}')
    
    # Move inputs to correct device
    args = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg 
                for arg in args)
    kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
             for k, v in kwargs.items()}
    
    # Execute forward pass
    outputs = module(*args, **kwargs)
    
    # Gather results if needed
    if isinstance(outputs, torch.Tensor):
        dist.all_reduce(outputs)
    elif isinstance(outputs, (tuple, list)):
        for out in outputs:
            if isinstance(out, torch.Tensor):
                dist.all_reduce(out)
                
    return outputs

def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False
) -> Tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.
    
    Args:
        tensor: Input tensor to split
        num_partitions: Number of splits
        contiguous_split_chunks: If True, make splits contiguous in memory
        
    Returns:
        Tuple of split tensors
    """
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)
    return tensor_list

def gather_from_model_parallel_region(
    tensor: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """Gather tensors from model parallel region.
    
    Args:
        tensor: Input tensor
        dim: Dimension along which to gather
        
    Returns:
        Gathered tensor
    """
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return tensor
        
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=dim)

def scatter_to_model_parallel_region(
    tensor: torch.Tensor,
    dim: int = 0
) -> torch.Tensor:
    """Scatter tensor to model parallel region.
    
    Args:
        tensor: Input tensor
        dim: Dimension along which to scatter
        
    Returns:
        Local portion of scattered tensor
    """
    world_size = get_model_parallel_world_size()
    if world_size == 1:
        return tensor
        
    rank = get_model_parallel_rank()
    tensor_chunks = torch.chunk(tensor, world_size, dim=dim)
    return tensor_chunks[rank]

def model_parallel_sync(func: Callable) -> Callable:
    """Decorator to synchronize model parallel processes.
    
    Args:
        func: Function to decorate
        
    Returns:
        Wrapped function that synchronizes across processes
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if dist.is_initialized():
            torch.distributed.barrier()
        output = func(*args, **kwargs)
        if dist.is_initialized():
            torch.distributed.barrier()
        return output
    return wrapper
