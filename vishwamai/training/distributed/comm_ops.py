"""Communication operations for TPU distributed training."""

from typing import Optional, List, Tuple, Union, Any
import functools

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from ...utils.logging import get_logger

logger = get_logger(__name__)

def tpu_barrier():
    """TPU-specific barrier synchronization."""
    xm.mark_step()
    xm.rendezvous("tpu_barrier")

def all_reduce(
    tensor: torch.Tensor,
    op: str = "sum",
    group: Optional[Any] = None
) -> torch.Tensor:
    """Perform all-reduce operation on TPUs.
    
    Args:
        tensor: Input tensor
        op: Reduction operation ("sum", "avg", "max", "min")
        group: Optional process group
        
    Returns:
        Reduced tensor
    """
    if op == "sum":
        reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
    elif op == "avg":
        reduced = xm.all_reduce(xm.REDUCE_MEAN, tensor)
    elif op == "max":
        reduced = xm.all_reduce(xm.REDUCE_MAX, tensor)
    elif op == "min":
        reduced = xm.all_reduce(xm.REDUCE_MIN, tensor)
    else:
        raise ValueError(f"Unknown reduction operation: {op}")
        
    return reduced

def all_gather(
    tensor: torch.Tensor,
    dim: int = 0,
    group: Optional[Any] = None
) -> List[torch.Tensor]:
    """Gather tensors from all TPU cores.
    
    Args:
        tensor: Input tensor
        dim: Dimension to gather on
        group: Optional process group
        
    Returns:
        List of gathered tensors
    """
    world_size = xm.xrt_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered

def reduce_scatter(
    tensor: torch.Tensor,
    op: str = "sum",
    group: Optional[Any] = None
) -> torch.Tensor:
    """Reduce-scatter operation on TPUs.
    
    Args:
        tensor: Input tensor
        op: Reduction operation ("sum", "avg", "max", "min")
        group: Optional process group
        
    Returns:
        Reduced and scattered tensor
    """
    if op == "sum":
        reduce_op = xm.REDUCE_SUM
    elif op == "avg":
        reduce_op = xm.REDUCE_MEAN
    elif op == "max":
        reduce_op = xm.REDUCE_MAX
    elif op == "min":
        reduce_op = xm.REDUCE_MIN
    else:
        raise ValueError(f"Unknown reduction operation: {op}")
        
    return xm.reduce_scatter(reduce_op, tensor)

def broadcast(
    tensor: torch.Tensor,
    src: int = 0,
    group: Optional[Any] = None
) -> torch.Tensor:
    """Broadcast tensor from source rank to all TPU cores.
    
    Args:
        tensor: Input tensor
        src: Source rank
        group: Optional process group
        
    Returns:
        Broadcasted tensor
    """
    return dist.broadcast(tensor, src)

@functools.lru_cache(maxsize=None)
def get_global_batch_shape(
    local_batch: torch.Tensor,
    batch_dim: int = 0
) -> torch.Size:
    """Get global batch shape across all TPU cores.
    
    Args:
        local_batch: Local batch tensor
        batch_dim: Batch dimension
        
    Returns:
        Global batch shape
    """
    local_shape = list(local_batch.shape)
    world_size = xm.xrt_world_size()
    local_shape[batch_dim] *= world_size
    return torch.Size(local_shape)

def pad_tensor_to_multiple(
    tensor: torch.Tensor,
    multiple: int,
    dim: int = 0,
    value: float = 0.0
) -> Tuple[torch.Tensor, int]:
    """Pad tensor dimension to multiple for TPU efficiency.
    
    Args:
        tensor: Input tensor
        multiple: Multiple to pad to
        dim: Dimension to pad
        value: Padding value
        
    Returns:
        Tuple containing:
        - Padded tensor
        - Number of padding elements added
    """
    size = tensor.size(dim)
    pad_size = (multiple - size % multiple) % multiple
    
    if pad_size == 0:
        return tensor, 0
        
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    padded = torch.cat([tensor, padding], dim=dim)
    
    return padded, pad_size

def split_tensor_along_dim(
    tensor: torch.Tensor,
    dim: int = 0,
    num_chunks: Optional[int] = None
) -> List[torch.Tensor]:
    """Split tensor along dimension for TPU cores.
    
    Args:
        tensor: Input tensor
        dim: Dimension to split
        num_chunks: Optional number of chunks (defaults to world size)
        
    Returns:
        List of split tensors
    """
    if num_chunks is None:
        num_chunks = xm.xrt_world_size()
        
    chunk_size = tensor.size(dim) // num_chunks
    return list(tensor.split(chunk_size, dim=dim))

def gather_from_tensor_model_parallel(
    tensor: torch.Tensor,
    dim: int = 0,
    dest_rank: int = 0
) -> Optional[torch.Tensor]:
    """Gather tensor from tensor model parallel TPU cores.
    
    Args:
        tensor: Input tensor
        dim: Dimension to gather on
        dest_rank: Destination rank to gather to
        
    Returns:
        Gathered tensor on destination rank, None on other ranks
    """
    if xm.get_ordinal() == dest_rank:
        gathered = [torch.zeros_like(tensor) for _ in range(xm.xrt_world_size())]
        dist.all_gather(gathered, tensor)
        return torch.cat(gathered, dim=dim)
    else:
        dist.all_gather([], tensor)
        return None

def init_device_mesh(
    mesh_shape: Tuple[int, ...],
    mesh_names: Optional[List[str]] = None
) -> torch.Tensor:
    """Initialize TPU device mesh for SPMD.
    
    Args:
        mesh_shape: Shape of device mesh
        mesh_names: Optional names for mesh dimensions
        
    Returns:
        Device mesh tensor
    """
    world_size = xm.xrt_world_size()
    
    if mesh_names is None:
        mesh_names = [f"dim_{i}" for i in range(len(mesh_shape))]
        
    if world_size != torch.prod(torch.tensor(mesh_shape)):
        raise ValueError(
            f"World size {world_size} does not match mesh shape {mesh_shape}"
        )
        
    rank = xm.get_ordinal()
    coords = []
    for dim_size in reversed(mesh_shape):
        coords.append(rank % dim_size)
        rank //= dim_size
        
    coords = coords[::-1]
    mesh = torch.arange(world_size).reshape(mesh_shape)
    
    logger.info(f"Initialized {len(mesh_shape)}D device mesh: {mesh_shape}")
    return mesh
