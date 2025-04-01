"""TPU memory management and optimization system."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, Optional, Tuple, List, NamedTuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class MemoryLayout(Enum):
    """TPU memory layout patterns."""
    BLOCKED = "blocked"
    STRIDED = "strided"
    CHUNKED = "chunked"
    INTERLEAVED = "interleaved"

@dataclass
class MemoryConfig:
    """Configuration for TPU memory management."""
    block_size: int = 128
    num_cores: int = 8
    use_bfloat16: bool = True
    prefetch_distance: int = 2
    max_live_buffers: int = 32

class MemoryAllocation(NamedTuple):
    """Memory allocation details."""
    device_buffer: Any
    shape: Tuple[int, ...]
    dtype: jnp.dtype
    layout: MemoryLayout
    is_sharded: bool

class TPUMemoryManager:
    """
    Memory management system optimized for TPU.
    
    Features:
    - Smart memory allocation
    - Automatic sharding
    - Prefetch optimization
    - Memory defragmentation
    - Layout optimization
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize TPU memory manager.
        
        Args:
            config: Optional memory configuration
        """
        self.config = config or MemoryConfig()
        if self.config.block_size % 128 != 0:
            raise ValueError("Block size must be multiple of 128 for TPU")
            
        self.allocations: Dict[str, MemoryAllocation] = {}
        self.device = jax.devices()[0]
        self._initialize()
        
    def _initialize(self):
        """Initialize TPU memory system."""
        # Get TPU core information
        self.memory_per_core = self.device.memory_per_core
        self.num_cores = len(jax.devices())
        
        # Initialize memory pools
        self.free_memory = self.memory_per_core * self.num_cores
        self.peak_memory = 0
        self.fragmentation = 0.0
        
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: jnp.dtype = jnp.float32,
        layout: Optional[MemoryLayout] = None,
        name: Optional[str] = None,
        shard: bool = True
    ) -> MemoryAllocation:
        """
        Allocate TPU memory with optimal layout.
        
        Args:
            shape: Tensor shape
            dtype: Data type
            layout: Memory layout pattern
            name: Optional buffer name
            shard: Whether to shard across TPU cores
            
        Returns:
            Memory allocation details
        """
        # Calculate required memory
        element_size = np.dtype(dtype).itemsize
        total_size = np.prod(shape) * element_size
        
        # Check memory availability
        if total_size > self.free_memory:
            self._defragment()
            if total_size > self.free_memory:
                raise MemoryError("Not enough TPU memory available")
                
        # Determine optimal layout
        if layout is None:
            layout = self._get_optimal_layout(shape, dtype)
            
        # Pad dimensions for TPU efficiency
        padded_shape = self._pad_shape(shape)
        
        # Allocate device buffer
        if shard and self.num_cores > 1:
            # Shard across TPU cores
            devices = np.array(jax.devices()).reshape(-1, 1)
            mesh = jax.sharding.Mesh(devices, ['batch'])
            
            buffer = jax.device_put(
                jnp.zeros(padded_shape, dtype=dtype),
                jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('batch'))
            )
        else:
            buffer = jax.device_put(
                jnp.zeros(padded_shape, dtype=dtype)
            )
            
        # Create allocation record
        allocation = MemoryAllocation(
            device_buffer=buffer,
            shape=padded_shape,
            dtype=dtype,
            layout=layout,
            is_sharded=shard
        )
        
        # Update memory tracking
        self.free_memory -= total_size
        self.peak_memory = max(self.peak_memory, total_size)
        if name:
            self.allocations[name] = allocation
            
        return allocation
        
    def free(self, name: str):
        """Free allocated memory."""
        if name in self.allocations:
            allocation = self.allocations[name]
            size = np.prod(allocation.shape) * np.dtype(allocation.dtype).itemsize
            self.free_memory += size
            del self.allocations[name]
            
    def defragment(self):
        """Defragment TPU memory."""
        # Collect live allocations
        live_allocs = list(self.allocations.items())
        
        # Sort by size for better packing
        live_allocs.sort(key=lambda x: np.prod(x[1].shape))
        
        # Clear current allocations
        self.allocations.clear()
        self.free_memory = self.memory_per_core * self.num_cores
        
        # Reallocate in sorted order
        for name, alloc in live_allocs:
            new_alloc = self.allocate(
                alloc.shape,
                alloc.dtype,
                alloc.layout,
                name,
                alloc.is_sharded
            )
            # Copy data to new location
            new_alloc.device_buffer.copy_from_host_async(
                alloc.device_buffer.to_py()
            )
            
    def optimize_layout(
        self,
        tensor: jnp.ndarray,
        layout: MemoryLayout
    ) -> jnp.ndarray:
        """
        Optimize tensor layout for TPU memory access.
        
        Args:
            tensor: Input tensor
            layout: Desired memory layout
            
        Returns:
            Tensor with optimized layout
        """
        if layout == MemoryLayout.BLOCKED:
            return self._to_blocked_layout(tensor)
        elif layout == MemoryLayout.STRIDED:
            return self._to_strided_layout(tensor)
        elif layout == MemoryLayout.CHUNKED:
            return self._to_chunked_layout(tensor)
        elif layout == MemoryLayout.INTERLEAVED:
            return self._to_interleaved_layout(tensor)
        return tensor
        
    def _to_blocked_layout(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Convert to blocked memory layout."""
        shape = tensor.shape
        if len(shape) < 2:
            return tensor
            
        # Compute block dimensions
        block_rows = (shape[-2] + self.config.block_size - 1) // self.config.block_size
        block_cols = (shape[-1] + self.config.block_size - 1) // self.config.block_size
        
        # Reshape and transpose for block access
        tensor = tensor.reshape(
            *shape[:-2],
            block_rows, self.config.block_size,
            block_cols, self.config.block_size
        )
        return tensor.transpose(
            *range(len(shape)-2),
            -4, -2, -3, -1
        )
        
    def _to_strided_layout(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Convert to strided memory layout."""
        shape = tensor.shape
        if len(shape) < 2:
            return tensor
            
        # Compute strides for TPU efficiency
        stride = self.config.block_size
        rows, cols = shape[-2:]
        
        # Reshape with strides
        return tensor.reshape(
            *shape[:-2],
            rows // stride, stride,
            cols // stride, stride
        ).transpose(
            *range(len(shape)-2),
            -4, -2, -3, -1
        )
        
    def _to_chunked_layout(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Convert to chunked memory layout."""
        shape = tensor.shape
        chunk_size = self.config.block_size
        
        # Split into chunks
        chunks = []
        for i in range(0, shape[-1], chunk_size):
            chunk = tensor[..., i:i+chunk_size]
            if chunk.shape[-1] < chunk_size:
                # Pad last chunk
                pad_width = [(0, 0)] * (len(shape)-1) + [(0, chunk_size - chunk.shape[-1])]
                chunk = jnp.pad(chunk, pad_width)
            chunks.append(chunk)
            
        return jnp.stack(chunks, axis=-2)
        
    def _to_interleaved_layout(self, tensor: jnp.ndarray) -> jnp.ndarray:
        """Convert to interleaved memory layout."""
        shape = tensor.shape
        if len(shape) < 2:
            return tensor
            
        # Compute interleaving pattern
        rows, cols = shape[-2:]
        block = self.config.block_size
        
        # Reshape and transpose for interleaved access
        return tensor.reshape(
            *shape[:-2],
            rows // block, block,
            cols // block, block
        ).transpose(
            *range(len(shape)-2),
            -4, -2, -3, -1
        ).reshape(*shape[:-2], rows, cols)
        
    def _get_optimal_layout(
        self,
        shape: Tuple[int, ...],
        dtype: jnp.dtype
    ) -> MemoryLayout:
        """Determine optimal memory layout."""
        if len(shape) <= 1:
            return MemoryLayout.STRIDED
            
        size = np.prod(shape) * np.dtype(dtype).itemsize
        if size < self.memory_per_core:
            # Small tensors: use blocked layout
            return MemoryLayout.BLOCKED
        elif len(shape) >= 3:
            # Multi-dimensional: use chunked layout
            return MemoryLayout.CHUNKED
        else:
            # Large 2D: use interleaved layout
            return MemoryLayout.INTERLEAVED
            
    def _pad_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Pad shape to TPU-efficient dimensions."""
        if len(shape) <= 1:
            return shape
            
        padded = []
        for dim in shape[:-2]:
            padded.append(dim)
            
        # Pad last two dimensions to block size
        for dim in shape[-2:]:
            padded_dim = ((dim + self.config.block_size - 1) // 
                         self.config.block_size * self.config.block_size)
            padded.append(padded_dim)
            
        return tuple(padded)
        
    def _defragment(self):
        """Internal memory defragmentation."""
        total_size = sum(
            np.prod(alloc.shape) * np.dtype(alloc.dtype).itemsize
            for alloc in self.allocations.values()
        )
        
        self.fragmentation = 1.0 - (total_size / (self.memory_per_core * self.num_cores))
        
        if self.fragmentation > 0.2:  # 20% fragmentation threshold
            self.defragment()