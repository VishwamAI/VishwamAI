"""Utilities for tensor operations and memory management."""
import jax
import jax.numpy as jnp
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
import gc
from functools import partial

def chunk_tensor(
    tensor: Union[jnp.ndarray, np.ndarray],
    chunk_size: int,
    clear_cache: bool = True
) -> List[jnp.ndarray]:
    """Split tensor into chunks for memory-efficient processing.
    
    Args:
        tensor: Input tensor to chunk
        chunk_size: Size of each chunk
        clear_cache: Whether to clear JAX cache between chunks
        
    Returns:
        List of tensor chunks
    """
    chunks = []
    for i in range(0, tensor.shape[0], chunk_size):
        end = min(i + chunk_size, tensor.shape[0])
        chunk = jnp.array(tensor[i:end])
        chunks.append(chunk)
        
        if clear_cache:
            jax.clear_caches()
            gc.collect()
    
    return chunks

def merge_chunks(
    chunks: List[jnp.ndarray],
    axis: int = 0,
    clear_cache: bool = True
) -> jnp.ndarray:
    """Merge tensor chunks back together.
    
    Args:
        chunks: List of tensor chunks to merge
        axis: Axis along which to concatenate
        clear_cache: Whether to clear JAX cache after merging
        
    Returns:
        Merged tensor
    """
    merged = jnp.concatenate(chunks, axis=axis)
    
    if clear_cache:
        jax.clear_caches()
        gc.collect()
    
    return merged

def apply_chunked(
    fn: callable,
    tensor: jnp.ndarray,
    chunk_size: int,
    clear_cache: bool = True,
    **fn_kwargs
) -> jnp.ndarray:
    """Apply a function to tensor in chunks to manage memory.
    
    Args:
        fn: Function to apply to each chunk
        tensor: Input tensor
        chunk_size: Size of each chunk
        clear_cache: Whether to clear JAX cache between operations
        fn_kwargs: Additional keyword arguments for fn
        
    Returns:
        Processed tensor
    """
    # Split into chunks
    chunks = chunk_tensor(tensor, chunk_size, clear_cache)
    
    # Process each chunk
    processed_chunks = []
    for chunk in chunks:
        processed = fn(chunk, **fn_kwargs)
        processed_chunks.append(processed)
        
        if clear_cache:
            jax.clear_caches()
            gc.collect()
    
    # Merge results
    return merge_chunks(processed_chunks, clear_cache=clear_cache)

def load_tensor_chunks(
    safetensor_file: str,
    key: str,
    chunk_size: int,
    dtype: Optional[jnp.dtype] = None,
    clear_cache: bool = True
) -> jnp.ndarray:
    """Load a tensor from safetensors file in chunks.
    
    Args:
        safetensor_file: Path to safetensors file
        key: Key of tensor to load
        chunk_size: Size of each chunk to load
        dtype: Optional dtype to convert tensor to
        clear_cache: Whether to clear JAX cache between operations
        
    Returns:
        Loaded tensor
    """
    import safetensors
    
    with safetensors.safe_open(safetensor_file, framework="flax") as f:
        info = f.get_tensor_info(key)
        shape = info['shape']
        
        chunks = []
        for start_idx in range(0, shape[0], chunk_size):
            end_idx = min(start_idx + chunk_size, shape[0])
            chunk = f.get_slice(key, start_idx, end_idx)
            
            if dtype is not None:
                chunk = chunk.astype(dtype)
            
            chunks.append(chunk)
            
            if clear_cache:
                jax.clear_caches()
                gc.collect()
    
    return merge_chunks(chunks, clear_cache=clear_cache)

def get_memory_usage() -> float:
    """Get current process memory usage in GB."""
    import psutil
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)

def estimate_tensor_size(shape: Tuple[int, ...], dtype: jnp.dtype) -> float:
    """Estimate size of tensor in GB."""
    size_bytes = np.prod(shape) * dtype.itemsize
    return size_bytes / (1024 * 1024 * 1024)

def suggest_chunk_size(
    shape: Tuple[int, ...],
    dtype: jnp.dtype,
    target_chunk_gb: float = 0.5
) -> int:
    """Suggest chunk size to keep memory usage under target.
    
    Args:
        shape: Shape of full tensor
        dtype: Dtype of tensor
        target_chunk_gb: Target size for each chunk in GB
        
    Returns:
        Suggested chunk size
    """
    full_size_gb = estimate_tensor_size(shape, dtype)
    chunk_ratio = max(1, int(np.ceil(full_size_gb / target_chunk_gb)))
    return max(1, shape[0] // chunk_ratio)

class ChunkedTensorCache:
    """Cache for managing chunked tensors with memory awareness."""
    
    def __init__(self, max_cache_gb: float = 4.0):
        self.max_cache_gb = max_cache_gb
        self.cache: Dict[str, jnp.ndarray] = {}
        self.cache_sizes: Dict[str, float] = {}
    
    def add(self, key: str, tensor: jnp.ndarray) -> None:
        """Add tensor to cache if there's space."""
        tensor_gb = estimate_tensor_size(tensor.shape, tensor.dtype)
        
        # Check if we need to clear space
        current_size = sum(self.cache_sizes.values())
        if current_size + tensor_gb > self.max_cache_gb:
            self._clear_oldest()
        
        self.cache[key] = tensor
        self.cache_sizes[key] = tensor_gb
    
    def get(self, key: str) -> Optional[jnp.ndarray]:
        """Get tensor from cache."""
        return self.cache.get(key)
    
    def _clear_oldest(self) -> None:
        """Clear oldest items from cache until under max size."""
        while self.cache and sum(self.cache_sizes.values()) > self.max_cache_gb:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.cache_sizes[oldest_key]
            jax.clear_caches()
            gc.collect()

# Global cache instance
TENSOR_CACHE = ChunkedTensorCache()
