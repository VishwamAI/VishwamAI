"""TPU-optimized key/value cache management."""

import jax
import jax.numpy as jnp
from typing import Dict, Optional, NamedTuple, Tuple
from dataclasses import dataclass

@dataclass
class CacheConfig:
    """Configuration for TPU cache management."""
    max_length: int = 2048
    num_heads: int = 12
    head_dim: int = 64
    batch_size: int = 1
    dtype: jnp.dtype = jnp.float32
    
class CacheState(NamedTuple):
    """State of cached key/value tensors."""
    keys: jnp.ndarray
    values: jnp.ndarray
    length: int

class TPUCacheManager:
    """Manages TPU-optimized caching of key/value tensors."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheState] = {}
        self.max_length = config.max_length
        
    def allocate_cache(
        self,
        cache_id: str
    ) -> None:
        """Allocate new cache tensors."""
        if cache_id not in self.cache:
            # Initialize empty cache tensors
            self.cache[cache_id] = CacheState(
                keys=jnp.zeros((
                    self.config.batch_size,
                    self.config.num_heads,
                    0,  # Current length 0
                    self.config.head_dim
                ), dtype=self.config.dtype),
                values=jnp.zeros((
                    self.config.batch_size,
                    self.config.num_heads,
                    0,  # Current length 0
                    self.config.head_dim
                ), dtype=self.config.dtype),
                length=0
            )
            
    def update_cache(
        self,
        key: jnp.ndarray,
        value: jnp.ndarray,
        cache_id: str,
        position_ids: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update cache with new key/value tensors."""
        # Get sequence length of new tensors
        seq_len = key.shape[2]
        
        # Allocate cache if needed
        if cache_id not in self.cache:
            self.allocate_cache(cache_id)
            
        cache = self.cache[cache_id]
        
        # Handle position-based updates if position_ids provided
        if position_ids is not None:
            # Update specific positions
            indices = position_ids.reshape(-1)
            max_idx = jnp.max(indices)
            new_length = max_idx + 1
            
            # Expand cache if needed
            if new_length > cache.length:
                padded_keys = jnp.pad(
                    cache.keys,
                    ((0,0), (0,0), (0, new_length - cache.length), (0,0))
                )
                padded_values = jnp.pad(
                    cache.values,
                    ((0,0), (0,0), (0, new_length - cache.length), (0,0))
                )
                
                # Update at specified positions
                updated_keys = padded_keys.at[:,:,indices,:].set(key)
                updated_values = padded_values.at[:,:,indices,:].set(value)
                
                self.cache[cache_id] = CacheState(
                    keys=updated_keys,
                    values=updated_values,
                    length=new_length
                )
                
        else:
            # Append to end of cache
            new_length = cache.length + seq_len
            
            if new_length > self.max_length:
                # Rotate cache when max length exceeded
                start_idx = new_length - self.max_length
                self.cache[cache_id] = CacheState(
                    keys=jnp.concatenate([
                        cache.keys[:, :, start_idx:],
                        key
                    ], axis=2),
                    values=jnp.concatenate([
                        cache.values[:, :, start_idx:],
                        value
                    ], axis=2),
                    length=self.max_length
                )
            else:
                # Simply concatenate
                self.cache[cache_id] = CacheState(
                    keys=jnp.concatenate([cache.keys, key], axis=2),
                    values=jnp.concatenate([cache.values, value], axis=2),
                    length=new_length
                )
                
        return self.get_cache(cache_id)
        
    def get_cache(
        self,
        cache_id: str
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Retrieve cached key/value tensors."""
        if cache_id not in self.cache:
            raise KeyError(f"No cache found for id: {cache_id}")
            
        cache = self.cache[cache_id]
        return cache.keys, cache.values
        
    def clear_cache(
        self,
        cache_id: Optional[str] = None
    ) -> None:
        """Clear cache for given id or all caches."""
        if cache_id is not None:
            if cache_id in self.cache:
                del self.cache[cache_id]
        else:
            self.cache.clear()
            
    @property
    def cache_size(self) -> int:
        """Get total size of all caches in bytes."""
        total_size = 0
        for cache in self.cache.values():
            total_size += (
                cache.keys.nbytes +
                cache.values.nbytes
            )
        return total_size