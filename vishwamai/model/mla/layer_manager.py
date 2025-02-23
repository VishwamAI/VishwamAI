"""Layer manager for handling multi-layer state caching."""

from typing import Optional, List, Dict, Any, Deque
from collections import deque
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

class MLALayerManager(nn.Module):
    """Manages layer states for Multi-Layer Attention."""
    
    hidden_size: int
    num_layers: int
    max_cache_size: int = 8
    dtype: Any = jnp.float32
    deterministic: bool = False
    normalize_cached: bool = True
    use_compressed_cache: bool = False
    compression_dim: Optional[int] = None
    eviction_policy: str = "lru"  # Options: lru, fifo
    
    def setup(self):
        """Initialize layer manager components."""
        # Layer state compression if enabled
        if self.use_compressed_cache:
            self.compressor = nn.Sequential([
                nn.Dense(
                    self.compression_dim or self.hidden_size // 4,
                    dtype=self.dtype,
                    kernel_init=nn.initializers.normal(stddev=0.02),
                    name="compress"
                ),
                nn.LayerNorm(epsilon=1e-5),
                nn.tanh
            ])
            self.decompressor = nn.Dense(
                self.hidden_size,
                dtype=self.dtype,
                kernel_init=nn.initializers.normal(stddev=0.02),
                name="decompress"
            )
            
        # Layer normalization for cached states
        if self.normalize_cached:
            self.cache_norm = nn.LayerNorm(epsilon=1e-5)
            
    def init_cache(self) -> Dict[str, Any]:
        """Initialize empty layer cache.
        
        Returns:
            Empty cache dictionary
        """
        return {
            'states': [],  # List of layer states
            'layer_ids': [],  # List of corresponding layer IDs
            'access_count': []  # List of access counts for LRU
        }
        
    def _compress_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Compress layer state for efficient storage.
        
        Args:
            state: Layer state tensor
            
        Returns:
            Compressed state
        """
        if not self.use_compressed_cache:
            return state
        return self.compressor(state)
        
    def _decompress_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Decompress cached layer state.
        
        Args:
            state: Compressed state tensor
            
        Returns:
            Decompressed state
        """
        if not self.use_compressed_cache:
            return state
        return self.decompressor(state)
        
    def _normalize_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Normalize layer state before caching.
        
        Args:
            state: Layer state tensor
            
        Returns:
            Normalized state
        """
        if not self.normalize_cached:
            return state
        return self.cache_norm(state)
        
    def update_cache(self,
                    cache: Dict[str, Any],
                    layer_state: jnp.ndarray,
                    layer_id: int) -> Dict[str, Any]:
        """Update cache with new layer state.
        
        Args:
            cache: Current cache dictionary
            layer_state: New layer state to cache
            layer_id: ID of the layer
            
        Returns:
            Updated cache dictionary
        """
        # Normalize and compress state
        processed_state = layer_state
        if self.normalize_cached:
            processed_state = self._normalize_state(processed_state)
        if self.use_compressed_cache:
            processed_state = self._compress_state(processed_state)
            
        # Update cache based on eviction policy
        if len(cache['states']) >= self.max_cache_size:
            if self.eviction_policy == "lru":
                # Remove least recently used state
                min_access = min(cache['access_count'])
                idx = cache['access_count'].index(min_access)
                del cache['states'][idx]
                del cache['layer_ids'][idx]
                del cache['access_count'][idx]
            else:  # FIFO
                # Remove oldest state
                cache['states'].pop(0)
                cache['layer_ids'].pop(0)
                cache['access_count'].pop(0)
                
        # Add new state
        cache['states'].append(processed_state)
        cache['layer_ids'].append(layer_id)
        cache['access_count'].append(0)
        
        return cache
        
    def get_layer_states(self,
                        cache: Dict[str, Any],
                        current_layer: int,
                        num_layers: Optional[int] = None) -> List[jnp.ndarray]:
        """Retrieve relevant previous layer states.
        
        Args:
            cache: Cache dictionary
            current_layer: Current layer ID
            num_layers: Number of previous layers to retrieve
            
        Returns:
            List of previous layer states
        """
        if not cache['states']:
            return []
            
        # Get number of layers to retrieve
        num_layers = num_layers or self.max_cache_size
        
        # Find relevant layer states
        relevant_states = []
        for idx, (state, layer_id) in enumerate(zip(cache['states'], cache['layer_ids'])):
            if layer_id < current_layer:
                # Decompress state if needed
                if self.use_compressed_cache:
                    state = self._decompress_state(state)
                relevant_states.append(state)
                # Update access count for LRU
                if self.eviction_policy == "lru":
                    cache['access_count'][idx] += 1
                    
        # Sort by layer ID and take most recent
        relevant_states = relevant_states[-num_layers:]
        
        return relevant_states
        
    def clear_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Clear the layer cache.
        
        Args:
            cache: Cache dictionary to clear
            
        Returns:
            Empty cache dictionary
        """
        cache['states'] = []
        cache['layer_ids'] = []
        cache['access_count'] = []
        return cache
        
    def get_cache_stats(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about cache usage.
        
        Args:
            cache: Cache dictionary
            
        Returns:
            Dictionary of cache statistics
        """
        num_states = len(cache['states'])
        memory_used = sum(state.size * state.dtype.itemsize 
                         for state in cache['states'])
        
        return {
            'num_cached_states': num_states,
            'memory_used_bytes': memory_used,
            'cache_fullness': num_states / self.max_cache_size,
            'layer_ids': cache['layer_ids'].copy(),
            'access_counts': cache['access_count'].copy()
        }
