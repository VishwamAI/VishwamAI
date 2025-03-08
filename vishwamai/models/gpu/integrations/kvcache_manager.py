"""
KVCache manager for 3FS integration with attention mechanisms.
"""

import torch
import os
from typing import Optional, Tuple, Dict
import numpy as np

class KVCacheManager:
    """Manages KV caching using 3FS for optimized inference"""
    def __init__(
        self,
        cache_dir: str,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 2048,
        cache_size_gb: float = 100
    ):
        self.cache_dir = cache_dir
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.cache_size = int(cache_size_gb * 1024 * 1024 * 1024)  # Convert to bytes
        
        # Initialize cache stats
        self.cache_hits = 0
        self.cache_misses = 0
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Calculate max entries based on tensor sizes
        bytes_per_entry = (
            2 *  # K and V tensors
            4 *  # Float32 bytes
            max_seq_len *
            num_heads * 
            (embed_dim // num_heads)
        )
        self.max_entries = self.cache_size // bytes_per_entry
        
        # Initialize cache mappings
        self.cache_index: Dict[str, str] = {}
        self.lru_order: list = []
    
    def _get_cache_key(self, batch_idx: int, seq_idx: int) -> str:
        """Generate unique cache key"""
        return f"{batch_idx}_{seq_idx}"
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get filesystem path for cache entry"""
        return os.path.join(self.cache_dir, f"{cache_key}.cache")
    
    def store(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        batch_idx: int,
        seq_idx: int
    ) -> None:
        """Store KV tensors in 3FS cache"""
        cache_key = self._get_cache_key(batch_idx, seq_idx)
        cache_path = self._get_cache_path(cache_key)
        
        # Combine K,V tensors for single write 
        combined = torch.cat([key_states, value_states], dim=0)
        
        # Save to 3FS
        torch.save(combined, cache_path)
        
        # Update mappings
        self.cache_index[cache_key] = cache_path
        self.lru_order.append(cache_key)
        
        # Evict if needed
        while len(self.cache_index) > self.max_entries:
            self._evict_lru()
    
    def retrieve(
        self,
        batch_idx: int,
        seq_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve KV tensors from cache if they exist"""
        cache_key = self._get_cache_key(batch_idx, seq_idx)
        
        if cache_key not in self.cache_index:
            self.cache_misses += 1
            return None
            
        cache_path = self.cache_index[cache_key]
        
        # Update LRU
        self.lru_order.remove(cache_key)
        self.lru_order.append(cache_key)
        
        # Load from 3FS
        combined = torch.load(cache_path)
        split_idx = combined.size(0) // 2
        
        self.cache_hits += 1
        return combined[:split_idx], combined[split_idx:]
    
    def _evict_lru(self) -> None:
        """Remove least recently used cache entry"""
        if not self.lru_order:
            return
            
        lru_key = self.lru_order.pop(0)
        cache_path = self.cache_index.pop(lru_key)
        
        if os.path.exists(cache_path):
            os.remove(cache_path)
    
    def clear(self) -> None:
        """Clear all cached data"""
        for cache_path in self.cache_index.values():
            if os.path.exists(cache_path):
                os.remove(cache_path)
        self.cache_index.clear()
        self.lru_order.clear()
        
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
        
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        return {
            "hit_rate": self.hit_rate,
            "entries": len(self.cache_index),
            "utilization": len(self.cache_index) / self.max_entries
        }
