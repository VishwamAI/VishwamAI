"""
KVCache manager with smallpond integration for distributed caching.
"""

import torch
import os
import smallpond
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import time
import json

@dataclass
class CacheEntry:
    """KVCache entry with metadata"""
    key_states: torch.Tensor
    value_states: torch.Tensor
    last_used: float
    hits: int = 0

class KVCacheManager:
    """Manages distributed KV caching using smallpond"""
    
    def __init__(self,
                cache_dir: str,
                embed_dim: int,
                num_heads: int,
                max_cache_size: int = 1000000,
                eviction_policy: str = 'lru',
                use_smallpond: bool = True):
        """
        Initialize KVCache manager.
        
        Args:
            cache_dir: Cache directory path
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            max_cache_size: Maximum number of cached entries
            eviction_policy: Cache eviction policy ('lru' or 'lfu')
            use_smallpond: Whether to use smallpond for distributed caching
        """
        self.cache_dir = cache_dir
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_cache_size = max_cache_size
        self.eviction_policy = eviction_policy
        self.use_smallpond = use_smallpond
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Local cache
        self.cache: Dict[Tuple[int, int], CacheEntry] = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size': 0
        }
        
        # Initialize smallpond
        if use_smallpond:
            try:
                self.sp_session = smallpond.init(
                    num_executors=torch.cuda.device_count(),
                    data_root=cache_dir,
                    bind_numa_node=True
                )
            except:
                self.sp_session = None
                self.use_smallpond = False
                
    def store(self,
             key_states: torch.Tensor,
             value_states: torch.Tensor,
             batch_idx: int,
             seq_idx: int):
        """Store key-value states in cache"""
        cache_key = (batch_idx, seq_idx)
        
        # Check cache size and evict if needed
        if len(self.cache) >= self.max_cache_size:
            self._evict_entries()
            
        # Create cache entry
        entry = CacheEntry(
            key_states=key_states.detach(),
            value_states=value_states.detach(),
            last_used=time.time()
        )
        
        self.cache[cache_key] = entry
        self.stats['total_size'] += 1
        
        # Store in distributed cache if available
        if self.use_smallpond and self.sp_session:
            try:
                # Convert to numpy for smallpond
                cache_data = {
                    'key_states': key_states.cpu().numpy(),
                    'value_states': value_states.cpu().numpy(),
                    'last_used': entry.last_used,
                    'hits': entry.hits
                }
                
                # Create DataFrame with single row
                df = self.sp_session.create_dataframe({
                    'batch_idx': [batch_idx],
                    'seq_idx': [seq_idx],
                    'cache_data': [cache_data]
                })
                
                # Write to parquet
                cache_path = os.path.join(
                    self.cache_dir,
                    f"cache_{batch_idx}_{seq_idx}.parquet"
                )
                df.write_parquet(cache_path)
                
            except Exception as e:
                print(f"Failed to store in distributed cache: {e}")
                
    def retrieve(self,
                batch_idx: int,
                seq_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve key-value states from cache"""
        cache_key = (batch_idx, seq_idx)
        
        # Check local cache first
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            entry.last_used = time.time()
            entry.hits += 1
            self.stats['hits'] += 1
            return entry.key_states, entry.value_states
            
        # Try distributed cache
        if self.use_smallpond and self.sp_session:
            try:
                cache_path = os.path.join(
                    self.cache_dir,
                    f"cache_{batch_idx}_{seq_idx}.parquet"
                )
                
                if os.path.exists(cache_path):
                    df = self.sp_session.read_parquet(cache_path)
                    cache_data = df.to_pandas().iloc[0]['cache_data']
                    
                    # Convert back to tensors
                    key_states = torch.from_numpy(cache_data['key_states'])
                    value_states = torch.from_numpy(cache_data['value_states'])
                    
                    # Update local cache
                    entry = CacheEntry(
                        key_states=key_states,
                        value_states=value_states,
                        last_used=time.time(),
                        hits=cache_data['hits'] + 1
                    )
                    self.cache[cache_key] = entry
                    self.stats['hits'] += 1
                    
                    return key_states, value_states
                    
            except Exception as e:
                print(f"Failed to retrieve from distributed cache: {e}")
                
        self.stats['misses'] += 1
        return None
        
    def _evict_entries(self, count: int = 1):
        """Evict entries based on policy"""
        if not self.cache:
            return
            
        for _ in range(count):
            if self.eviction_policy == 'lru':
                # Find least recently used
                lru_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].last_used
                )
                victim_key = lru_key
            else:  # 'lfu'
                # Find least frequently used
                lfu_key = min(
                    self.cache.keys(),
                    key=lambda k: self.cache[k].hits
                )
                victim_key = lfu_key
                
            # Remove from local cache
            del self.cache[victim_key]
            
            # Remove from distributed cache
            if self.use_smallpond and self.sp_session:
                try:
                    cache_path = os.path.join(
                        self.cache_dir,
                        f"cache_{victim_key[0]}_{victim_key[1]}.parquet"
                    )
                    if os.path.exists(cache_path):
                        os.remove(cache_path)
                except:
                    pass
                    
            self.stats['evictions'] += 1
            self.stats['total_size'] -= 1
            
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        
        if self.use_smallpond and self.sp_session:
            try:
                # Remove all cache files
                for file in os.listdir(self.cache_dir):
                    if file.startswith('cache_') and file.endswith('.parquet'):
                        os.remove(os.path.join(self.cache_dir, file))
            except:
                pass
                
        self.stats['total_size'] = 0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = (
            self.stats['hits'] /
            (self.stats['hits'] + self.stats['misses'])
            if (self.stats['hits'] + self.stats['misses']) > 0
            else 0
        )
        
        return {
            **self.stats,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
        
    def save_stats(self):
        """Save statistics to file"""
        stats_path = os.path.join(self.cache_dir, 'cache_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.get_stats(), f)
            
    def cleanup(self):
        """Cleanup resources"""
        self.clear()
        self.save_stats()
        
        if self.use_smallpond and self.sp_session:
            self.sp_session.shutdown()
