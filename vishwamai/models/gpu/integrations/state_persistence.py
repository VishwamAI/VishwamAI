"""
State persistence and memory management using 3FS for GPU optimized layers.
"""

import torch
import os
from typing import Dict, Optional, Tuple
import torch.nn as nn
import time

class StateManager:
    """Manages state persistence and memory operations using 3FS"""
    def __init__(
        self,
        storage_dir: str,
        model_dim: int,
        cache_size_gb: float = 20
    ):
        self.storage_dir = storage_dir
        self.model_dim = model_dim
        self.cache_size = int(cache_size_gb * 1024 * 1024 * 1024)
        
        # Storage paths
        self.weight_dir = os.path.join(storage_dir, 'weights')
        self.activation_dir = os.path.join(storage_dir, 'activations')
        self.gradient_dir = os.path.join(storage_dir, 'gradients')
        
        os.makedirs(self.weight_dir, exist_ok=True)
        os.makedirs(self.activation_dir, exist_ok=True)
        os.makedirs(self.gradient_dir, exist_ok=True)
        
        # Performance tracking
        self.stats = {
            'weight_loads': 0,
            'weight_stores': 0,
            'activation_hits': 0,
            'activation_misses': 0
        }
        
    def store_layer_weights(
        self,
        layer_id: str,
        weights: Dict[str, torch.Tensor],
        metadata: Optional[Dict] = None
    ) -> None:
        """Store layer weights in 3FS"""
        path = os.path.join(self.weight_dir, f"{layer_id}.pt")
        data = {
            'weights': weights,
            'metadata': metadata,
            'timestamp': time.time()
        }
        torch.save(data, path)
        self.stats['weight_stores'] += 1
        
    def load_layer_weights(
        self,
        layer_id: str
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[Dict]]:
        """Load layer weights from 3FS"""
        path = os.path.join(self.weight_dir, f"{layer_id}.pt")
        if not os.path.exists(path):
            return None, None
            
        try:
            data = torch.load(path)
            self.stats['weight_loads'] += 1
            return data['weights'], data['metadata']
        except:
            return None, None

    def cache_activations(
        self,
        layer_id: str,
        batch_id: int,
        activations: torch.Tensor
    ) -> None:
        """Cache layer activations in 3FS"""
        path = os.path.join(self.activation_dir, f"{layer_id}_{batch_id}.pt")
        torch.save(activations, path)
        
    def get_cached_activations(
        self,
        layer_id: str,
        batch_id: int
    ) -> Optional[torch.Tensor]:
        """Retrieve cached activations if available"""
        path = os.path.join(self.activation_dir, f"{layer_id}_{batch_id}.pt")
        if os.path.exists(path):
            self.stats['activation_hits'] += 1
            return torch.load(path)
        self.stats['activation_misses'] += 1
        return None
        
    def store_gradients(
        self,
        layer_id: str,
        gradients: Dict[str, torch.Tensor]
    ) -> None:
        """Store layer gradients in 3FS"""
        path = os.path.join(self.gradient_dir, f"{layer_id}_grad.pt")
        torch.save(gradients, path)
        
    def load_gradients(
        self,
        layer_id: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Load layer gradients from 3FS"""
        path = os.path.join(self.gradient_dir, f"{layer_id}_grad.pt")
        if os.path.exists(path):
            return torch.load(path)
        return None

    def get_stats(self) -> Dict[str, float]:
        """Get storage statistics"""
        total_hits = self.stats['activation_hits']
        total_accesses = total_hits + self.stats['activation_misses']
        hit_rate = total_hits / total_accesses if total_accesses > 0 else 0
        
        return {
            'weight_loads': self.stats['weight_loads'],
            'weight_stores': self.stats['weight_stores'],
            'activation_hit_rate': hit_rate,
            'total_accesses': total_accesses
        }
        
    def clear_cache(self, cache_type: str = 'all') -> None:
        """Clear specified cache type"""
        if cache_type in ['all', 'activations']:
            for f in os.listdir(self.activation_dir):
                os.remove(os.path.join(self.activation_dir, f))
        if cache_type in ['all', 'gradients']:
            for f in os.listdir(self.gradient_dir):
                os.remove(os.path.join(self.gradient_dir, f))
                
class OptimizedStateManager:
    """
    Enhanced state manager with GPU memory optimizations
    """
    def __init__(
        self,
        base_manager: StateManager,
        device: torch.device,
        max_gpu_cache: int = 2 * 1024 * 1024 * 1024  # 2GB default GPU cache
    ):
        self.base_manager = base_manager
        self.device = device
        self.max_gpu_cache = max_gpu_cache
        
        # GPU cache for frequently accessed data
        self.gpu_cache: Dict[str, torch.Tensor] = {}
        self.cache_scores: Dict[str, float] = {}
        self.access_count: Dict[str, int] = {}
        
    def _update_cache_score(self, key: str) -> None:
        """Update access-based cache scoring"""
        current_time = time.time()
        count = self.access_count.get(key, 0) + 1
        self.access_count[key] = count
        
        # Score based on access frequency and recency
        self.cache_scores[key] = count / (current_time + 1.0)
        
    def _manage_gpu_cache(self) -> None:
        """Manage GPU cache size"""
        cache_size = sum(tensor.element_size() * tensor.nelement() 
                        for tensor in self.gpu_cache.values())
                        
        while cache_size > self.max_gpu_cache and self.gpu_cache:
            # Remove least important cached item
            min_key = min(self.cache_scores, key=self.cache_scores.get)
            removed_tensor = self.gpu_cache.pop(min_key)
            self.cache_scores.pop(min_key)
            cache_size -= removed_tensor.element_size() * removed_tensor.nelement()
            
    def get_tensor(
        self,
        key: str,
        loader_func,
        to_gpu: bool = True
    ) -> Optional[torch.Tensor]:
        """Get tensor with optimized GPU caching"""
        # Check GPU cache first
        if key in self.gpu_cache:
            self._update_cache_score(key)
            return self.gpu_cache[key]
            
        # Load using provided function
        tensor = loader_func()
        if tensor is None:
            return None
            
        # Cache on GPU if requested
        if to_gpu and self.device.type == 'cuda':
            tensor = tensor.to(self.device)
            self.gpu_cache[key] = tensor
            self._update_cache_score(key)
            self._manage_gpu_cache()
            
        return tensor
        
    def clear_gpu_cache(self) -> None:
        """Clear GPU cache"""
        self.gpu_cache.clear()
        self.cache_scores.clear()
        self.access_count.clear()
        torch.cuda.empty_cache()
