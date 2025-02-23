"""Expert state management for Mixture of Experts."""

from typing import Optional, Dict, Any, List, NamedTuple
from dataclasses import dataclass
import threading
from collections import defaultdict

import jax
import jax.numpy as jnp

@dataclass
class ExpertStats:
    """Statistics for a single expert."""
    total_tokens: int = 0
    dropped_tokens: int = 0
    load: float = 0.0
    usage_count: int = 0
    capacity_factor: float = 1.0
    avg_token_load: float = 0.0
    recent_loads: List[float] = None
    
    def __post_init__(self):
        """Initialize stats tracking."""
        if self.recent_loads is None:
            self.recent_loads = []

class ExpertState:
    """State management for a single expert."""
    
    def __init__(self,
                expert_id: int,
                hidden_size: int,
                capacity_factor: float = 1.25,
                max_cache_size: int = 1000):
        """Initialize expert state.
        
        Args:
            expert_id: Expert identifier
            hidden_size: Hidden state dimension
            capacity_factor: Expert capacity multiplier
            max_cache_size: Maximum size of activation cache
        """
        self.expert_id = expert_id
        self.hidden_size = hidden_size
        self.capacity_factor = capacity_factor
        self.max_cache_size = max_cache_size
        
        # Stats tracking
        self.stats = ExpertStats(capacity_factor=capacity_factor)
        self.activation_cache = {}
        self._lock = threading.Lock()
        
    def update_stats(self,
                    num_tokens: int,
                    dropped_tokens: int,
                    load: float) -> None:
        """Update expert statistics.
        
        Args:
            num_tokens: Number of tokens processed
            dropped_tokens: Number of tokens dropped
            load: Expert load factor
        """
        with self._lock:
            self.stats.total_tokens += num_tokens
            self.stats.dropped_tokens += dropped_tokens
            self.stats.load = load
            self.stats.usage_count += 1
            
            # Update moving averages
            self.stats.recent_loads.append(load)
            if len(self.stats.recent_loads) > 100:
                self.stats.recent_loads.pop(0)
            self.stats.avg_token_load = sum(self.stats.recent_loads) / len(self.stats.recent_loads)
            
    def cache_activations(self,
                         key: str,
                         activations: jnp.ndarray) -> None:
        """Cache expert activations.
        
        Args:
            key: Cache key
            activations: Activation tensor
        """
        with self._lock:
            if len(self.activation_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.activation_cache))
                del self.activation_cache[oldest_key]
            self.activation_cache[key] = activations
            
    def get_cached_activations(self, key: str) -> Optional[jnp.ndarray]:
        """Retrieve cached activations.
        
        Args:
            key: Cache key
            
        Returns:
            Cached activations if found, else None
        """
        return self.activation_cache.get(key)
        
    def clear_cache(self) -> None:
        """Clear activation cache."""
        with self._lock:
            self.activation_cache.clear()
            
    def get_load_metrics(self) -> Dict[str, float]:
        """Get expert load metrics.
        
        Returns:
            Dictionary of load metrics
        """
        return {
            'avg_load': self.stats.avg_token_load,
            'current_load': self.stats.load,
            'drop_rate': (self.stats.dropped_tokens / 
                         max(1, self.stats.total_tokens)),
            'usage_rate': self.stats.usage_count
        }

class ExpertStateManager:
    """Manages states for all experts."""
    
    def __init__(self,
                num_experts: int,
                hidden_size: int,
                capacity_factor: float = 1.25,
                max_cache_size: int = 1000):
        """Initialize expert state manager.
        
        Args:
            num_experts: Number of experts
            hidden_size: Hidden state dimension
            capacity_factor: Expert capacity multiplier
            max_cache_size: Maximum cache size per expert
        """
        self.expert_states = [
            ExpertState(i, hidden_size, capacity_factor, max_cache_size)
            for i in range(num_experts)
        ]
        
    def update_expert_stats(self,
                          expert_id: int,
                          num_tokens: int,
                          dropped_tokens: int,
                          load: float) -> None:
        """Update stats for specific expert.
        
        Args:
            expert_id: Expert identifier
            num_tokens: Number of tokens processed
            dropped_tokens: Number of tokens dropped
            load: Expert load factor
        """
        self.expert_states[expert_id].update_stats(
            num_tokens=num_tokens,
            dropped_tokens=dropped_tokens,
            load=load
        )
        
    def get_expert_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get metrics for all experts.
        
        Returns:
            Dictionary of expert metrics
        """
        return {
            f"expert_{i}": state.get_load_metrics()
            for i, state in enumerate(self.expert_states)
        }
        
    def clear_all_caches(self) -> None:
        """Clear activation caches for all experts."""
        for state in self.expert_states:
            state.clear_cache()
            
    def get_total_stats(self) -> Dict[str, float]:
        """Get aggregated statistics across all experts.
        
        Returns:
            Dictionary of aggregated stats
        """
        total_tokens = sum(state.stats.total_tokens for state in self.expert_states)
        total_dropped = sum(state.stats.dropped_tokens for state in self.expert_states)
        avg_load = (sum(state.stats.avg_token_load for state in self.expert_states) / 
                   len(self.expert_states))
                   
        return {
            'total_tokens': total_tokens,
            'total_dropped': total_dropped,
            'avg_expert_load': avg_load,
            'overall_drop_rate': total_dropped / max(1, total_tokens)
        }
