"""
Expert state management using 3FS for distributed MoE models.
"""

import torch
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

class ExpertStateManager:
    """Manages expert states and parameters using 3FS distributed storage"""
    def __init__(
        self,
        storage_dir: str,
        num_experts: int,
        expert_dim: int,
        capacity_gb: float = 50
    ):
        self.storage_dir = storage_dir
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.capacity = int(capacity_gb * 1024 * 1024 * 1024)  # Convert to bytes
        
        # Expert access statistics
        self.access_counts = np.zeros(num_experts)
        self.last_access = np.zeros(num_experts)
        self._step = 0
        
        # Storage paths
        self.state_dir = os.path.join(storage_dir, 'expert_states')
        self.stats_dir = os.path.join(storage_dir, 'expert_stats')
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

        # Initialize statistics tracking
        self.stats: Dict[str, List[float]] = {
            'load_times': [],
            'store_times': [],
            'hit_rates': []
        }
        
    def _get_expert_path(self, expert_id: int) -> str:
        """Get storage path for expert state"""
        return os.path.join(self.state_dir, f'expert_{expert_id}.pt')
        
    def _get_stats_path(self, expert_id: int) -> str:
        """Get storage path for expert statistics"""
        return os.path.join(self.stats_dir, f'expert_{expert_id}_stats.pt')

    def store_expert_state(
        self,
        expert_id: int,
        state_dict: Dict[str, torch.Tensor],
        stats: Optional[Dict] = None
    ) -> None:
        """Store expert state and statistics in 3FS"""
        # Save state dictionary
        state_path = self._get_expert_path(expert_id)
        torch.save(state_dict, state_path)
        
        # Save expert statistics if provided
        if stats is not None:
            stats_path = self._get_stats_path(expert_id)
            torch.save(stats, stats_path)
            
        # Update access tracking
        self.access_counts[expert_id] += 1
        self.last_access[expert_id] = self._step
        self._step += 1

    def load_expert_state(
        self,
        expert_id: int,
        load_stats: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """Load expert state and optionally statistics from 3FS"""
        state_path = self._get_expert_path(expert_id)
        state_dict = torch.load(state_path)
        
        stats = None
        if load_stats:
            stats_path = self._get_stats_path(expert_id)
            if os.path.exists(stats_path):
                stats = torch.load(stats_path)
                
        # Update access tracking
        self.access_counts[expert_id] += 1
        self.last_access[expert_id] = self._step
        self._step += 1
                
        return state_dict, stats
        
    def get_expert_stats(self, expert_id: int) -> Optional[Dict]:
        """Load expert statistics if available"""
        stats_path = self._get_stats_path(expert_id)
        if os.path.exists(stats_path):
            return torch.load(stats_path)
        return None

    def clear_expert_state(self, expert_id: int) -> None:
        """Remove stored state for an expert"""
        state_path = self._get_expert_path(expert_id)
        stats_path = self._get_stats_path(expert_id)
        
        if os.path.exists(state_path):
            os.remove(state_path)
        if os.path.exists(stats_path):
            os.remove(stats_path)

    def get_access_stats(self) -> Dict[str, np.ndarray]:
        """Get expert access statistics"""
        total_accesses = np.sum(self.access_counts)
        access_fractions = self.access_counts / total_accesses if total_accesses > 0 else np.zeros_like(self.access_counts)
        
        return {
            'access_counts': self.access_counts,
            'access_fractions': access_fractions,
            'last_access': self.last_access
        }
        
    def update_stats(self, load_time: float, store_time: float, hit_rate: float) -> None:
        """Update timing and efficiency statistics"""
        self.stats['load_times'].append(load_time)
        self.stats['store_times'].append(store_time)
        self.stats['hit_rates'].append(hit_rate)
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Calculate performance statistics"""
        stats = {}
        for key, values in self.stats.items():
            if values:
                stats[f'avg_{key}'] = np.mean(values)
                stats[f'std_{key}'] = np.std(values)
                stats[f'min_{key}'] = np.min(values)
                stats[f'max_{key}'] = np.max(values)
        return stats
