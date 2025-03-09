"""
Expert state management with distributed storage via smallpond.
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
class ExpertStats:
    """Expert statistics"""
    compute_time: float = 0.0
    num_calls: int = 0
    num_tokens: int = 0
    hit_rate: float = 0.0

class ExpertStateManager:
    """Manages expert states with distributed storage"""
    
    def __init__(self,
                storage_dir: str,
                num_experts: int,
                expert_dim: int,
                max_cache_size: int = 1000,
                use_smallpond: bool = True):
        self.storage_dir = storage_dir
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.max_cache_size = max_cache_size
        self.use_smallpond = use_smallpond
        
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'experts'), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'stats'), exist_ok=True)
        
        # Initialize statistics
        self.stats = {i: ExpertStats() for i in range(num_experts)}
        
        # Initialize smallpond
        if use_smallpond:
            try:
                self.sp_session = smallpond.init(
                    num_executors=torch.cuda.device_count(),
                    data_root=storage_dir,
                    bind_numa_node=True
                )
            except:
                self.sp_session = None
                self.use_smallpond = False
                
    def store_expert_state(self,
                         expert_id: int,
                         state_dict: Dict[str, torch.Tensor],
                         stats: Optional[Dict[str, float]] = None):
        """Store expert state and stats"""
        # Update statistics
        if stats:
            self.stats[expert_id].compute_time += stats.get('compute_time', 0)
            self.stats[expert_id].num_calls += 1
            self.stats[expert_id].num_tokens += stats.get('num_tokens', 0)
            
        if not self.use_smallpond or self.sp_session is None:
            # Save locally
            state_path = os.path.join(
                self.storage_dir,
                'experts',
                f'expert_{expert_id}.pt'
            )
            torch.save(state_dict, state_path)
            return
            
        # Convert state dict to numpy
        numpy_state = {
            k: v.cpu().numpy()
            for k, v in state_dict.items()
        }
        
        # Create DataFrame
        df = self.sp_session.create_dataframe({
            'expert_id': [expert_id],
            'state': [numpy_state],
            'stats': [self.stats[expert_id].__dict__]
        })
        
        # Save to parquet
        state_path = os.path.join(
            self.storage_dir,
            'experts',
            f'expert_{expert_id}.parquet'
        )
        df.write_parquet(state_path)
        
    def load_expert_state(self,
                        expert_id: int,
                        load_stats: bool = False) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Load expert state and optionally stats"""
        if not self.use_smallpond or self.sp_session is None:
            # Load locally
            state_path = os.path.join(
                self.storage_dir,
                'experts',
                f'expert_{expert_id}.pt'
            )
            if os.path.exists(state_path):
                state_dict = torch.load(state_path)
                stats_dict = self.stats[expert_id].__dict__ if load_stats else None
                return state_dict, stats_dict
            return None, None
            
        # Load from distributed storage
        state_path = os.path.join(
            self.storage_dir,
            'experts',
            f'expert_{expert_id}.parquet'
        )
        
        if not os.path.exists(state_path):
            return None, None
            
        df = self.sp_session.read_parquet(state_path)
        row = df.to_pandas().iloc[0]
        
        # Convert back to torch tensors
        state_dict = {
            k: torch.from_numpy(v)
            for k, v in row['state'].items()
        }
        
        stats_dict = row['stats'] if load_stats else None
        return state_dict, stats_dict
        
    def update_stats(self,
                    expert_id: Optional[int] = None,
                    load_time: float = 0.0,
                    store_time: float = 0.0,
                    hit_rate: float = 0.0):
        """Update storage statistics"""
        if expert_id is not None:
            stats = self.stats[expert_id]
            stats.hit_rate = (
                stats.hit_rate * 0.9 + hit_rate * 0.1
                if stats.num_calls > 0
                else hit_rate
            )
        
    def get_expert_stats(self,
                       expert_id: Optional[int] = None) -> Dict[str, Any]:
        """Get expert statistics"""
        if expert_id is not None:
            return self.stats[expert_id].__dict__
            
        return {
            i: stats.__dict__
            for i, stats in self.stats.items()
        }
        
    def save_stats(self):
        """Save statistics to file"""
        stats_path = os.path.join(
            self.storage_dir,
            'stats',
            'expert_stats.json'
        )
        with open(stats_path, 'w') as f:
            json.dump(self.get_expert_stats(), f)
            
    def cleanup(self):
        """Cleanup resources"""
        self.save_stats()
        
        if self.use_smallpond and self.sp_session:
            self.sp_session.shutdown()
