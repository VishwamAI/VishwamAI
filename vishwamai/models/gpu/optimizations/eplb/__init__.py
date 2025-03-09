"""
Expert Parallelism Load Balancing (EPLB) with distributed processing via smallpond.
"""

import torch
import torch.nn.functional as F
import math
import os
import time
import smallpond
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

@dataclass
class LoadStats:
    """Load balancing statistics"""
    total_tokens: int = 0
    num_experts: int = 0
    expert_counts: Optional[List[int]] = None
    load_std: float = 0.0
    max_load: float = 0.0
    min_load: float = 0.0

class EPLB:
    """Expert Parallelism Load Balancer with smallpond integration"""
    
    def __init__(self,
                num_experts: int,
                use_smallpond: bool = True,
                cache_dir: Optional[str] = "/tmp/vishwamai/eplb_cache"):
        self.num_experts = num_experts
        self.use_smallpond = use_smallpond
        self.cache_dir = cache_dir
        
        # Expert stats
        self.expert_counts = torch.zeros(num_experts, dtype=torch.long)
        self.expert_loads = torch.zeros(num_experts)
        self.expert_capacities = torch.ones(num_experts)
        
        # Load history for adaptive balancing
        self.load_history = []
        self.max_history_size = 1000
        
        # Initialize smallpond
        if use_smallpond:
            os.makedirs(cache_dir, exist_ok=True)
            try:
                self.sp_session = smallpond.init(
                    num_executors=torch.cuda.device_count(),
                    data_root=cache_dir,
                    bind_numa_node=True
                )
            except:
                self.sp_session = None
                self.use_smallpond = False
                
    def get_expert_assignment(self, router_logits: torch.Tensor,
                            top_k: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get expert assignments with load balancing"""
        def balance_fn(logits):
            # Get router probabilities
            router_probs = F.softmax(logits, dim=-1)
            
            # Get top-k experts and their probabilities
            top_probs, top_indices = router_probs.topk(k=top_k, dim=-1)
            
            # Apply load balancing
            capacity = self.expert_capacities.to(logits.device)
            load = self.expert_loads.to(logits.device)
            
            # Scale probabilities by available capacity
            avail_capacity = F.relu(capacity - load).unsqueeze(0)
            balanced_probs = top_probs * avail_capacity
            
            # Re-normalize probabilities
            balanced_probs = balanced_probs / (balanced_probs.sum(dim=-1, keepdim=True) + 1e-6)
            
            return top_indices, balanced_probs
            
        # Use distributed computation if available
        if self.use_smallpond and self.sp_session:
            # Convert to numpy
            logits_np = router_logits.detach().cpu().numpy()
            
            # Create dataframe
            df = self.sp_session.create_dataframe({'logits': [logits_np]})
            df = df.repartition(self.sp_session.num_executors)
            
            # Process in parallel
            def process_partition(partition):
                import torch
                import numpy as np
                logits = torch.from_numpy(np.array(partition['logits'].iloc[0]))
                indices, probs = balance_fn(logits)
                return {
                    'indices': indices.cpu().numpy(),
                    'probs': probs.cpu().numpy()
                }
                
            result_df = df.map_partitions(process_partition)
            result = result_df.to_pandas().iloc[0]
            
            indices = torch.from_numpy(result['indices']).to(router_logits.device)
            probs = torch.from_numpy(result['probs']).to(router_logits.device)
        else:
            indices, probs = balance_fn(router_logits)
            
        # Update statistics
        self._update_stats(indices, probs)
        return indices, probs
        
    def get_load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Calculate load balancing auxiliary loss"""
        # Get expert loads
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(-1, self.expert_counts.unsqueeze(0), 1.0)
        
        # Calculate load variance
        mean_load = router_probs.mean(0)
        load_variance = ((router_probs - mean_load) ** 2).mean()
        
        # Add capacity regularization
        capacity_penalty = torch.relu(
            (router_probs.sum(0) / self.expert_capacities) - 1.0
        ).mean()
        
        return load_variance + 0.01 * capacity_penalty
        
    def _update_stats(self, indices: torch.Tensor, probs: torch.Tensor):
        """Update expert statistics"""
        # Update counts
        for idx in range(self.num_experts):
            mask = (indices == idx)
            count = mask.sum().item()
            self.expert_counts[idx] += count
            
        # Update loads
        new_loads = torch.zeros_like(self.expert_loads)
        for i in range(indices.size(-1)):
            new_loads.scatter_add_(
                0, indices[..., i].flatten(),
                probs[..., i].flatten()
            )
        self.expert_loads = 0.9 * self.expert_loads + 0.1 * new_loads
        
        # Update load history
        stats = LoadStats(
            total_tokens=indices.size(0) * indices.size(1),
            num_experts=self.num_experts,
            expert_counts=self.expert_counts.tolist(),
            load_std=self.expert_loads.std().item(),
            max_load=self.expert_loads.max().item(),
            min_load=self.expert_loads.min().item()
        )
        self.load_history.append(stats)
        if len(self.load_history) > self.max_history_size:
            self.load_history.pop(0)
            
        # Adapt capacities based on history
        if len(self.load_history) >= 100:
            recent_loads = torch.stack([
                torch.tensor(s.expert_counts) 
                for s in self.load_history[-100:]
            ]).float()
            mean_loads = recent_loads.mean(0)
            load_stds = recent_loads.std(0)
            
            # Increase capacity for overloaded experts
            overload_mask = mean_loads > (mean_loads.mean() + load_stds)
            self.expert_capacities[overload_mask] *= 1.1
            
            # Decrease capacity for underutilized experts
            underload_mask = mean_loads < (mean_loads.mean() - load_stds)
            self.expert_capacities[underload_mask] *= 0.9
            
            # Normalize capacities
            self.expert_capacities /= self.expert_capacities.mean()
            
    def reset_counts(self):
        """Reset expert counts"""
        self.expert_counts.zero_()
        
    def get_stats(self) -> Dict:
        """Get current load balancing statistics"""
        if not self.load_history:
            return {}
            
        latest = self.load_history[-1]
        return {
            'total_tokens': latest.total_tokens,
            'num_experts': latest.num_experts,
            'expert_counts': latest.expert_counts,
            'load_std': latest.load_std,
            'max_load': latest.max_load,
            'min_load': latest.min_load,
            'capacities': self.expert_capacities.tolist()
        }
        
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'sp_session') and self.sp_session:
            self.sp_session.shutdown()

def init_load_balancer():
    """Initialize EPLB state"""
    torch.cuda.empty_cache()
    
def get_load_stats(balancer: EPLB) -> Dict:
    """Get load balancing statistics"""
    return balancer.get_stats()

# Initialize components
init_load_balancer()