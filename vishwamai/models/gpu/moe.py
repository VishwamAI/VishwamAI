"""
GPU-optimized MoE with 3FS integration for distributed caching and state management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

# Import GPU optimizations and integrations
from vishwamai.models.gpu.optimizations.deep_ep import Buffer
from vishwamai.models.gpu.optimizations.eplb import EPLB
from vishwamai.models.gpu.integrations.kvcache_manager import KVCacheManager
from vishwamai.models.gpu.integrations.expert_state_manager import ExpertStateManager

class OptimizedMoE(nn.Module):
    """MoE with optimized load balancing, expert parallelism and 3FS integration"""
    def __init__(
        self,
        embed_dim,
        num_experts,
        dropout=0.1,
        use_3fs=True,
        cache_dir="/tmp/vishwamai/moe_cache"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.dropout = dropout
        self.use_3fs = use_3fs
        
        # Initialize experts
        self.experts = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_experts)
        ])
        
        # Router with gating
        self.router = nn.Linear(embed_dim, num_experts)
        
        # Load balancer
        self.load_balancer = EPLB(num_experts)
        
        # Expert parallelism buffer
        self._buffer = None
        
        # 3FS integration components
        if use_3fs:
            os.makedirs(cache_dir, exist_ok=True)
            kvcache_dir = os.path.join(cache_dir, "kvcache")
            expert_state_dir = os.path.join(cache_dir, "expert_states")
            
            self.kvcache = KVCacheManager(
                cache_dir=kvcache_dir,
                embed_dim=embed_dim,
                num_heads=1  # MoE uses single-head attention
            )
            
            self.expert_manager = ExpertStateManager(
                storage_dir=expert_state_dir,
                num_experts=num_experts,
                expert_dim=embed_dim
            )
        else:
            self.kvcache = None
            self.expert_manager = None
        
    def get_buffer(self):
        if self._buffer is None:
            self._buffer = Buffer.get_buffer(hidden_bytes=self.embed_dim * 2)
        return self._buffer
    
    def _try_load_expert_state(self, expert_id):
        """Try to load expert state from 3FS storage"""
        if not self.use_3fs or self.expert_manager is None:
            return None, None
            
        start_time = time.time()
        try:
            state_dict, stats = self.expert_manager.load_expert_state(expert_id, load_stats=True)
            load_time = time.time() - start_time
            self.expert_manager.update_stats(load_time=load_time, store_time=0, hit_rate=1.0)
            return state_dict, stats
        except:
            return None, None
            
    def _store_expert_state(self, expert_id, expert, stats=None):
        """Store expert state to 3FS storage"""
        if not self.use_3fs or self.expert_manager is None:
            return
            
        start_time = time.time()
        try:
            self.expert_manager.store_expert_state(
                expert_id,
                expert.state_dict(),
                stats=stats
            )
            store_time = time.time() - start_time
            self.expert_manager.update_stats(load_time=0, store_time=store_time, hit_rate=1.0)
        except:
            pass
    
    def _get_cached_value(self, key, batch_idx=0, seq_idx=0):
        """Try to get cached value from KVCache"""
        if not self.use_3fs or self.kvcache is None:
            return None
        return self.kvcache.retrieve(batch_idx, seq_idx)
        
    def forward(self, x, batch_idx=0, seq_idx=0):
        batch_size, seq_len, _ = x.shape
        
        # Try to get cached value first
        cached_value = self._get_cached_value(x, batch_idx, seq_idx)
        if cached_value is not None:
            return cached_value[0]  # Return cached output
            
        # Get router logits and load-balanced assignments
        router_logits = self.router(x)
        indices, weights = self.load_balancer.get_expert_assignment(router_logits)
        
        # Get expert parallelism buffer
        buffer = self.get_buffer()
        
        # Dispatch to experts
        expert_inputs, _, weights, counts, handle, _ = buffer.dispatch(
            x, indices.unsqueeze(-1), weights.unsqueeze(-1)
        )
        
        # Process with experts
        expert_outputs = []
        expert_stats = {}
        
        for i, expert in enumerate(self.experts):
            if counts[i] > 0:
                # Try to load expert state from 3FS
                state_dict, stats = self._try_load_expert_state(i)
                if state_dict is not None:
                    expert.load_state_dict(state_dict)
                
                # Process inputs through expert
                start_time = time.time()
                expert_output = expert(expert_inputs[i, :counts[i]])
                expert_outputs.append(expert_output)
                
                # Update expert statistics
                process_time = time.time() - start_time
                expert_stats[i] = {
                    'process_time': process_time,
                    'input_size': counts[i],
                    'compute_efficiency': process_time / counts[i]
                }
                
                # Store updated expert state
                self._store_expert_state(i, expert, stats=expert_stats[i])
                
        expert_outputs = torch.cat(expert_outputs, dim=0)
        
        # Combine expert outputs
        output, _ = buffer.combine(expert_outputs, handle, weights)
        
        # Calculate load balancing loss
        routing_probs = F.softmax(router_logits, dim=-1)
        self.aux_loss = self.load_balancer.get_load_balancing_loss(routing_probs)
        
        # Cache the output if using 3FS
        if self.use_3fs and self.kvcache is not None:
            self.kvcache.store(output, output, batch_idx, seq_idx)
        
        # Reset load balancer counts
        self.load_balancer.reset_counts()
        
        return output
