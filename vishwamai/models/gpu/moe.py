"""
GPU-optimized MoE with 3FS integration for distributed caching and state management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os

# Import GPU optimizations and integrations
from vishwamai.models.gpu.optimizations.deep_ep import Buffer, get_num_sms
from vishwamai.models.gpu.optimizations.eplb import EPLB
from vishwamai.models.gpu.kernel_layers import DeepGEMMLinear
from vishwamai.models.gpu.integrations.kvcache_manager import KVCacheManager
from vishwamai.models.gpu.integrations.expert_state_manager import ExpertStateManager

class OptimizedMoE(nn.Module):
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
        
        # Initialize experts with optimized linear layers
        self.experts = nn.ModuleList([
            DeepGEMMLinear(embed_dim, embed_dim, use_amp=True) 
            for _ in range(num_experts)
        ])
        
        # Router with gating
        self.router = DeepGEMMLinear(embed_dim, num_experts, use_amp=True)
        
        # Expert parallelism buffer 
        self._buffer = None
        Buffer.set_num_sms(get_num_sms())
        
        # Load balancer
        self.load_balancer = EPLB(num_experts)
        
        # 3FS components
        if use_3fs:
            os.makedirs(cache_dir, exist_ok=True)
            
            self.kvcache = KVCacheManager(
                cache_dir=os.path.join(cache_dir, "kvcache"),
                embed_dim=embed_dim,
                num_heads=1  # MoE uses single-head attention
            )
            
            self.expert_manager = ExpertStateManager(
                storage_dir=os.path.join(cache_dir, "expert_states"),
                num_experts=num_experts,
                expert_dim=embed_dim
            )
        else:
            self.kvcache = None
            self.expert_manager = None

    def get_buffer(self):
        if self._buffer is None:
            self._buffer = Buffer.get_buffer(
                hidden_bytes=self.embed_dim * 2,
                allocate_sm_parts=True
            )
        return self._buffer
    
    def _try_load_expert_state(self, expert_id):
        if not self.use_3fs or self.expert_manager is None:
            return None, None
            
        try:
            state_dict, stats = self.expert_manager.load_expert_state(expert_id)
            return state_dict, stats
        except:
            return None, None
            
    def _store_expert_state(self, expert_id, expert, stats=None):
        if not self.use_3fs or self.expert_manager is None:
            return
            
        try:
            self.expert_manager.store_expert_state(
                expert_id,
                expert.state_dict(),
                stats=stats
            )
        except:
            pass
    
    def forward(self, x, batch_idx=0, seq_idx=0):
        batch_size, seq_len, _ = x.shape
        
        # Try cached value first
        if self.use_3fs and self.kvcache is not None:
            cached = self.kvcache.retrieve(batch_idx, seq_idx)
            if cached is not None:
                return cached[0]

        # Get router logits and load-balanced assignments
        router_logits = self.router(x)
        indices, weights = self.load_balancer.get_expert_assignment(
            router_logits,
            top_k=2  # Use top-2 gating
        )
        
        # Get parallelism buffer
        buffer = self.get_buffer()
        
        # Optimized parallel dispatch
        expert_inputs, indices, weights, counts, handle, event = buffer.dispatch(
            x, indices, weights,
            async_finish=True
        )

        # Process with experts
        expert_outputs = []
        expert_stats = {}
        
        for i, expert in enumerate(self.experts):
            if counts[i] > 0:
                # Try loading expert state
                state_dict, stats = self._try_load_expert_state(i)
                if state_dict:
                    expert.load_state_dict(state_dict)
                
                # Process inputs 
                start = time.time()
                expert_output = expert(expert_inputs[i, :counts[i]])
                expert_outputs.append(expert_output)
                
                # Update stats
                process_time = time.time() - start
                expert_stats[i] = {
                    'process_time': process_time,
                    'input_size': counts[i],
                    'compute_efficiency': process_time / counts[i] 
                }
                
                # Store updated state
                self._store_expert_state(i, expert, stats=expert_stats[i])
                
        expert_outputs = torch.cat(expert_outputs, dim=0)
        
        # Optimized combine with buffer
        output, _ = buffer.combine(expert_outputs, handle, weights)
        
        # Routing loss
        routing_probs = F.softmax(router_logits, dim=-1)
        self.aux_loss = self.load_balancer.get_load_balancing_loss(routing_probs)
        
        # Cache if using 3FS
        if self.use_3fs and self.kvcache is not None:
            self.kvcache.store(
                output, output,
                batch_idx=batch_idx,
                seq_idx=seq_idx
            )
        
        # Reset load balancer
        self.load_balancer.reset_counts()
        
        return output
