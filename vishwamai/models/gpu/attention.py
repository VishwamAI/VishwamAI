"""
GPU-optimized attention mechanisms for VishwamAI with distributed processing via smallpond.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from abc import ABC, abstractmethod
from torch.cuda.amp import autocast
import smallpond

# Import local GPU optimizations
from vishwamai.models.gpu.optimizations.flash_mla import (
    Flash_fwd_kernel_traits_mla,
    Flash_fwd_mla_params,
    flash_mla_with_kvcache,
    get_mla_metadata,
    run_mha_fwd_splitkv_mla
)
from vishwamai.models.gpu.optimizations.deep_ep import Buffer
from vishwamai.models.gpu.optimizations.eplb import EPLB

class BaseAttention(nn.Module, ABC):
    """Enhanced base class with GPU-aware initialization and mixed precision support"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True, use_smallpond=True):
        super(BaseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.use_amp = use_amp
        self.use_smallpond = use_smallpond
        self._reset_parameters()
        
        # Initialize smallpond session
        if use_smallpond:
            try:
                self.sp_session = smallpond.init(
                    num_executors=torch.cuda.device_count(),
                    bind_numa_node=True
                )
            except:
                self.sp_session = None
                self.use_smallpond = False

    def _reset_parameters(self):
        # GPU-optimized Xavier initialization
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.o_proj.bias, 0.0)

    @abstractmethod
    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0):
        pass

    def _reshape_for_multihead(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
        
    def _distribute_computation(self, x, compute_fn):
        """Distribute computation using smallpond if available"""
        if not self.use_smallpond or self.sp_session is None:
            return compute_fn(x)
            
        # Convert to numpy for smallpond processing
        x_np = x.detach().cpu().numpy()
        
        # Create smallpond DataFrame and partition
        df = self.sp_session.create_dataframe({'data': [x_np]})
        df = df.repartition(self.sp_session.num_executors)
        
        # Process in parallel
        def process_partition(partition):
            import torch
            import numpy as np
            data = torch.from_numpy(np.array(partition['data'].iloc[0]))
            result = compute_fn(data)
            return result.cpu().numpy()
            
        result_df = df.map_partitions(process_partition)
        
        # Gather results
        result = torch.from_numpy(result_df.to_pandas()['data'].iloc[0])
        return result.to(x.device)

class FlashMLAAttention(BaseAttention):
    """Optimized MLA attention with mixed precision support and FlashMLA kernels"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True, use_smallpond=True):
        super().__init__(embed_dim, num_heads, dropout, use_amp, use_smallpond)
        self.flash_config = Flash_fwd_kernel_traits_mla()
        
    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0):
        def attention_forward(inputs):
            # Project q,k,v with optimized kernels
            q = self._reshape_for_multihead(self.q_proj(inputs))
            k = self._reshape_for_multihead(self.k_proj(inputs if context is None else context))
            v = self._reshape_for_multihead(self.v_proj(inputs if context is None else context))

            # Get sequence metadata for FlashMLA
            seqlens_k = torch.tensor([k.size(2)], dtype=torch.int32, device=inputs.device)

            # Configure FlashMLA parameters
            mla_metadata, num_splits = get_mla_metadata(
                seqlens_k,
                self.num_heads,
                self.num_heads,
                8  # Num SM parts
            )

            # Run optimized FlashMLA kernel
            output = flash_mla_with_kvcache(
                q, k, v,
                seqlens_k,
                head_size=self.head_dim,
                tile_scheduler_metadata=mla_metadata,
                num_splits=num_splits,
                causal=False,
                sm_scale=0.5
            )

            # Reshape and project output
            batch_size, seq_len, _ = inputs.size()
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            return self.o_proj(output)
            
        # Distribute computation if possible
        return self._distribute_computation(x, attention_forward)

class OptimizedMoEAttention(BaseAttention):
    """GPU-optimized MoE attention with DeepEP and smallpond integration"""
    def __init__(self, embed_dim, num_heads, num_experts=4, dropout=0.1, use_amp=True, use_smallpond=True):
        super().__init__(embed_dim, num_heads, dropout, use_amp, use_smallpond)
        self.num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            FlashMLAAttention(embed_dim, num_heads, dropout, use_amp, use_smallpond)
            for _ in range(num_experts)
        ])
        
        # Expert routing
        self.router = nn.Linear(embed_dim, num_experts)
        
        # DeepEP buffer for efficient dispatch/combine
        self._buffer = None

    def _get_buffer(self):
        if self._buffer is None:
            self._buffer = Buffer(
                group=None,
                hidden_bytes=self.embed_dim * 2,
                num_nvl_bytes=0,
                num_rdma_bytes=0
            )
        return self._buffer

    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0):
        def moe_forward(inputs):
            batch_size, seq_len, _ = inputs.size()

            # Get expert assignments
            router_logits = self.router(inputs)
            router_probs = F.softmax(router_logits, dim=-1)
            _, indices = router_probs.topk(k=2, dim=-1)
            
            # DeepEP dispatch
            buffer = self._get_buffer()
            dispatched_x, idx, weights, expert_counts, handle, event = buffer.dispatch(
                inputs, indices, router_probs
            )
            
            # Process tokens with experts in parallel
            expert_outputs = []
            start_idx = 0
            
            if self.use_smallpond and self.sp_session:
                # Create dataframe with expert inputs
                expert_dfs = []
                for i, expert in enumerate(self.experts):
                    if expert_counts[i] > 0:
                        end_idx = start_idx + expert_counts[i]
                        expert_input = dispatched_x[start_idx:end_idx]
                        df = self.sp_session.create_dataframe({
                            'expert_id': i,
                            'input': expert_input.cpu().numpy(),
                            'count': expert_counts[i]
                        })
                        expert_dfs.append(df)
                        start_idx = end_idx
                        
                # Process experts in parallel
                if expert_dfs:
                    combined_df = self.sp_session.concat(expert_dfs)
                    def process_expert(partition):
                        import torch
                        import numpy as np
                        results = []
                        for _, row in partition.iterrows():
                            expert_id = row['expert_id']
                            expert_input = torch.from_numpy(row['input'])
                            expert_output = self.experts[expert_id](
                                expert_input, context, mask
                            )
                            results.append(expert_output.cpu().numpy())
                        return np.concatenate(results)
                        
                    result_df = combined_df.map_partitions(process_expert)
                    expert_outputs = torch.from_numpy(
                        result_df.to_pandas().values[0]
                    ).to(x.device)
            else:
                # Process sequentially if smallpond unavailable
                for i, expert in enumerate(self.experts):
                    if expert_counts[i] > 0:
                        end_idx = start_idx + expert_counts[i]
                        expert_output = expert(
                            dispatched_x[start_idx:end_idx],
                            context,
                            mask
                        )
                        expert_outputs.append(expert_output)
                        start_idx = end_idx
                expert_outputs = torch.cat(expert_outputs, dim=0)
            
            # DeepEP combine
            output, _ = buffer.combine(
                expert_outputs, 
                handle,
                weights,
                event
            )
            
            return output
            
        return self._distribute_computation(x, moe_forward)
        
    def cleanup(self):
        """Cleanup resources"""
        if self.use_smallpond and self.sp_session:
            self.sp_session.shutdown()

# Initialize components for export
flash_mla = FlashMLAAttention
moe_attention = OptimizedMoEAttention
