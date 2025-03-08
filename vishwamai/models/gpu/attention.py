"""
GPU-optimized attention mechanisms for VishwamAI.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from abc import ABC, abstractmethod
from torch.cuda.amp import autocast

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
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True):
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
        self.use_amp = use_amp  # Enable mixed precision
        self._reset_parameters()

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

class FlashMLAAttention(BaseAttention):
    """Optimized MLA attention with mixed precision support and FlashMLA kernels"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True):
        super().__init__(embed_dim, num_heads, dropout, use_amp)
        self.flash_config = Flash_fwd_kernel_traits_mla()
        
    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0):
        # Shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.size()

        # Project q,k,v with optimized kernels
        q = self._reshape_for_multihead(self.q_proj(x))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))

        # Get sequence metadata for FlashMLA
        seqlens_k = torch.tensor([k.size(2)], dtype=torch.int32, device=x.device)

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
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.o_proj(output)

class OptimizedMoEAttention(BaseAttention):
    """GPU-optimized MoE attention with DeepEP integration"""
    def __init__(self, embed_dim, num_heads, num_experts=4, dropout=0.1, use_amp=True):
        super().__init__(embed_dim, num_heads, dropout, use_amp)
        self.num_experts = num_experts
        
        # Initialize experts
        self.experts = nn.ModuleList([
            FlashMLAAttention(embed_dim, num_heads, dropout, use_amp)
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
        batch_size, seq_len, _ = x.size()

        # Get expert assignments
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        _, indices = router_probs.topk(k=2, dim=-1)
        
        # DeepEP dispatch
        buffer = self._get_buffer()
        dispatched_x, idx, weights, expert_counts, handle, event = buffer.dispatch(
            x, indices, router_probs
        )
        
        # Process tokens with experts
        expert_outputs = []
        start_idx = 0
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
        
        # DeepEP combine
        expert_outputs = torch.cat(expert_outputs, dim=0)
        output, _ = buffer.combine(
            expert_outputs, 
            handle,
            weights,
            event
        )
        
        return output

# Initialize components for export
flash_mla = FlashMLAAttention
moe_attention = OptimizedMoEAttention
