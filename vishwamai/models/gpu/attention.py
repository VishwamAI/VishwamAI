import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from abc import ABC, abstractmethod
from torch.cuda.amp import autocast
import torch.distributed as dist

# Assuming these are custom modules for VishwamAI
from vishwamai.models.gpu.optimizations.flash_mla import (
    Flash_fwd_kernel_traits_mla,
    Flash_fwd_mla_params,
    run_mha_fwd_splitkv_mla,
    get_mla_metadata
)
from vishwamai.models.gpu.optimizations.eplb import EPLB

# Base Attention Class
class BaseAttention(nn.Module, ABC):
    """Enhanced base class with GPU-aware initialization and mixed precision support"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True):
        super(BaseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for queries, keys, values, and output
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.use_amp = use_amp
        self._reset_parameters()

    def _reset_parameters(self):
        """GPU-optimized Xavier initialization for weights"""
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
        """Reshape tensor for multi-head attention"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
        
    def _distribute_computation(self, x, compute_fn):
        """Distribute computation across devices using PyTorch distributed"""
        if dist.is_initialized():
            dist.broadcast(x, src=0)
            return compute_fn(x)
        else:
            return compute_fn(x)

# Flash MLA Attention Class with Integrated Components
class FlashMLAAttention(BaseAttention):
    """Optimized Multi-Head Attention with FlashMLA kernels, Flash_fwd_mla_params, and run_mha_fwd_splitkv_mla"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True, causal=False):
        super().__init__(embed_dim, num_heads, dropout, use_amp)
        self.flash_config = Flash_fwd_kernel_traits_mla()  # Kernel traits for configuration
        self.causal = causal  # Configurable causal attention
        
    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0):
        def attention_forward(inputs):
            with autocast(enabled=self.use_amp):
                # Project and reshape queries, keys, and values
                q = self._reshape_for_multihead(self.q_proj(inputs))
                k = self._reshape_for_multihead(self.k_proj(inputs if context is None else context))
                v = self._reshape_for_multihead(self.v_proj(inputs if context is None else context))
                seqlens_k = torch.tensor([k.size(2)], dtype=torch.int32, device=inputs.device)
                
                # Get MLA metadata for scheduling
                mla_metadata, num_splits = get_mla_metadata(
                    seqlens_k,
                    self.num_heads,
                    self.num_heads,
                    8  # Number of SM parts
                )
                
                # Configure attention parameters using Flash_fwd_mla_params
                params = Flash_fwd_mla_params(
                    traits=self.flash_config,  # Kernel traits
                    head_size=self.head_dim,
                    num_heads=self.num_heads,
                    causal=self.causal,
                    sm_scale=0.5,  # Softmax scaling factor
                    mask=mask  # Attention mask
                )
                
                # Compute attention using run_mha_fwd_splitkv_mla
                output = run_mha_fwd_splitkv_mla(
                    q, k, v,                  # Queries, keys, values
                    seqlens_k,                # Key sequence lengths
                    params,                   # Configured parameters
                    tile_scheduler_metadata=mla_metadata,  # Metadata for tiling
                    num_splits=num_splits     # Number of splits
                )
                
                # Reshape output and apply final projection
                batch_size, seq_len, _ = inputs.size()
                output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
                return self.o_proj(output)
            
        # Distribute computation if in a distributed environment
        return self._distribute_computation(x, attention_forward)