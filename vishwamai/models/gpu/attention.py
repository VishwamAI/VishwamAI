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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y

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
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0):
        def attention_forward(inputs):
            with autocast(enabled=self.use_amp):
                # Project and reshape queries, keys, and values
                q = self._reshape_for_multihead(self.q_proj(inputs))
                k = self._reshape_for_multihead(self.k_proj(inputs if context is None else context))
                v = self._reshape_for_multihead(self.v_proj(inputs if context is None else context))

                # Get device properties for SM configuration
                device_props = torch.cuda.get_device_properties(inputs.device)
                num_sm = device_props.multi_processor_count
                
                seqlens_k = torch.tensor([k.size(2)], dtype=torch.int32, device=inputs.device)
                
                # Configure SM parts based on device capabilities
                mla_metadata, num_splits = get_mla_metadata(
                    seqlens_k,
                    self.num_heads,
                    self.num_heads,
                    num_sm // 2  # Dynamic SM partitioning
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
                
                # Compute attention with gradient scaling
                with torch.cuda.amp.autocast(enabled=self.use_amp):
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
                output = self.scaler.scale(self.o_proj(output)) if self.training else self.o_proj(output)

                # Synchronize if in distributed mode
                if dist.is_initialized() and self.training:
                    dist.all_reduce(output)
                    output.div_(dist.get_world_size())
                    
                return output
            
        # Distribute computation if in a distributed environment
        if dist.is_initialized():
            torch.cuda.synchronize() # Ensure CUDA operations are complete
            with torch.cuda.stream(torch.cuda.Stream()):
                output = self._distribute_computation(x, attention_forward)
            torch.cuda.synchronize()
            return output
        return attention_forward(x)