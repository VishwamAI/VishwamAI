"""CUDA-optimized Flash Multi-head Linear Attention implementation."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math
import warnings

try:
    # Try importing the CUDA extension
    from torch.utils.cpp_extension import load

    # Get the directory containing this file
    module_path = os.path.dirname(os.path.abspath(__file__))
    csrc_path = os.path.join(module_path, "csrc")

    # Load and compile the CUDA extension
    flash_mla_cuda_module = load(
        name="flash_mla_cuda",
        sources=[
            os.path.join(csrc_path, "flash_mla_cuda.cpp"),
            os.path.join(csrc_path, "flash_mla_kernel.cu")
        ],
        verbose=True
    )
    CUDA_ENABLED = True

except ImportError as e:
    warnings.warn(f"Failed to import CUDA extension: {e}. Using fallback implementation.")
    CUDA_ENABLED = False

class FlashMLAFunction(torch.autograd.Function):
    """Autograd function for Flash MLA."""
    
    @staticmethod
    def forward(ctx, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                scale: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for Flash MLA.
        
        Args:
            query: [batch_size, seqlen_q, num_heads, head_dim]
            key: [batch_size, seqlen_k, num_heads_k, head_dim]
            value: [batch_size, seqlen_k, num_heads_k, head_dim_v]
            mask: Optional attention mask
            dropout_p: Dropout probability
            is_causal: Whether to use causal masking
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
            
        Returns:
            Tuple of:
            - output: [batch_size, seqlen_q, num_heads, head_dim_v]
            - softmax_lse: [batch_size, num_heads, seqlen_q]
        """
        # Save context for backward pass
        ctx.save_for_backward(query, key, value, mask)
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        
        if scale is None:
            scale = 1.0 / math.sqrt(query.size(-1))
        
        # Use CUDA implementation if available
        if CUDA_ENABLED:
            # Get sequence lengths for KV cache
            batch_size = query.size(0)
            seqlen_k = key.size(1)
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, 
                                    dtype=torch.int32, device=query.device)
            
            # Create dummy block table for non-cached version
            block_table = torch.arange(seqlen_k, device=query.device).reshape(1, -1).expand(batch_size, -1)
            
            # Get scheduling metadata
            tile_metadata, num_splits = flash_mla_cuda_module.get_mla_metadata(
                cu_seqlens[:-1],  # Remove last entry which is total length
                batch_size=batch_size,
                block_size_n=64,
                fixed_overhead_num_blocks=2,
                num_sm_parts=6
            )
            
            # Call CUDA kernel
            output, softmax_lse = flash_mla_cuda_module.flash_mla_forward(
                query, key, value,
                block_table,
                cu_seqlens,
                scale,
                is_causal,
                tile_metadata,
                num_splits
            )
            
        else:
            # Fallback PyTorch implementation
            batch_size, seqlen_q, num_heads, head_dim = query.shape
            _, seqlen_k, _, _ = key.shape
            
            # Compute attention scores
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            
            if is_causal:
                causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, 
                                                  device=query.device), diagonal=1)
                scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            
            if mask is not None:
                scores = scores + mask.unsqueeze(1)
                
            # Compute softmax and dropout
            softmax_lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            attn = torch.exp(scores - softmax_lse)
            
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p)
            
            # Compute output
            output = torch.matmul(attn, value)
            softmax_lse = softmax_lse.squeeze(-1)
            
        return output, softmax_lse
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, 
                grad_softmax_lse: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """Backward pass for Flash MLA."""
        query, key, value, mask = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        is_causal = ctx.is_causal
        
        grad_query = grad_key = grad_value = grad_mask = None
        
        # TODO: Implement proper backward pass with CUDA
        # For now, let PyTorch handle the backward pass automatically
        
        return grad_query, grad_key, grad_value, grad_mask, None, None, None

class FlashMLA(nn.Module):
    """Flash Multi-head Linear Attention module."""
    
    def __init__(self,
                 head_dim: int = 64,
                 num_heads: int = 8,
                 dropout_p: float = 0.0,
                 is_causal: bool = False,
                 scale: Optional[float] = None):
        """Initialize FlashMLA module.
        
        Args:
            head_dim: Dimension of each attention head
            num_heads: Number of attention heads
            dropout_p: Dropout probability
            is_causal: Whether to use causal masking
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
        """
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.is_causal = is_causal
        self.scale = scale if scale is not None else 1.0 / math.sqrt(head_dim)
        
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: [batch_size, seqlen_q, num_heads, head_dim]
            key: [batch_size, seqlen_k, num_heads_k, head_dim]
            value: [batch_size, seqlen_k, num_heads_k, head_dim_v]
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seqlen_q, num_heads, head_dim_v]
        """
        return FlashMLAFunction.apply(
            query, key, value, mask,
            self.dropout_p,
            self.is_causal,
            self.scale
        )[0]  # Only return output, not softmax_lse

def flash_mla_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Functional interface for Flash MLA.
    
    Args:
        query: [batch_size, seqlen_q, num_heads, head_dim]
        key: [batch_size, seqlen_k, num_heads_k, head_dim]
        value: [batch_size, seqlen_k, num_heads_k, head_dim_v]
        mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Optional scaling factor (default: 1/sqrt(head_dim))
        
    Returns:
        Tuple of:
        - output: [batch_size, seqlen_q, num_heads, head_dim_v]
        - softmax_lse: [batch_size, num_heads, seqlen_q]
    """
    return FlashMLAFunction.apply(query, key, value, mask, dropout_p, is_causal, scale)