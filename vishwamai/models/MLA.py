"""
Multi-Level Attention (MLA) implementation for VishwamAI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from vishwamai.models.base_layers import Linear, LayerNorm
from vishwamai.utils.config import MLAConfig

class MLA(nn.Module):
    """
    Multi-Level Attention module that processes information at multiple 
    granularities using hierarchical attention mechanisms.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        num_levels: int = 3,
        combine_method: str = "concat",
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_levels = num_levels
        self.combine_method = combine_method
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = hidden_size // num_heads
        
        # Create attention layers for each level
        self.query = nn.ModuleList([
            Linear(hidden_size, hidden_size) 
            for _ in range(num_levels)
        ])
        self.key = nn.ModuleList([
            Linear(hidden_size, hidden_size)
            for _ in range(num_levels)
        ])
        self.value = nn.ModuleList([
            Linear(hidden_size, hidden_size)
            for _ in range(num_levels)
        ])
        
        # Output projection
        if combine_method == "concat":
            self.out_proj = Linear(hidden_size * num_levels, hidden_size)
        else:
            self.out_proj = Linear(hidden_size, hidden_size)
            
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)
        self.dropout = nn.Dropout(dropout)
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def _level_attention(
        self,
        level: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        scale_factor: int = 1
    ) -> torch.Tensor:
        batch_size, seq_len, _ = query.size()
        
        # Project query, key, value
        q = self._shape(self.query[level](query), seq_len, batch_size)
        k = self._shape(self.key[level](key), seq_len, batch_size)
        v = self._shape(self.value[level](value), seq_len, batch_size)
        
        # Scale query
        q = q * (self.head_dim ** -0.5)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1))
        
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            
        # Scale attention scores based on level
        attention_scores = attention_scores * scale_factor
        
        # Convert scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Get context
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, self.hidden_size)
        
        return context
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        
        # Process each attention level
        level_outputs = []
        for level in range(self.num_levels):
            # Adjust scale factor based on level
            scale_factor = 2 ** level
            
            level_output = self._level_attention(
                level=level,
                query=hidden_states,
                key=hidden_states,
                value=hidden_states,
                attention_mask=attention_mask,
                scale_factor=scale_factor
            )
            level_outputs.append(level_output)
            
        # Combine outputs from different levels
        if self.combine_method == "concat":
            combined = torch.cat(level_outputs, dim=-1)
        else:  # weighted sum
            weights = F.softmax(self.level_weights, dim=0)
            combined = sum(w * o for w, o in zip(weights, level_outputs))
            
        output = self.out_proj(combined)
        return output
