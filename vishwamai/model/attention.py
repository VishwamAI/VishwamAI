"""
Attention mechanisms for Vishwamai model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from ..utils.t4_utils import get_device_capabilities
from .embeddings import apply_rotary_embeddings
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with precision and flash attention support
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_flash_attention = use_flash_attention and get_device_capabilities()["flash_attention"]
        
        assert self.head_dim * num_heads == hidden_size, \
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        
        # Linear projections with specified dtype
        self.q_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.out_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights"""
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.normal_(proj.weight, mean=0.0, std=0.02)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
                
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass implementing either flash or standard attention
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape to [batch, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if position_embeddings is not None:
            query = apply_rotary_embeddings(query, position_embeddings)
            key = apply_rotary_embeddings(key, position_embeddings)
        
        # Choose attention implementation
        if self.use_flash_attention and self.training:
            attn_output = self._flash_attention(
                query, key, value, attention_mask
            )
        else:
            attn_output = self._standard_attention(
                query, key, value, attention_mask
            )
            
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attn_output)
        
    def _flash_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flash attention implementation for training
        """
        try:
            from flash_attn import flash_attn_func
            
            # Handle dropout in training
            dropout_p = self.dropout.p if self.training else 0.0
            
            # Flash attention expects inputs in [batch, seq_len, num_heads, head_dim]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            
            output = flash_attn_func(
                query, key, value,
                dropout_p=dropout_p,
                causal=False,
                softmax_scale=1.0 / (self.head_dim ** 0.5)
            )
            
            # Return to [batch, num_heads, seq_len, head_dim]
            return output.transpose(1, 2)
            
        except ImportError:
            return self._standard_attention(query, key, value, attention_mask)
            
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention
        """
        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Add large negative value to padded positions
            attention_scores = attention_scores + (attention_mask * -10000.0)
        
        # Apply softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Get context
        context = torch.matmul(attention_probs, value)
        return context

class SelfAttention(MultiHeadAttention):
    """Self-attention wrapper"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings
        )

class CrossAttention(MultiHeadAttention):
    """Cross-attention for encoder-decoder attention"""
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Project query from decoder hidden states
        query = self.q_proj(hidden_states)
        
        # Project key and value from encoder hidden states
        key = self.k_proj(encoder_hidden_states)
        value = self.v_proj(encoder_hidden_states)
        
        # Reshape and apply attention
        batch_size, seq_len, _ = hidden_states.shape
        tgt_len = encoder_hidden_states.size(1)
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if provided
        if position_embeddings is not None:
            query = apply_rotary_embeddings(query, position_embeddings)
            key = apply_rotary_embeddings(key, position_embeddings)
            
        # Use standard attention for cross-attention
        attn_output = self._standard_attention(
            query, key, value, encoder_attention_mask
        )
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        
        return self.out_proj(attn_output)
