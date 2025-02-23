"""Self-attention implementation with optimizations for TPU."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Multi-head self-attention with optimizations for TPU."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float = 0.1,
        attention_scale: Optional[float] = None,
        causal: bool = True,
        use_rotary: bool = False,
        use_flash: bool = False,
    ):
        """Initialize self-attention module.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            dropout_prob: Dropout probability
            attention_scale: Optional custom attention scale factor
            causal: Whether to use causal attention masking
            use_rotary: Whether to use rotary position embeddings
            use_flash: Whether to use flash attention when available
        """
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by number of heads ({num_attention_heads})"
            )
            
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = attention_scale or 1.0 / math.sqrt(self.head_dim)
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Attention options
        self.causal = causal
        self.use_rotary = use_rotary
        self.use_flash = use_flash
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=1.0)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split hidden dimension into multiple attention heads.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        return x.transpose(1, 2)
        
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back into hidden dimension.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.hidden_size)
        
    def _apply_rotary_embeddings(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to queries and keys.
        
        Args:
            q: Query tensor
            k: Key tensor
            cos: Cosine part of rotary embeddings
            sin: Sine part of rotary embeddings
            
        Returns:
            Tuple of transformed (query, key) tensors
        """
        # Split dimensions for rotation
        q_split = q.view(*q.shape[:-1], -1, 2)
        k_split = k.view(*k.shape[:-1], -1, 2)
        
        # Apply rotation using einsum
        q_rot = torch.stack(
            [
                q_split[..., 0] * cos - q_split[..., 1] * sin,
                q_split[..., 1] * cos + q_split[..., 0] * sin
            ],
            dim=-1
        ).flatten(-2)
        
        k_rot = torch.stack(
            [
                k_split[..., 0] * cos - k_split[..., 1] * sin,
                k_split[..., 1] * cos + k_split[..., 0] * sin
            ],
            dim=-1
        ).flatten(-2)
        
        return q_rot, k_rot
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute self-attention over input hidden states.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            position_embeddings: Optional tuple of (cos, sin) rotary position embeddings
            past_key_value: Optional tuple of cached (key, value) tensors
            use_cache: Whether to return key/value tensors for incremental decoding
            
        Returns:
            Tuple containing:
            - Output tensor of shape [batch_size, seq_len, hidden_size]
            - Optional tuple of cached (key, value) tensors
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Project queries, keys and values
        q = self._split_heads(self.q_proj(hidden_states))  # [B, H, L, D]
        k = self._split_heads(self.k_proj(hidden_states))  # [B, H, L, D]  
        v = self._split_heads(self.v_proj(hidden_states))  # [B, H, L, D]
        
        # Apply rotary embeddings if provided
        if self.use_rotary and position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_rotary_embeddings(q, k, cos, sin)
            
        # Handle cached key-value pairs for incremental decoding
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        
        # Apply causal mask if needed
        if self.causal:
            causal_mask = torch.triu(
                torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device),
                diagonal=1
            )
            scores.masked_fill_(causal_mask, float("-inf"))
            
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # Compute attention weights and apply dropout
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D]
        attn_output = self._merge_heads(attn_output)  # [B, L, H*D]
        
        # Final projection
        output = self.o_proj(attn_output)
        
        # Return key-value pair if using cache
        if use_cache:
            return output, (k, v)
            
        return output, None

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_attention_heads}, "
            f"head_dim={self.head_dim}, "
            f"causal={self.causal}, "
            f"rotary={self.use_rotary}, "
            f"flash={self.use_flash}"
        )
