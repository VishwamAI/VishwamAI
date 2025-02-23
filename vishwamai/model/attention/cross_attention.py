"""Cross-attention implementation for encoder-decoder architectures."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """Multi-head cross-attention module."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout_prob: float = 0.1,
        attention_scale: Optional[float] = None,
        use_rotary: bool = False,
        bias: bool = False,
    ):
        """Initialize cross-attention module.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            dropout_prob: Dropout probability
            attention_scale: Optional custom attention scale factor
            use_rotary: Whether to use rotary position embeddings
            bias: Whether to use bias in projection layers
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
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        
        # Attention options
        self.use_rotary = use_rotary
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights."""
        # Initialize using Xavier uniform with small gain for stable training
        gain = 1.0 / math.sqrt(2.0)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.o_proj.weight, gain=gain)
        
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.o_proj.bias)
        
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
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute cross-attention over encoder hidden states.
        
        Args:
            hidden_states: Decoder hidden states of shape [batch_size, tgt_len, hidden_size]
            encoder_hidden_states: Encoder hidden states of shape [batch_size, src_len, hidden_size]
            attention_mask: Optional attention mask tensor
            position_embeddings: Optional tuple of (cos, sin) rotary position embeddings for queries
            encoder_position_embeddings: Optional tuple of (cos, sin) rotary embeddings for keys
            past_key_value: Optional tuple of cached (key, value) tensors
            use_cache: Whether to return key/value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple containing:
            - Output tensor of shape [batch_size, tgt_len, hidden_size]
            - Optional attention weights tensor
            - Optional tuple of cached (key, value) tensors
        """
        batch_size, tgt_len, _ = hidden_states.size()
        src_len = encoder_hidden_states.size(1)
        
        # Project queries from decoder hidden states
        q = self._split_heads(self.q_proj(hidden_states))  # [B, H, T, D]
        
        # Project keys and values from encoder hidden states
        if past_key_value is not None:
            k, v = past_key_value
        else:
            k = self._split_heads(self.k_proj(encoder_hidden_states))  # [B, H, S, D]
            v = self._split_heads(self.v_proj(encoder_hidden_states))  # [B, H, S, D]
            
        # Apply rotary embeddings if provided
        if self.use_rotary:
            if position_embeddings is not None:
                q, _ = self._apply_rotary_embeddings(q, q, *position_embeddings)
            if encoder_position_embeddings is not None:
                k, _ = self._apply_rotary_embeddings(k, k, *encoder_position_embeddings)
            
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, S]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # Compute attention weights and apply dropout
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, D]
        attn_output = self._merge_heads(attn_output)  # [B, T, H*D]
        
        # Final projection
        output = self.o_proj(attn_output)
        
        # Return outputs based on flags
        outputs = (output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += ((k, v),)
            
        return outputs

    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_attention_heads}, "
            f"head_dim={self.head_dim}, "
            f"rotary={self.use_rotary}"
        )
