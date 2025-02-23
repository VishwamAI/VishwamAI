"""Flash Attention implementation optimized for TPU."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class FlashAttention(nn.Module):
    """Memory-efficient attention implementation using block-sparse patterns."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        block_size: int = 256,
        dropout_prob: float = 0.1,
        attention_scale: Optional[float] = None,
        causal: bool = True,
        use_rotary: bool = False,
    ):
        """Initialize flash attention module.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            block_size: Size of attention blocks for chunked computation
            dropout_prob: Dropout probability
            attention_scale: Optional custom attention scale factor
            causal: Whether to use causal attention masking
            use_rotary: Whether to use rotary position embeddings
        """
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by number of heads ({num_attention_heads})"
            )
            
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.block_size = block_size
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

    def _chunk_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention scores in chunks to save memory.
        
        Args:
            q: Query tensor of shape [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor of shape [batch_size, num_heads, seq_len, head_dim]
            v: Value tensor of shape [batch_size, num_heads, seq_len, head_dim]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Output tensor of shape [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Compute number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # Initialize output tensor
        output = torch.zeros_like(q)
        
        # Compute attention scores in blocks
        for i in range(num_blocks):
            # Get current block range
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, seq_len)
            
            # Extract current query block
            q_block = q[:, :, start_idx:end_idx]
            
            # Initialize accumulator for current block
            block_output = torch.zeros_like(q_block)
            normalizer = torch.zeros((batch_size, num_heads, end_idx-start_idx, 1), device=q.device)
            
            # Process key-value blocks
            for j in range(num_blocks):
                # Get key-value block range
                k_start_idx = j * self.block_size
                k_end_idx = min(k_start_idx + self.block_size, seq_len)
                
                # Skip if causal and this block is ahead
                if self.causal and k_start_idx > start_idx:
                    continue
                
                # Extract current key-value block
                k_block = k[:, :, k_start_idx:k_end_idx]
                v_block = v[:, :, k_start_idx:k_end_idx]
                
                # Compute attention scores for current block
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * self.scale
                
                # Apply causal mask within block if needed
                if self.causal and k_start_idx <= end_idx:
                    causal_mask = torch.triu(
                        torch.ones((end_idx-start_idx, k_end_idx-k_start_idx), 
                                 dtype=torch.bool, device=scores.device),
                        diagonal=k_start_idx-start_idx+1
                    )
                    scores.masked_fill_(causal_mask, float("-inf"))
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    block_mask = attention_mask[:, :, start_idx:end_idx, k_start_idx:k_end_idx]
                    scores = scores + block_mask
                
                # Apply softmax and dropout
                attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
                attn_weights = self.dropout(attn_weights)
                
                # Update block output and normalizer
                block_output += torch.matmul(attn_weights, v_block)
                normalizer += attn_weights.sum(dim=-1, keepdim=True)
            
            # Normalize block output
            block_output = block_output / (normalizer + 1e-6)
            
            # Update output tensor
            output[:, :, start_idx:end_idx] = block_output
            
        return output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute flash attention over input hidden states.
        
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
        
        # Handle cached key-value pairs for incremental decoding
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            
        # Compute chunked attention
        attn_output = self._chunk_attention(q, k, v, attention_mask)
        
        # Merge heads and apply output projection
        output = self.o_proj(self._merge_heads(attn_output))
        
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
            f"block_size={self.block_size}, "
            f"causal={self.causal}, "
            f"rotary={self.use_rotary}"
        )
