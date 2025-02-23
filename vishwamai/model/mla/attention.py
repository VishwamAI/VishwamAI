"""Multi-Level Attention mechanisms for MLA layers."""

from typing import Optional, Tuple, Dict, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLAAttention(nn.Module):
    """Multi-Level Attention with grouped heads."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_attention_groups: int,
        attention_dropout_prob: float = 0.1,
        position_dropout_prob: float = 0.1,
        attention_scale: Optional[float] = None,
        use_rotary: bool = False,
        max_position_embeddings: int = 2048,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize MLA attention.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Total number of attention heads
            num_attention_groups: Number of attention head groups
            attention_dropout_prob: Attention score dropout probability
            position_dropout_prob: Position embedding dropout probability
            attention_scale: Optional custom attention scale factor
            use_rotary: Whether to use rotary position embeddings
            max_position_embeddings: Maximum sequence length for positions
            bias: Whether to use bias in projections
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible by number of heads ({num_attention_heads})"
            )
            
        if num_attention_heads % num_attention_groups != 0:
            raise ValueError(
                f"Number of heads ({num_attention_heads}) must be divisible by number of groups ({num_attention_groups})"
            )
            
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_groups = num_attention_groups
        self.heads_per_group = num_attention_heads // num_attention_groups
        self.head_dim = hidden_size // num_attention_heads
        self.scale = attention_scale or 1.0 / math.sqrt(self.head_dim)
        self.use_rotary = use_rotary
        
        # Multi-level projections
        self.q_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // num_attention_groups, bias=bias, **factory_kwargs)
            for _ in range(num_attention_groups)
        ])
        
        self.k_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // num_attention_groups, bias=bias, **factory_kwargs)
            for _ in range(num_attention_groups)
        ])
        
        self.v_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size // num_attention_groups, bias=bias, **factory_kwargs)
            for _ in range(num_attention_groups)
        ])
        
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias, **factory_kwargs)
        
        # Dropouts
        self.attention_dropout = nn.Dropout(attention_dropout_prob)
        self.position_dropout = nn.Dropout(position_dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize attention weights."""
        # Initialize Q projections
        for proj in self.q_projs:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
                
        # Initialize K projections
        for proj in self.k_projs:
            nn.init.xavier_uniform_(proj.weight, gain=1.0 / math.sqrt(2))
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
                
        # Initialize V projections
        for proj in self.v_projs:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
                
        # Initialize output projection
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)
            
    def _split_heads_for_group(
        self,
        x: torch.Tensor,
        group_idx: int
    ) -> torch.Tensor:
        """Split hidden dimension into attention heads for a group.
        
        Args:
            x: Input tensor of shape [batch_size, seq_length, hidden_size//num_groups]
            group_idx: Index of attention group
            
        Returns:
            Tensor of shape [batch_size, num_heads_per_group, seq_length, head_dim]
        """
        batch_size, seq_length, _ = x.size()
        x = x.view(batch_size, seq_length, self.heads_per_group, self.head_dim)
        return x.transpose(1, 2)
        
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads back into hidden dimension.
        
        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_length, head_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_length, hidden_size]
        """
        batch_size, _, seq_length, _ = x.size()
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_length, self.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
        """Compute multi-level attention over input hidden states.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            position_embeddings: Optional tuple of (cos, sin) rotary position embeddings
            past_key_value: Optional tuple of cached (key, value) tensors
            use_cache: Whether to return key/value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple containing:
            - Output tensor of shape [batch_size, seq_length, hidden_size]
            - Optional tuple of cached (key, value) tensors
            - Optional attention weights tensor
        """
        batch_size, seq_length, _ = hidden_states.size()
        
        # Process each attention group
        group_outputs = []
        group_attentions = []
        
        # Initialize cached key/value tensors
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_k = past_k.chunk(self.num_attention_groups, dim=2)
            past_v = past_v.chunk(self.num_attention_groups, dim=2)
        else:
            past_k = [None] * self.num_attention_groups
            past_v = [None] * self.num_attention_groups
            
        # Process each attention group
        for group_idx in range(self.num_attention_groups):
            # Project Q/K/V for this group
            q = self.q_projs[group_idx](hidden_states)  # [B, L, H/G]
            k = self.k_projs[group_idx](hidden_states)  # [B, L, H/G]
            v = self.v_projs[group_idx](hidden_states)  # [B, L, H/G]
            
            # Split heads
            q = self._split_heads_for_group(q, group_idx)  # [B, H/G, L, D]
            k = self._split_heads_for_group(k, group_idx)  # [B, H/G, L, D]
            v = self._split_heads_for_group(v, group_idx)  # [B, H/G, L, D]
            
            # Apply rotary embeddings if provided
            if self.use_rotary and position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
                
            # Handle cached key/value tensors
            if past_k[group_idx] is not None:
                k = torch.cat([past_k[group_idx], k], dim=2)
                v = torch.cat([past_v[group_idx], v], dim=2)
                
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H/G, L, L]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                scores = scores + attention_mask
                
            # Compute attention weights and apply dropout
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
            attn_weights = self.attention_dropout(attn_weights)
            
            # Compute attention output
            group_output = torch.matmul(attn_weights, v)  # [B, H/G, L, D]
            group_outputs.append(group_output)
            
            if output_attentions:
                group_attentions.append(attn_weights)
                
        # Concatenate group outputs
        attn_output = torch.cat(group_outputs, dim=1)  # [B, H, L, D]
        
        # Merge heads and apply output projection
        attn_output = self._merge_heads(attn_output)  # [B, L, H]
        output = self.o_proj(attn_output)  # [B, L, H]
        
        # Prepare outputs
        outputs = (output,)
        
        if use_cache:
            # Concatenate cached key/value tensors
            k_cache = torch.cat([k.unsqueeze(2) for k in past_k], dim=2)
            v_cache = torch.cat([v.unsqueeze(2) for v in past_v], dim=2)
            outputs += ((k_cache, v_cache),)
            
        if output_attentions:
            # Stack attention weights from all groups
            attention_weights = torch.stack(group_attentions, dim=1)  # [B, G, H/G, L, L]
            outputs += (attention_weights,)
            
        return outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_attention_heads}, "
            f"num_groups={self.num_attention_groups}, "
            f"head_dim={self.head_dim}, "
            f"rotary={self.use_rotary}"
        )
