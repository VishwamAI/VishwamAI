"""
Transformer block implementation with advanced features.

This module provides the core building block for transformer models,
incorporating multi-head attention, feed-forward networks,
and normalization layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from vishwamai.utils.config import ModelConfig
from vishwamai.models.base_layers import Linear, LayerNorm
from vishwamai.models.MLA import MLA
from vishwamai.models.MLP import MLP
from vishwamai.models.rmsnorm import RMSNorm

class Block(nn.Module):
    """
    Transformer block with configurable components.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        use_rmsnorm: bool = False
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Attention
        if config.use_mla:
            self.attention = MLA(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            
        # Feed-forward network
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
        
        # Normalization layers
        norm_class = RMSNorm if use_rmsnorm else LayerNorm
        self.ln1 = norm_class(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = norm_class(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Optional rotary positional embeddings
        self.rotary_emb = None
        if getattr(config, 'use_rotary_embeddings', False):
            from vishwamai.models.rope import RotaryPositionalEmbedding
            self.rotary_emb = RotaryPositionalEmbedding(
                dim=config.hidden_size // config.num_heads,
                max_position_embeddings=config.max_position_embeddings
            )
            
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass for transformer block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            position_ids: Optional position IDs for positional encoding
            past_key_value: Optional cached key/value states
            output_attentions: Whether to return attention weights
            use_cache: Whether to use cached key/value states
            
        Returns:
            Dict containing:
                hidden_states: Output tensor
                attention_weights: Optional attention weights
                past_key_value: Optional cached states
        """
        # Pre-LayerNorm architecture
        residual = hidden_states
        hidden_states = self.ln1(hidden_states)
        
        # Apply attention
        if isinstance(self.attention, MLA):
            attention_outputs = self.attention(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = attention_outputs
            attention_weights = None
        else:
            # Standard multi-head attention
            if self.rotary_emb is not None and position_ids is not None:
                # Apply rotary embeddings
                hidden_states = self.rotary_emb(hidden_states, position_ids)
                
            attention_outputs = self.attention(
                hidden_states,
                hidden_states,
                hidden_states,
                key_padding_mask=attention_mask,
                need_weights=output_attentions
            )
            hidden_states = attention_outputs[0]
            attention_weights = attention_outputs[1] if output_attentions else None
            
        # First residual connection
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.ln2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = {'hidden_states': hidden_states}
        if output_attentions:
            outputs['attention_weights'] = attention_weights
        if use_cache:
            outputs['past_key_value'] = past_key_value
            
        return outputs
        
    def extra_repr(self) -> str:
        """String representation of module."""
        return f'layer_idx={self.layer_idx}'
