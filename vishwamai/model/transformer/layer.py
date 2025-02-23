"""Base transformer layer implementations."""

from typing import Optional, Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn

from ..attention import SelfAttention, create_attention_mask
from .config import TransformerConfig

class TransformerLayer(nn.Module):
    """Base transformer layer with self-attention and FFN."""
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize transformer layer.
        
        Args:
            config: Transformer configuration
            layer_idx: Index of this layer
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        # Layer normalization
        self.pre_attention_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            **factory_kwargs
        )
        self.pre_ffn_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps,
            **factory_kwargs
        )
        
        # Self-attention
        self.attention = SelfAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.attention_dropout_prob,
            max_position_embeddings=config.max_position_embeddings,
            use_rotary=config.position_embedding_type == "rotary",
            bias=True,
            **factory_kwargs
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, **factory_kwargs),
            self._get_activation(config.hidden_act),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size, **factory_kwargs),
            nn.Dropout(config.hidden_dropout_prob)
        )
        
        # Initialize weights
        self._init_weights(config)
        
    def _init_weights(self, config: TransformerConfig):
        """Initialize layer weights.
        
        Args:
            config: Transformer configuration
        """
        # Initialize FFN weights
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _get_activation(self, activation: Union[str, nn.Module]) -> nn.Module:
        """Get activation function.
        
        Args:
            activation: Activation function name or module
            
        Returns:
            Activation module
        """
        if isinstance(activation, str):
            return {
                "relu": nn.ReLU(),
                "gelu": nn.GELU(),
                "silu": nn.SiLU(),
                "swish": nn.SiLU(),
            }[activation]
        return activation
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through transformer layer.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask tensor
            position_embeddings: Optional tuple of (cos, sin) rotary position embeddings
            past_key_value: Optional tuple of cached (key, value) tensors
            use_cache: Whether to return key/value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            **kwargs: Additional arguments passed to attention module
            
        Returns:
            Tuple containing:
            - Output tensor
            - Optional tuple of cached (key, value) tensors
            - Optional attention weights tensor
        """
        # Store residual
        residual = hidden_states
        
        # Pre-attention normalization
        hidden_states = self.pre_attention_norm(hidden_states)
        
        # Self-attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = attention_outputs[0]
        
        # Add residual connection
        hidden_states = residual + hidden_states
        
        # Store residual
        residual = hidden_states
        
        # Pre-FFN normalization
        hidden_states = self.pre_ffn_norm(hidden_states)
        
        # FFN
        hidden_states = self.ffn(hidden_states)
        
        # Add residual connection
        hidden_states = residual + hidden_states
        
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (attention_outputs[1],)
            
        if output_attentions:
            attention_weights = attention_outputs[2 if use_cache else 1]
            outputs += (attention_weights,)
            
        return outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"layer_idx={self.layer_idx}, "
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_attention_heads}"
        )

class PreNormTransformerLayer(TransformerLayer):
    """Transformer layer with pre-normalization architecture."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through pre-norm transformer layer."""
        # Store residual
        residual = hidden_states
        
        # Pre-attention normalization
        hidden_states = self.pre_attention_norm(hidden_states)
        
        # Self-attention
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        hidden_states = attention_outputs[0]
        
        # Add residual connection
        hidden_states = residual + hidden_states
        
        # Store residual
        residual = hidden_states
        
        # FFN with pre-normalization
        hidden_states = self.pre_ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        
        # Add residual connection
        hidden_states = residual + hidden_states
        
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (attention_outputs[1],)
            
        if output_attentions:
            outputs += (attention_outputs[2 if use_cache else 1],)
            
        return outputs

def get_transformer_layer(
    config: TransformerConfig,
    layer_idx: int,
    pre_norm: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> TransformerLayer:
    """Get appropriate transformer layer based on configuration.
    
    Args:
        config: Transformer configuration
        layer_idx: Layer index
        pre_norm: Whether to use pre-normalization architecture
        device: Device to create tensors on
        dtype: Data type for parameters
        
    Returns:
        Transformer layer instance
    """
    layer_class = PreNormTransformerLayer if pre_norm else TransformerLayer
    return layer_class(
        config=config,
        layer_idx=layer_idx,
        device=device,
        dtype=dtype
    )
