"""Multi-Level Attention block implementation."""

from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .attention import MLAAttention
from .layer_manager import MLALayerManager
from .residual import MLAResidual, AdaptiveResidual

class MLABlock(nn.Module):
    """Multi-Level Attention transformer block."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_attention_levels: int = 3,
        intermediate_size: Optional[int] = None,
        activation: Union[str, Callable] = "gelu",
        attention_dropout_prob: float = 0.1,
        hidden_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_adaptive_residual: bool = True,
        use_layer_scale: bool = True,
        layer_scale_init: float = 0.1,
        level_scale_factor: float = 0.5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize MLA block.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            num_attention_levels: Number of attention levels
            intermediate_size: Size of FFN intermediate dimension
            activation: Activation function or name
            attention_dropout_prob: Attention dropout probability
            hidden_dropout_prob: Hidden state dropout probability
            layer_norm_eps: Layer normalization epsilon
            use_adaptive_residual: Whether to use adaptive residual connections
            use_layer_scale: Whether to use layer scaling
            layer_scale_init: Initial value for layer scale parameters
            level_scale_factor: Scaling factor between attention levels
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = 4 * hidden_size
            
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_attention_levels = num_attention_levels
        self.intermediate_size = intermediate_size
        
        # Pre-normalization layers
        self.attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, **factory_kwargs)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, **factory_kwargs)
        
        # Multi-level attention
        self.attention = MLAAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_attention_groups=num_attention_levels,
            attention_dropout_prob=attention_dropout_prob,
            position_dropout_prob=hidden_dropout_prob,
            use_rotary=True,
            **factory_kwargs
        )
        
        # FFN layers
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, **factory_kwargs),
            self._get_activation(activation),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(intermediate_size, hidden_size, **factory_kwargs),
            nn.Dropout(hidden_dropout_prob)
        )
        
        # Layer manager
        self.layer_manager = MLALayerManager(
            hidden_size=hidden_size,
            num_attention_levels=num_attention_levels,
            adaptive_computation=True,
            layer_dropout=hidden_dropout_prob,
            **factory_kwargs
        )
        
        # Residual handler
        residual_class = AdaptiveResidual if use_adaptive_residual else MLAResidual
        self.residual = residual_class(
            hidden_size=hidden_size,
            num_attention_levels=num_attention_levels,
            num_heads=num_attention_heads,
            use_layer_scale=use_layer_scale,
            init_scale=layer_scale_init,
            level_scale_factor=level_scale_factor,
            dropout_prob=hidden_dropout_prob,
            **factory_kwargs
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize module weights."""
        # Initialize FFN weights
        for module in self.ffn.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _get_activation(self, activation: Union[str, Callable]) -> Callable:
        """Get activation function.
        
        Args:
            activation: Activation function name or callable
            
        Returns:
            Activation function
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
        attention_level: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict]]:
        """Forward pass through MLA block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask tensor
            position_embeddings: Optional tuple of (cos, sin) rotary position embeddings
            past_key_value: Optional tuple of cached (key, value) tensors
            use_cache: Whether to return key/value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            attention_level: Optional specific attention level to use
            
        Returns:
            Tuple containing:
            - Output tensor
            - Optional tuple of cached (key, value) tensors
            - Optional dictionary of auxiliary outputs
        """
        # Get layer configuration and importance scores
        layer_inputs, layer_outputs = self.layer_manager(hidden_states, attention_mask)
        importance_scores = layer_outputs["importance_scores"]
        
        # Store residual pre-norm
        attn_residual = hidden_states
        
        # Self-attention
        hidden_states = self.attn_norm(hidden_states)
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = attention_outputs[0]
        
        # Apply attention residual connection
        hidden_states = self.residual(
            hidden_states=hidden_states,
            residual=attn_residual,
            attention_level=attention_level or 0,
            pre_norm=True
        )
        
        # Store residual pre-norm
        ffn_residual = hidden_states
        
        # FFN layers
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        
        # Apply FFN residual connection
        hidden_states = self.residual(
            hidden_states=hidden_states,
            residual=ffn_residual,
            attention_level=attention_level or 0,
            pre_norm=True
        )
        
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (attention_outputs[1],)
            
        if output_attentions:
            outputs += (attention_outputs[2 if use_cache else 1],)
            
        # Include auxiliary outputs
        aux_outputs = {
            **layer_outputs,
            "importance_scores": importance_scores,
        }
        outputs += (aux_outputs,)
        
        return outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_heads={self.num_attention_heads}, "
            f"num_levels={self.num_attention_levels}, "
            f"ffn_dim={self.intermediate_size}"
        )
