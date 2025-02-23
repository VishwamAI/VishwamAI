"""Combined MoE-MLA transformer block implementation."""

from typing import Optional, Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn

from ..moe import MoELayer, create_mla_block
from ..mla import MLABlock, create_mla_block

class MoEMLABlock(nn.Module):
    """Transformer block combining MoE and MLA mechanisms."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_attention_levels: int = 3,
        num_experts: int = 8,
        expert_capacity_factor: float = 1.25,
        moe_layer_position: str = "post_attention",
        use_expert_choice: bool = True,
        share_expert_params: bool = False,
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
        """Initialize MoE-MLA block.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_heads: Number of attention heads
            num_attention_levels: Number of attention levels
            num_experts: Number of experts in MoE layer
            expert_capacity_factor: Expert capacity factor
            moe_layer_position: Position of MoE layer ('pre_attention' or 'post_attention')
            use_expert_choice: Whether to use expert choice routing
            share_expert_params: Whether to share expert parameters across levels
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
        self.moe_layer_position = moe_layer_position
        self.share_expert_params = share_expert_params
        
        # MLA block configuration
        mla_config = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_attention_levels": num_attention_levels,
            "intermediate_size": intermediate_size,
            "activation": activation,
            "attention_dropout_prob": attention_dropout_prob,
            "hidden_dropout_prob": hidden_dropout_prob,
            "layer_norm_eps": layer_norm_eps,
            "use_adaptive_residual": use_adaptive_residual,
            "use_layer_scale": use_layer_scale,
            "layer_scale_init": layer_scale_init,
            "level_scale_factor": level_scale_factor,
        }
        
        # Create MLA block
        self.mla = create_mla_block(mla_config, device=device, dtype=dtype)
        
        # MoE layer configuration
        moe_config = {
            "hidden_size": hidden_size,
            "num_experts": num_experts,
            "expert_capacity_factor": expert_capacity_factor,
            "use_expert_choice": use_expert_choice,
            "intermediate_size": intermediate_size,
            "dropout_prob": hidden_dropout_prob,
            "router_z_loss_coef": 0.001,
            "router_aux_loss_coef": 0.001,
        }
        
        # Create MoE layers (one per attention level if not sharing)
        if share_expert_params:
            self.moe = MoELayer(**moe_config, **factory_kwargs)
        else:
            self.moe = nn.ModuleList([
                MoELayer(**moe_config, **factory_kwargs)
                for _ in range(num_attention_levels)
            ])
            
        # Layer normalization
        self.pre_moe_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps, **factory_kwargs)
            
    def _get_moe_layer(self, level: int) -> MoELayer:
        """Get MoE layer for given attention level.
        
        Args:
            level: Attention level index
            
        Returns:
            MoE layer for level
        """
        if self.share_expert_params:
            return self.moe
        else:
            return self.moe[level]
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        attention_level: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Forward pass through MoE-MLA block.
        
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
            - Optional dictionary of MLA auxiliary outputs
            - Optional dictionary of MoE auxiliary outputs
        """
        # Pre-MoE processing
        if self.moe_layer_position == "pre_attention":
            residual = hidden_states
            hidden_states = self.pre_moe_norm(hidden_states)
            
            # Process through MoE layer
            moe_layer = self._get_moe_layer(attention_level or 0)
            hidden_states, moe_aux = moe_layer(hidden_states)
            
            # Add residual
            hidden_states = hidden_states + residual
            
        # Process through MLA block
        mla_outputs = self.mla(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            attention_level=attention_level,
        )
        hidden_states = mla_outputs[0]
        mla_aux = mla_outputs[-1] if len(mla_outputs) > 2 else None
        
        # Post-MLA MoE processing
        if self.moe_layer_position == "post_attention":
            residual = hidden_states
            hidden_states = self.pre_moe_norm(hidden_states)
            
            # Process through MoE layer
            moe_layer = self._get_moe_layer(attention_level or 0)
            hidden_states, moe_aux = moe_layer(hidden_states)
            
            # Add residual
            hidden_states = hidden_states + residual
        
        # Prepare outputs
        outputs = (hidden_states,)
        if use_cache:
            outputs += (mla_outputs[1],)
            
        return outputs + (mla_aux, moe_aux)
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"moe_position={self.moe_layer_position}, "
            f"shared_experts={self.share_expert_params}"
        )
