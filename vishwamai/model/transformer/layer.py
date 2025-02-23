"""Transformer layer implementation managing block stacking."""

from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .block import TransformerBlock
from .moe_mla_block import MoEMLABlock

@dataclass
class LayerCache:
    """Cache for transformer layer states."""
    mla_caches: List[Dict[str, Any]]
    layer_outputs: List[jnp.ndarray]
    aux_outputs: List[Dict[str, Any]]

class TransformerLayer(nn.Module):
    """Manages transformer blocks and their connections."""
    
    # Architecture
    hidden_size: int
    num_attention_heads: int
    num_layers: int
    intermediate_size: Optional[int] = None
    head_dim: Optional[int] = None
    
    # Block Configuration
    num_moe_layers: int = 0
    moe_layer_frequency: int = 2  # Add MoE every N layers
    num_experts: int = 8
    expert_capacity_factor: float = 1.25
    num_experts_per_token: int = 2
    expert_hidden_size: Optional[int] = None
    
    # MLA Configuration
    use_mla: bool = True
    num_prev_layers: int = 4
    attention_window: int = 4
    
    # Common Configuration
    activation: str = "gelu"
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    drop_path: float = 0.0
    use_flash_attention: bool = False
    use_rope: bool = True
    max_sequence_length: int = 2048
    layer_norm_eps: float = 1e-5
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    deterministic: bool = False
    
    def setup(self):
        """Initialize transformer layer components."""
        # Determine layer types
        self.layer_types = self._determine_layer_types()
        
        # Create layers
        self.layers = []
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "moe":
                layer = MoEMLABlock(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    num_experts=self.num_experts,
                    expert_capacity_factor=self.expert_capacity_factor,
                    num_experts_per_token=self.num_experts_per_token,
                    expert_hidden_size=self.expert_hidden_size,
                    head_dim=self.head_dim,
                    num_prev_layers=self.num_prev_layers,
                    attention_window=self.attention_window,
                    layer_id=layer_idx,
                    dropout_rate=self.hidden_dropout,
                    attention_dropout=self.attention_dropout,
                    use_rope=self.use_rope,
                    max_sequence_length=self.max_sequence_length,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    deterministic=self.deterministic,
                    name=f"moe_mla_block_{layer_idx}"
                )
            else:
                layer = TransformerBlock(
                    hidden_size=self.hidden_size,
                    num_attention_heads=self.num_attention_heads,
                    intermediate_size=self.intermediate_size,
                    head_dim=self.head_dim,
                    activation=self.activation,
                    attention_dropout=self.attention_dropout,
                    hidden_dropout=self.hidden_dropout,
                    drop_path=self.drop_path,
                    use_flash_attention=self.use_flash_attention,
                    use_rope=self.use_rope,
                    max_sequence_length=self.max_sequence_length,
                    layer_norm_eps=self.layer_norm_eps,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    deterministic=self.deterministic,
                    name=f"transformer_block_{layer_idx}"
                )
            self.layers.append(layer)
            
    def _determine_layer_types(self) -> List[str]:
        """Determine the type of each layer.
        
        Returns:
            List of layer types ("moe" or "transformer")
        """
        layer_types = ["transformer"] * self.num_layers
        
        if self.num_moe_layers > 0:
            # Add MoE layers at specified frequency
            moe_positions = list(range(
                self.moe_layer_frequency - 1,
                self.num_layers,
                self.moe_layer_frequency
            ))[:self.num_moe_layers]
            
            for pos in moe_positions:
                layer_types[pos] = "moe"
                
        return layer_types
        
    def init_cache(self) -> LayerCache:
        """Initialize layer cache.
        
        Returns:
            Initialized layer cache
        """
        mla_caches = []
        for layer_idx, layer_type in enumerate(self.layer_types):
            if layer_type == "moe":
                mla_caches.append(self.layers[layer_idx].init_cache())
            else:
                mla_caches.append({})
                
        return LayerCache(
            mla_caches=mla_caches,
            layer_outputs=[],
            aux_outputs=[]
        )
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 cache: Optional[LayerCache] = None,
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: Optional[bool] = None,
                 output_hidden_states: bool = False,
                 output_attentions: bool = False) -> Tuple[jnp.ndarray, LayerCache, Dict[str, Any]]:
        """Apply transformer layers to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            cache: Optional layer cache
            attention_mask: Optional attention mask
            deterministic: Whether to run in deterministic mode
            output_hidden_states: Whether to return all layer outputs
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Output tensor
                - Updated layer cache
                - Dict of auxiliary outputs
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # Initialize cache if needed
        if cache is None:
            cache = self.init_cache()
            
        # Initialize auxiliary outputs
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        all_aux_losses = []
        
        # Process each layer
        for layer_idx, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
                
            if layer_type == "moe":
                # MoE-MLA block
                hidden_states, aux_outputs = layer(
                    hidden_states=hidden_states,
                    mla_cache=cache.mla_caches[layer_idx],
                    attention_mask=attention_mask,
                    deterministic=deterministic
                )
                
                # Update MLA cache
                cache.mla_caches[layer_idx] = aux_outputs['mla_cache']
                
                # Collect auxiliary losses
                if 'total_aux_loss' in aux_outputs:
                    all_aux_losses.append(aux_outputs['total_aux_loss'])
                    
            else:
                # Standard transformer block
                hidden_states, aux_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    deterministic=deterministic,
                    output_attentions=output_attentions
                )
                
            # Store layer outputs and auxiliary outputs
            if output_hidden_states:
                cache.layer_outputs.append(hidden_states)
            cache.aux_outputs.append(aux_outputs)
            
            if output_attentions and 'attention' in aux_outputs:
                all_attentions.append(aux_outputs['attention'])
                
        # Final layer output
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
            
        # Collect all outputs
        aux_outputs = {
            'hidden_states': all_hidden_states,
            'attentions': all_attentions,
            'aux_losses': jnp.mean(jnp.stack(all_aux_losses)) if all_aux_losses else None
        }
        
        return hidden_states, cache, aux_outputs
