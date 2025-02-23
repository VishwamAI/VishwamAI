"""VishwamAI transformer model implementation."""

from typing import Optional, Tuple, Dict, Any, List, Union
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .config import TransformerConfig
from .layer import TransformerLayer, LayerCache
from ..embeddings import TokenEmbedding, PositionalEncoding

class VishwamAIModel(nn.Module):
    """VishwamAI transformer model with MoE and MLA."""
    
    config: TransformerConfig
    
    def setup(self):
        """Initialize model components."""
        # Token embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=self.config.extra_params.get('vocab_size', 32000),
            hidden_size=self.config.hidden_size,
            padding_idx=self.config.extra_params.get('pad_token_id', None),
            factorized=self.config.extra_params.get('factorized_embeddings', False),
            factorized_dim=self.config.extra_params.get('factorized_dim', None),
            embedding_std=self.config.initializer_range,
            layer_norm_eps=self.config.layer_norm_eps,
            dropout=self.config.hidden_dropout,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype
        )
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(
            max_seq_length=self.config.max_sequence_length,
            hidden_size=self.config.hidden_size,
            dropout_rate=self.config.hidden_dropout,
            scale=self.config.extra_params.get('position_embedding_scale', 1.0),
            deterministic=self.config.extra_params.get('deterministic', False),
            dtype=self.config.dtype
        )
        
        # Transformer layers
        self.layers = TransformerLayer(
            hidden_size=self.config.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            num_layers=self.config.num_hidden_layers,
            intermediate_size=self.config.intermediate_size,
            head_dim=self.config.head_dim,
            num_moe_layers=self.config.num_moe_layers,
            moe_layer_frequency=self.config.moe_layer_frequency,
            num_experts=self.config.num_experts,
            expert_capacity_factor=self.config.expert_capacity_factor,
            num_experts_per_token=self.config.num_experts_per_token,
            expert_hidden_size=self.config.expert_hidden_size,
            num_prev_layers=self.config.num_prev_layers,
            attention_window=self.config.attention_window,
            activation=self.config.activation,
            attention_dropout=self.config.attention_dropout,
            hidden_dropout=self.config.hidden_dropout,
            drop_path=self.config.drop_path,
            use_flash_attention=self.config.use_flash_attention,
            use_rope=self.config.use_rope,
            max_sequence_length=self.config.max_sequence_length,
            layer_norm_eps=self.config.layer_norm_eps,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype,
            deterministic=self.config.extra_params.get('deterministic', False)
        )
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps,
            dtype=self.config.dtype,
            param_dtype=self.config.param_dtype
        )
        
    def __call__(self,
                 input_ids: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 position_ids: Optional[jnp.ndarray] = None,
                 layer_cache: Optional[LayerCache] = None,
                 deterministic: Optional[bool] = None,
                 output_hidden_states: bool = False,
                 output_attentions: bool = False,
                 output_aux_losses: bool = True) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            layer_cache: Optional layer state cache
            deterministic: Whether to run in deterministic mode
            output_hidden_states: Whether to return all layer hidden states
            output_attentions: Whether to return attention weights
            output_aux_losses: Whether to return auxiliary losses
            
        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_size]
                - Dict of auxiliary outputs
        """
        deterministic = (
            deterministic if deterministic is not None
            else self.config.extra_params.get('deterministic', False)
        )
        
        # Get embeddings
        hidden_states = self.token_embedding(
            input_ids,
            deterministic=deterministic
        )
        
        # Add positional encodings
        if position_ids is None:
            position_ids = jnp.arange(input_ids.shape[1])[None, :]
            
        hidden_states = self.position_encoding(
            hidden_states,
            deterministic=deterministic
        )
        
        # Process layers
        layer_outputs, layer_cache, layer_aux = self.layers(
            hidden_states=hidden_states,
            cache=layer_cache,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions
        )
        
        # Final layer norm
        hidden_states = self.final_norm(layer_outputs)
        
        # Collect outputs
        outputs = {
            'last_hidden_state': hidden_states,
            'layer_cache': layer_cache
        }
        
        if output_hidden_states:
            outputs['hidden_states'] = layer_aux['hidden_states']
            
        if output_attentions:
            outputs['attentions'] = layer_aux['attentions']
            
        if output_aux_losses and 'aux_losses' in layer_aux:
            outputs['aux_losses'] = layer_aux['aux_losses']
            
        return hidden_states, outputs
        
    def init_cache(self) -> LayerCache:
        """Initialize empty layer cache.
        
        Returns:
            Empty layer cache
        """
        return self.layers.init_cache()
        
    @property
    def num_parameters(self) -> int:
        """Get total number of model parameters.
        
        Returns:
            Number of parameters
        """
        return self.config.num_parameters
