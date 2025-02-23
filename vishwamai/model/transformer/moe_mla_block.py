"""Transformer block combining MoE and MLA mechanisms."""

from typing import Optional, Tuple, Dict, Any, List
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from ..moe import MoELayer
from ..mla import MLABlock

class MoEMLABlock(nn.Module):
    """Transformer block with both MoE and MLA capabilities."""
    
    # Architecture
    hidden_size: int
    num_attention_heads: int
    num_experts: int
    expert_capacity_factor: float = 1.25
    num_experts_per_token: int = 2
    expert_hidden_size: Optional[int] = None
    head_dim: Optional[int] = None
    
    # MLA Configuration
    num_prev_layers: int = 4
    attention_window: int = 4
    layer_id: int = 0
    
    # MoE Configuration
    expert_dropout: float = 0.1
    router_jitter_noise: float = 0.1
    router_dtype: Any = jnp.float32
    gate_type: str = "top_k"
    gate_temperature: float = 0.1
    gate_noise_type: str = "multiplicative"
    gate_noise_scale: float = 1.0
    
    # Common Configuration
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    use_rope: bool = True
    max_sequence_length: int = 2048
    use_gate: bool = True
    gate_init_eps: float = 0.1
    layer_scale: bool = True
    layer_scale_init_value: float = 0.1
    z_loss_scale: float = 0.01
    load_balance_scale: float = 0.01
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    deterministic: bool = False
    
    def setup(self):
        """Initialize block components."""
        # MLA components
        self.mla = MLABlock(
            hidden_size=self.hidden_size,
            num_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            num_prev_layers=self.num_prev_layers,
            attention_window=self.attention_window,
            dropout_rate=self.dropout_rate,
            attention_dropout=self.attention_dropout,
            use_rope=self.use_rope,
            max_sequence_length=self.max_sequence_length,
            layer_id=self.layer_id,
            use_gate=self.use_gate,
            gate_init_eps=self.gate_init_eps,
            layer_scale=self.layer_scale,
            layer_scale_init_value=self.layer_scale_init_value,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            deterministic=self.deterministic
        )
        
        # MoE components
        self.moe = MoELayer(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            expert_capacity_factor=self.expert_capacity_factor,
            num_experts_per_token=self.num_experts_per_token,
            expert_hidden_size=self.expert_hidden_size,
            expert_dropout=self.expert_dropout,
            router_jitter_noise=self.router_jitter_noise,
            router_dtype=self.router_dtype,
            gate_type=self.gate_type,
            gate_temperature=self.gate_temperature,
            gate_noise_type=self.gate_noise_type,
            gate_noise_scale=self.gate_noise_scale,
            z_loss_scale=self.z_loss_scale,
            load_balance_scale=self.load_balance_scale,
            deterministic=self.deterministic,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        
        # Layer normalization
        self.pre_attention_norm = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        self.pre_moe_norm = nn.LayerNorm(
            epsilon=1e-5,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 mla_cache: Dict[str, Any],
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: Optional[bool] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply MoE-MLA block to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            mla_cache: MLA layer state cache
            attention_mask: Optional attention mask
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Tuple of:
                - Output tensor
                - Dict of auxiliary outputs
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # MLA with normalization
        mla_normed = self.pre_attention_norm(hidden_states)
        mla_output, mla_aux = self.mla(
            hidden_states=mla_normed,
            cache=mla_cache,
            attention_mask=attention_mask,
            deterministic=deterministic
        )
        hidden_states = hidden_states + mla_output
        
        # MoE with normalization
        moe_normed = self.pre_moe_norm(hidden_states)
        moe_output, moe_aux = self.moe(
            hidden_states=moe_normed,
            attention_mask=attention_mask,
            deterministic=deterministic
        )
        hidden_states = hidden_states + moe_output
        
        # Combine auxiliary outputs
        aux_outputs = {
            'mla': mla_aux,
            'moe': moe_aux,
            'mla_cache': mla_aux['cache'],
            'total_aux_loss': (
                mla_aux.get('aux_loss', 0.0) +
                moe_aux.get('aux_loss', 0.0)
            )
        }
        
        return hidden_states, aux_outputs
        
    def init_cache(self) -> Dict[str, Any]:
        """Initialize empty MLA cache.
        
        Returns:
            Empty cache dictionary
        """
        return self.mla.init_cache()
        
    def clear_cache(self, cache: Dict[str, Any]) -> Dict[str, Any]:
        """Clear the MLA cache.
        
        Args:
            cache: Cache to clear
            
        Returns:
            Empty cache dictionary
        """
        return self.mla.clear_cache(cache)
