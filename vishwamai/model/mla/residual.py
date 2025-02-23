"""Residual connections for Multi-Layer Attention blocks."""

from typing import Optional, Tuple, Dict, Any
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

class MLAResidual(nn.Module):
    """Gated residual connections for MLA blocks."""
    
    hidden_size: int
    dropout_rate: float = 0.1
    use_gate: bool = True
    gate_init_eps: float = 0.1
    layer_scale: bool = True
    layer_scale_init_value: float = 0.1
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    deterministic: bool = False
    
    def setup(self):
        """Initialize residual components."""
        if self.use_gate:
            # Gating mechanism
            self.gate = nn.Dense(
                2,  # Output dimension for both transform gate and cross-layer gate
                use_bias=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.zeros,
                bias_init=lambda *_: jnp.array([1.0 - self.gate_init_eps,
                                              self.gate_init_eps]),
                name="gate"
            )
            
        if self.layer_scale:
            # Learned scaling parameters
            self.layer_scale_1 = self.param(
                'layer_scale_1',
                nn.initializers.constant(self.layer_scale_init_value),
                (self.hidden_size,)
            )
            self.layer_scale_2 = self.param(
                'layer_scale_2',
                nn.initializers.constant(self.layer_scale_init_value),
                (self.hidden_size,)
            )
            
    def _apply_gate(self, 
                   x: jnp.ndarray,
                   transformed: jnp.ndarray,
                   cross_layer: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply gating mechanism to control information flow.
        
        Args:
            x: Input tensor
            transformed: Transformed input
            cross_layer: Optional cross-layer input
            
        Returns:
            Tuple of:
                - Gated output
                - Gate weights
        """
        # Global average pooling for gate computation
        pooled = jnp.mean(x, axis=1, keepdims=True)  # [batch, 1, hidden]
        
        # Compute gate values
        gate_logits = self.gate(pooled)  # [batch, 1, 2]
        gate_weights = jax.nn.softmax(gate_logits, axis=-1)
        
        # Split gates for transform and cross-layer paths
        transform_gate = gate_weights[..., 0:1]
        cross_gate = gate_weights[..., 1:2]
        
        # Apply gates
        if cross_layer is not None:
            output = (transform_gate * transformed) + (cross_gate * cross_layer)
        else:
            output = transform_gate * transformed
            
        return output, gate_weights
            
    def __call__(self,
                 x: jnp.ndarray,
                 transformed: jnp.ndarray,
                 cross_layer: Optional[jnp.ndarray] = None,
                 deterministic: Optional[bool] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply residual connection with optional gating and scaling.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_size]
            transformed: Transformed input
            cross_layer: Optional cross-layer input
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Tuple of:
                - Output tensor
                - Dict of auxiliary outputs
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        
        # Apply layer scaling if enabled
        if self.layer_scale:
            transformed = transformed * self.layer_scale_1
            if cross_layer is not None:
                cross_layer = cross_layer * self.layer_scale_2
                
        # Apply gating if enabled
        if self.use_gate:
            residual, curr_gate_weights = self._apply_gate(x, transformed, cross_layer)
        else:
            # Standard residual connection
            residual = transformed
            curr_gate_weights = None
            if cross_layer is not None:
                residual = residual + cross_layer
                
        # Apply dropout during training
        if not deterministic:
            residual = nn.Dropout(
                rate=self.dropout_rate,
                deterministic=deterministic
            )(residual, deterministic=deterministic)
            
        # Add input (final residual)
        output = x + residual
        
        # Collect auxiliary outputs
        aux_outputs = {}
        if self.use_gate:
            aux_outputs['gate_weights'] = curr_gate_weights
            
        return output, aux_outputs
        
    def init_layer_scale(self, scale_factor: float = 1.0) -> None:
        """Initialize layer scale parameters.
        
        Args:
            scale_factor: Scaling factor for initialization
        """
        if self.layer_scale:
            init_value = self.layer_scale_init_value * scale_factor
            self.layer_scale_1 = jnp.full(
                (self.hidden_size,),
                init_value,
                dtype=self.dtype
            )
            self.layer_scale_2 = jnp.full(
                (self.hidden_size,),
                init_value,
                dtype=self.dtype
            )
