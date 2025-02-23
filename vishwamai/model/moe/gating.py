"""Gating mechanisms for MoE and MLA layers."""

from typing import Optional, Tuple, Dict, Any, Callable
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .router import compute_router_z_loss

def compute_gate_entropy_loss(gate_logits: jnp.ndarray) -> jnp.ndarray:
    """Compute entropy loss to encourage diverse gate usage.
    
    Args:
        gate_logits: Raw gate logits [batch, seq_len, num_gates]
        
    Returns:
        Entropy loss value
    """
    gate_probs = jax.nn.softmax(gate_logits, axis=-1)
    entropy = -jnp.sum(gate_probs * jnp.log(gate_probs + 1e-9), axis=-1)
    target_entropy = jnp.log(gate_probs.shape[-1])  # Maximum possible entropy
    return jnp.mean(jnp.square(entropy - target_entropy))

def compute_cv_loss(gate_weights: jnp.ndarray) -> jnp.ndarray:
    """Compute coefficient of variation loss for balanced gate usage.
    
    Args:
        gate_weights: Gate weight values [batch, seq_len, num_gates]
        
    Returns:
        CV loss value
    """
    # Calculate mean gate usage
    mean_gates = jnp.mean(gate_weights, axis=(0, 1))
    
    # Calculate coefficient of variation (std/mean)
    std_gates = jnp.std(gate_weights, axis=(0, 1))
    cv = std_gates / (mean_gates + 1e-9)
    
    return jnp.mean(cv)

class GatingMechanism(nn.Module):
    """Gating mechanism with multiple gating strategies."""
    
    hidden_size: int
    num_gates: int
    gate_type: str = "top_k"  # Options: top_k, multiplicative
    gate_temperature: float = 0.1
    noise_type: str = "multiplicative"  # Options: multiplicative, additive
    noise_scale: float = 1.0
    z_loss_scale: float = 0.01
    deterministic: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    def setup(self):
        """Initialize gating components."""
        if self.gate_type == "top_k":
            # For top-k gating, project to logits
            self.gate_proj = nn.Dense(
                self.num_gates,
                use_bias=True,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                kernel_init=nn.initializers.normal(stddev=0.02),
                bias_init=nn.initializers.zeros,
                name="gate_proj"
            )
        else:
            # For multiplicative gating, project to gates directly
            self.gate_proj = nn.Sequential([
                nn.Dense(
                    self.hidden_size,
                    use_bias=True,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    kernel_init=nn.initializers.normal(stddev=0.02),
                    name="gate_proj_1"
                ),
                nn.gelu,
                nn.Dense(
                    self.num_gates,
                    use_bias=True,
                    dtype=self.dtype,
                    param_dtype=self.param_dtype,
                    kernel_init=nn.initializers.normal(stddev=0.02),
                    name="gate_proj_2"
                )
            ])
            
    def _add_noise(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Add noise to logits during training.
        
        Args:
            logits: Input logits
            
        Returns:
            Logits with added noise
        """
        if self.deterministic or self.noise_scale == 0:
            return logits
            
        if self.noise_type == "multiplicative":
            noise = jax.random.uniform(
                self.make_rng('noise'),
                logits.shape,
                minval=1.0 - self.noise_scale,
                maxval=1.0 + self.noise_scale
            )
            return logits * noise
        else:  # additive
            noise = jax.random.normal(
                self.make_rng('noise'),
                logits.shape
            ) * self.noise_scale
            return logits + noise
            
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 compute_aux_loss: bool = True) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Apply gating mechanism.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            compute_aux_loss: Whether to compute auxiliary losses
            
        Returns:
            Tuple of:
                - Gate weights [batch, seq_len, num_gates]
                - Dict of auxiliary outputs including gate logits and losses
        """
        # Project inputs to gate logits
        gate_logits = self.gate_proj(hidden_states)
        
        # Add noise during training
        gate_logits = self._add_noise(gate_logits)
        
        if self.gate_type == "top_k":
            # Apply temperature scaling
            gate_logits = gate_logits / self.gate_temperature
            
            # Get top-k gate weights
            gate_probs = jax.nn.softmax(gate_logits, axis=-1)
            top_probs, top_indices = jax.lax.top_k(gate_probs, k=2)
            
            # Create sparse gate weights
            gate_weights = jnp.zeros_like(gate_probs)
            gate_weights = gate_weights.at[
                jnp.arange(gate_probs.shape[0])[:, None, None],
                jnp.arange(gate_probs.shape[1])[None, :, None],
                top_indices
            ].set(top_probs)
            
        else:  # multiplicative
            # Apply sigmoid for multiplicative gating
            gate_weights = jax.nn.sigmoid(gate_logits)
            
        aux_outputs = {'gate_logits': gate_logits}
        
        if compute_aux_loss:
            # Compute auxiliary losses
            aux_outputs['gate_entropy_loss'] = compute_gate_entropy_loss(gate_logits)
            aux_outputs['gate_cv_loss'] = compute_cv_loss(gate_weights)
            
            if self.gate_type == "top_k":
                aux_outputs['gate_z_loss'] = compute_router_z_loss(gate_logits) * self.z_loss_scale
                
        return gate_weights, aux_outputs
        
    def get_gate_statistics(self, gate_weights: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Compute statistics about gate usage.
        
        Args:
            gate_weights: Gate weight values [batch, seq_len, num_gates]
            
        Returns:
            Dictionary of gate usage statistics
        """
        # Compute mean gate usage
        mean_gates = jnp.mean(gate_weights, axis=(0, 1))
        
        # Compute gate standard deviation
        std_gates = jnp.std(gate_weights, axis=(0, 1))
        
        # Compute gate entropy
        gate_probs = jnp.mean(gate_weights, axis=(0, 1))
        entropy = -jnp.sum(gate_probs * jnp.log(gate_probs + 1e-9))
        
        return {
            'mean_gate_values': mean_gates,
            'std_gate_values': std_gates,
            'gate_entropy': entropy
        }
