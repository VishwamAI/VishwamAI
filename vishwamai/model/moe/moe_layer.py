"""Mixture of Experts layer combining routing and expert computation."""

from typing import Optional, Tuple, Dict, Any, List
import math

import jax
import jax.numpy as jnp
from flax import linen as nn

from .expert import ExpertLayer
from .router import ExpertRouter
from .gating import GatingMechanism

class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing."""
    
    hidden_size: int
    num_experts: int
    expert_capacity_factor: float = 1.25
    num_experts_per_token: int = 2
    expert_hidden_size: Optional[int] = None
    expert_activation: str = "swiglu"
    expert_dropout: float = 0.1
    router_jitter_noise: float = 0.1
    router_dtype: Any = jnp.float32
    gate_type: str = "top_k"
    gate_temperature: float = 0.1
    gate_noise_type: str = "multiplicative"
    gate_noise_scale: float = 1.0
    z_loss_scale: float = 0.01
    load_balance_scale: float = 0.01
    deterministic: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    
    def setup(self):
        """Initialize MoE components."""
        # Create experts
        self.experts = [
            ExpertLayer(
                hidden_size=self.hidden_size,
                intermediate_size=self.expert_hidden_size,
                activation=self.expert_activation,
                dropout_rate=self.expert_dropout,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                deterministic=self.deterministic,
                name=f"expert_{i}"
            )
            for i in range(self.num_experts)
        ]
        
        # Create router
        self.router = ExpertRouter(
            num_experts=self.num_experts,
            capacity_factor=self.expert_capacity_factor,
            num_experts_per_token=self.num_experts_per_token,
            jitter_noise=self.router_jitter_noise,
            dtype=self.router_dtype,
            param_dtype=self.param_dtype
        )
        
        # Create gating mechanism
        self.gate = GatingMechanism(
            hidden_size=self.hidden_size,
            num_gates=self.num_experts,
            gate_type=self.gate_type,
            gate_temperature=self.gate_temperature,
            noise_type=self.gate_noise_type,
            noise_scale=self.gate_noise_scale,
            z_loss_scale=self.z_loss_scale,
            deterministic=self.deterministic,
            dtype=self.dtype,
            param_dtype=self.param_dtype
        )
        
    def _process_expert_outputs(self,
                             expert_outputs: List[jnp.ndarray],
                             router_weights: jnp.ndarray,
                             gate_weights: jnp.ndarray) -> jnp.ndarray:
        """Combine expert outputs using routing and gating weights.
        
        Args:
            expert_outputs: List of expert outputs [num_experts, batch, seq, hidden]
            router_weights: Router weights [batch, seq, num_experts]
            gate_weights: Gate weights [batch, seq, num_experts]
            
        Returns:
            Combined output tensor
        """
        # Stack expert outputs
        stacked_outputs = jnp.stack(expert_outputs)  # [num_experts, batch, seq, hidden]
        
        # Combine router and gate weights
        combined_weights = router_weights * gate_weights
        combined_weights = combined_weights / (jnp.sum(combined_weights, axis=-1, keepdims=True) + 1e-9)
        
        # Transpose for broadcasting
        combined_weights = jnp.transpose(combined_weights, (2, 0, 1))  # [num_experts, batch, seq]
        combined_weights = combined_weights[..., None]  # Add hidden dim
        
        # Weighted combination of expert outputs
        combined_output = jnp.sum(stacked_outputs * combined_weights, axis=0)
        
        return combined_output
        
    def __call__(self,
                 hidden_states: jnp.ndarray,
                 attention_mask: Optional[jnp.ndarray] = None,
                 deterministic: Optional[bool] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Apply MoE layer to input.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            deterministic: Whether to run in deterministic mode
            
        Returns:
            Tuple of:
                - Output tensor [batch, seq_len, hidden_size]
                - Dict of auxiliary outputs
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        batch_size, seq_length, hidden_size = hidden_states.shape
        
        # Get router weights and auxiliary outputs
        router_weights, router_aux = self.router(
            hidden_states,
            deterministic=deterministic
        )
        
        # Get gate weights and auxiliary outputs
        gate_weights, gate_aux = self.gate(
            hidden_states,
            compute_aux_loss=True
        )
        
        # Initialize expert outputs list
        expert_outputs = []
        
        # Process each expert
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens routed to this expert
            expert_weights = router_weights[..., expert_idx, :]  # [batch, seq, slots]
            
            # Compute expert output for all tokens
            expert_output = expert(
                hidden_states,
                deterministic=deterministic
            )
            expert_outputs.append(expert_output)
            
        # Combine expert outputs
        output = self._process_expert_outputs(
            expert_outputs,
            router_weights=router_aux['router_probs'],
            gate_weights=gate_weights
        )
        
        # Compute total auxiliary loss
        aux_loss = (
            router_aux['router_z_loss'] * self.z_loss_scale +
            router_aux['load_balancing_loss'] * self.load_balance_scale +
            gate_aux['gate_entropy_loss'] +
            gate_aux['gate_cv_loss']
        )
        
        # Combine auxiliary outputs
        aux_outputs = {
            'router_probs': router_aux['router_probs'],
            'expert_mask': router_aux['expert_mask'],
            'gate_weights': gate_weights,
            'aux_loss': aux_loss,
            **{f'expert_{i}_norm': jnp.linalg.norm(expert_outputs[i])
               for i in range(self.num_experts)}
        }
        
        return output, aux_outputs
        
    def init_experts(self, rng: jax.random.PRNGKey,
                   scales: Optional[List[float]] = None) -> None:
        """Initialize expert weights with optional per-expert scaling.
        
        Args:
            rng: PRNG key
            scales: Optional list of scaling factors per expert
        """
        if scales is None:
            scales = [1.0] * self.num_experts
            
        for expert_idx, (expert, scale) in enumerate(zip(self.experts, scales)):
            expert_rng = jax.random.fold_in(rng, expert_idx)
            expert.init_expert(expert_rng, scale=scale)
