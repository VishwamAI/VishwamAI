"""TPU-optimized neural network layers with conditional computation"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Optional, Tuple, Dict, List, Callable
from vishwamai.flash_attention import FlashAttention, flash_attention_inference
from vishwamai.kernels.kernel import fp8_gemm_optimized, act_quant, optimize_kernel_layout

class TPUGEMMLinear(nn.Module):
    """Linear layer with TPU-optimized GEMM operations."""
    features: int
    use_bias: bool = True
    dtype: Any = jnp.float32
    precision: Any = None
    kernel_init: Any = nn.initializers.lecun_normal()
    bias_init: Any = nn.initializers.zeros
    use_fp8: bool = True
    block_size: int = 128
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (x.shape[-1], self.features),
            self.dtype
        )
        
        if self.use_fp8:
            # Use FP8 GEMM for faster computation
            x_quant, x_scale = act_quant(x, block_size=self.block_size)
            kernel_quant, kernel_scale = act_quant(kernel, block_size=self.block_size)
            
            y = fp8_gemm_optimized(
                x_quant,
                x_scale,
                kernel_quant,
                kernel_scale,
                block_size=self.block_size
            )
        else:
            kernel = optimize_kernel_layout(kernel)
            y = jnp.dot(x, kernel, precision=self.precision)
            
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,), self.dtype)
            y = y + bias
            
        return y

class TPULayerNorm(nn.Module):
    """TPU-optimized Layer Normalization."""
    epsilon: float = 1e-6
    dtype: Any = jnp.float32
    scale_init: Any = nn.initializers.ones
    bias_init: Any = nn.initializers.zeros
    use_bias: bool = True
    use_scale: bool = True
    axis: int = -1
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        feature_shape = (x.shape[self.axis],)
        mean = jnp.mean(x, axis=self.axis, keepdims=True)
        var = jnp.var(x, axis=self.axis, keepdims=True)
        
        # Compute normalization in FP32 for stability
        y = (x - mean) / jnp.sqrt(var + self.epsilon)
        
        if self.use_scale:
            scale = self.param('scale', self.scale_init, feature_shape, self.dtype)
            y = y * scale
            
        if self.use_bias:
            bias = self.param('bias', self.bias_init, feature_shape, self.dtype)
            y = y + bias
            
        return y.astype(self.dtype)

class TPUMultiHeadAttention(nn.Module):
    """TPU-optimized Multi-Head Attention with Flash Attention."""
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    qkv_bias: bool = True
    use_flash_attn: bool = True
    use_fp8: bool = True
    block_size: int = 128
    
    def setup(self):
        # Projection layers
        self.qkv = TPUGEMMLinear(
            features=3 * self.num_heads * self.head_dim,
            use_bias=self.qkv_bias,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )
        
        self.out = TPUGEMMLinear(
            features=self.num_heads * self.head_dim,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )
        
        # Flash Attention module
        if self.use_flash_attn:
            self.flash_attn = FlashAttention(
                block_size=self.block_size,
                use_fp8=self.use_fp8,
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )
    
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
        past_key_value: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, jnp.ndarray]]]:
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # [3, batch, heads, seq, dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Add past key/values for inference
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = jnp.concatenate([past_k, k], axis=2)
            v = jnp.concatenate([past_v, v], axis=2)
        
        # Apply attention
        if self.use_flash_attn:
            if deterministic:
                attn_output, present = flash_attention_inference(
                    q, k, v,
                    mask=mask,
                    past_key_values=past_key_value,
                    block_size=self.block_size,
                    head_dim=self.head_dim,
                    num_heads=self.num_heads,
                    use_fp8=self.use_fp8
                )
            else:
                attn_output = self.flash_attn(
                    q, k, v,
                    mask=mask,
                    deterministic=deterministic
                )
                present = (k, v) if past_key_value is not None else None
        else:
            # Standard scaled dot-product attention
            scale = 1.0 / jnp.sqrt(self.head_dim)
            attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
            
            if mask is not None:
                attn_weights = jnp.where(mask, attn_weights, jnp.finfo(self.dtype).min)
            
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            
            if not deterministic and self.dropout_rate > 0.0:
                attn_weights = nn.Dropout(
                    rate=self.dropout_rate,
                    deterministic=deterministic
                )(attn_weights)
            
            attn_output = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
            present = (k, v) if past_key_value is not None else None
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = self.out(attn_output)
        
        return attn_output, present

class TPUMoELayer(nn.Module):
    """TPU-optimized Mixture of Experts layer."""
    num_experts: int
    expert_dim: int
    capacity_factor: float = 1.0
    dropout_rate: float = 0.0
    dtype: Any = jnp.float32
    use_fp8: bool = True
    block_size: int = 128
    
    def setup(self):
        # Router components
        self.router = TPUGEMMLinear(
            features=self.num_experts,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )
        
        # Expert feed-forward networks
        self.experts = [
            TPUGEMMLinear(
                features=self.expert_dim,
                dtype=self.dtype,
                use_fp8=self.use_fp8,
                block_size=self.block_size
            )
            for _ in range(self.num_experts)
        ]
        
        self.output = TPUGEMMLinear(
            features=self.expert_dim,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )
    
    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        batch_size, seq_len, hidden_dim = x.shape
        
        # Get router scores
        router_logits = self.router(x)
        router_probs = jax.nn.softmax(router_logits, axis=-1)
        
        # Select top-k experts (k=2 by default)
        k = 2
        top_k_probs, top_k_indices = jax.lax.top_k(router_probs, k)
        top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)
        
        # Compute capacity
        capacity = int(self.capacity_factor * batch_size * seq_len / self.num_experts)
        
        # Dispatch to experts
        expert_inputs = []
        expert_mask = []
        
        for expert_idx in range(self.num_experts):
            # Get inputs routed to this expert
            expert_mask_idx = (top_k_indices == expert_idx)
            expert_mask.append(expert_mask_idx)
            
            if jnp.any(expert_mask_idx):
                # Process expert inputs
                expert_input = jnp.where(
                    expert_mask_idx[..., None],
                    x,
                    0.0
                )
                expert_inputs.append(expert_input)
        
        # Process each expert in parallel
        expert_outputs = []
        for expert_idx, (expert_input, mask) in enumerate(zip(expert_inputs, expert_mask)):
            if jnp.any(mask):
                # Apply expert network
                expert_output = self.experts[expert_idx](expert_input)
                expert_output = jax.nn.gelu(expert_output)
                
                if not deterministic:
                    expert_output = nn.Dropout(rate=self.dropout_rate)(
                        expert_output,
                        deterministic=False
                    )
                
                expert_outputs.append(expert_output)
            else:
                expert_outputs.append(jnp.zeros_like(x))
        
        # Combine expert outputs
        combined_output = jnp.zeros_like(x)
        for expert_output, probs in zip(expert_outputs, top_k_probs.transpose(2, 0, 1)):
            combined_output += expert_output * probs[..., None]
        
        # Final output projection
        output = self.output(combined_output)
        
        if not deterministic:
            output = nn.Dropout(rate=self.dropout_rate)(
                output,
                deterministic=False
            )
        
        return output

class DynamicChannelGating(nn.Module):
    """Dynamic channel gating layer for conditional computation."""
    channels: int
    hidden_dim: int = 256
    temperature: float = 1.0
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Gate generation network
        gate_net = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype
        )(x)
        gate_net = jax.nn.gelu(gate_net)
        gate_logits = TPUGEMMLinear(
            features=self.channels,
            dtype=self.dtype
        )(gate_net)
        
        # Apply temperature scaling and gumbel softmax for training
        if training:
            gates = jax.nn.gumbel_softmax(
                gate_logits,
                temperature=self.temperature,
                axis=-1
            )
        else:
            gates = jax.nn.softmax(gate_logits / self.temperature, axis=-1)
            
        # Apply gates to channels
        gated_output = x * gates
        
        return gated_output, gates

class ConditionalInfoGainNode(nn.Module):
    """Node for Conditional Information Gain Networks."""
    hidden_dim: int
    output_dim: int
    num_splits: int = 2
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Split function network
        split_net = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype
        )(x)
        split_net = jax.nn.gelu(split_net)
        
        # Generate split probabilities
        split_logits = TPUGEMMLinear(
            features=self.num_splits,
            dtype=self.dtype
        )(split_net)
        
        # Compute split probabilities
        if training:
            split_probs = jax.nn.gumbel_softmax(split_logits, axis=-1)
        else:
            split_probs = jax.nn.softmax(split_logits, axis=-1)
            
        # Transform input based on split
        transforms = []
        for i in range(self.num_splits):
            transform = TPUGEMMLinear(
                features=self.output_dim,
                dtype=self.dtype
            )(x)
            transforms.append(transform)
            
        transforms = jnp.stack(transforms, axis=1)
        output = jnp.sum(transforms * split_probs[..., None], axis=1)
        
        return output, split_probs

class CIGTLayer(nn.Module):
    """Conditional Information Gain Trellis layer."""
    features: int
    num_paths: int = 4
    info_threshold: float = 0.1
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self, 
        x: jnp.ndarray,
        prev_info: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        batch_size = x.shape[0]
        
        # Information gain computation network
        info_net = TPUGEMMLinear(
            features=self.num_paths,
            dtype=self.dtype
        )(x)
        info_gains = jax.nn.sigmoid(info_net)
        
        # Update cumulative information
        if prev_info is None:
            prev_info = jnp.zeros((batch_size, self.num_paths), dtype=self.dtype)
        cumul_info = prev_info + info_gains
        
        # Path selection based on information gain
        active_paths = cumul_info > self.info_threshold
        if training:
            # Use soft masks during training
            path_weights = jax.nn.sigmoid((cumul_info - self.info_threshold) * 10.0)
        else:
            path_weights = active_paths.astype(self.dtype)
            
        # Process through parallel paths
        path_outputs = []
        for i in range(self.num_paths):
            path_transform = TPUGEMMLinear(
                features=self.features // self.num_paths,
                dtype=self.dtype
            )(x)
            path_outputs.append(path_transform)
            
        path_outputs = jnp.concatenate(path_outputs, axis=-1)
        output = path_outputs * path_weights[..., None]
        
        return output, cumul_info

class RLBasedConditionalLayer(nn.Module):
    """Reinforcement learning based conditional computation layer."""
    features: int
    num_experts: int = 8
    temperature: float = 1.0
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        prev_rewards: Optional[jnp.ndarray] = None,
        training: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # Policy network
        policy_net = TPUGEMMLinear(
            features=256,
            dtype=self.dtype
        )(x)
        policy_net = jax.nn.gelu(policy_net)
        action_logits = TPUGEMMLinear(
            features=self.num_experts,
            dtype=self.dtype
        )(policy_net)
        
        # Action selection
        if training:
            # Use Gumbel-Softmax for differentiable sampling
            actions = jax.nn.gumbel_softmax(
                action_logits,
                temperature=self.temperature,
                axis=-1
            )
        else:
            # Greedy selection during inference
            actions = jax.nn.one_hot(
                jnp.argmax(action_logits, axis=-1),
                self.num_experts
            )
            
        # Expert computation
        experts = []
        for i in range(self.num_experts):
            expert = TPUGEMMLinear(
                features=self.features,
                dtype=self.dtype
            )(x)
            experts.append(expert)
            
        experts = jnp.stack(experts, axis=1)
        output = jnp.sum(experts * actions[..., None], axis=1)
        
        # Compute auxiliary outputs for training
        aux = {
            'actions': actions,
            'action_logits': action_logits
        }
        
        if prev_rewards is not None:
            # Update policy based on previous rewards
            policy_loss = -jnp.mean(
                prev_rewards * jnp.log(actions + 1e-10)
            )
            aux['policy_loss'] = policy_loss
            
        return output, aux

class MLABlock(nn.Module):
    """Multi-Level Attention block with TPU optimizations."""
    num_heads: int
    head_dim: Optional[int] = None
    qkv_dim: Optional[int] = None
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None
    dtype: Any = jnp.float32
    use_fp8: bool = True
    block_size: int = 128

    def setup(self):
        # Determine dimensions
        self.qkv_dim = self.qkv_dim or self.num_heads * (self.head_dim or 64)
        self.head_dim = self.head_dim or self.qkv_dim // self.num_heads

        # QKV projections
        self.qkv = TPUGEMMLinear(
            features=3 * self.qkv_dim,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )

        # Output projection
        self.out = TPUGEMMLinear(
            features=self.qkv_dim,
            dtype=self.dtype,
            use_fp8=self.use_fp8,
            block_size=self.block_size
        )

        # Flash Attention for efficient computation
        self.flash_attention = FlashAttention(
            block_size=self.block_size,
            use_fp8=self.use_fp8,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )

    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None, deterministic: Optional[bool] = None) -> jnp.ndarray:
        """
        Apply multi-level attention to input.
        Args:
            x: Input of shape [batch, sequence, hidden_dim]
            mask: Optional attention mask
            deterministic: Whether to run in deterministic mode (no dropout)
        Returns:
            Output of shape [batch, sequence, hidden_dim]
        """
        deterministic = deterministic if deterministic is not None else self.deterministic
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv(x)
        qkv = jnp.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        q, k, v = jnp.split(qkv, indices_or_sections=3, axis=2)
        q = q.squeeze(2)
        k = k.squeeze(2)
        v = v.squeeze(2)

        # Apply Flash Attention
        x = self.flash_attention(
            q=q,
            k=k,
            v=v,
            mask=mask,
            deterministic=deterministic
        )

        # Project output
        x = jnp.reshape(x, (batch_size, seq_len, self.qkv_dim))
        x = self.out(x)

        if not deterministic:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)

        return x

def create_layer_factory(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create layer factory with TPU-optimized configurations."""
    return {
        "linear": lambda features: TPUGEMMLinear(
            features=features,
            dtype=config["model"].get("dtype", jnp.float32),
            use_fp8=config["optimization"].get("use_fp8", True),
            block_size=config["optimization"].get("block_size", 128)
        ),
        "layer_norm": lambda: TPULayerNorm(
            dtype=config["model"].get("dtype", jnp.float32)
        ),
        "attention": lambda: TPUMultiHeadAttention(
            num_heads=config["model"]["num_heads"],
            head_dim=config["model"]["head_dim"],
            dropout_rate=config["model"].get("attention_dropout_rate", 0.0),
            dtype=config["model"].get("dtype", jnp.float32),
            use_flash_attn=config["optimization"].get("use_flash_attention", True),
            use_fp8=config["optimization"].get("use_fp8", True),
            block_size=config["optimization"].get("block_size", 128)
        ),
        "moe": lambda: TPUMoELayer(
            num_experts=config["model"].get("num_experts", 8),
            expert_dim=config["model"]["hidden_dim"],
            dropout_rate=config["model"].get("dropout_rate", 0.0),
            dtype=config["model"].get("dtype", jnp.float32),
            use_fp8=config["optimization"].get("use_fp8", True),
            block_size=config["optimization"].get("block_size", 128)
        )
    }

def create_conditional_layer_factory(
    config: Dict[str, Any]
) -> Dict[str, Callable]:
    """Create factory for conditional computation layers."""
    return {
        "dynamic_gating": lambda: DynamicChannelGating(
            channels=config["model"]["hidden_dim"],
            dtype=config["model"].get("dtype", jnp.float32)
        ),
        "cign": lambda: ConditionalInfoGainNode(
            hidden_dim=config["model"]["hidden_dim"],
            output_dim=config["model"]["hidden_dim"],
            dtype=config["model"].get("dtype", jnp.float32)
        ),
        "cigt": lambda: CIGTLayer(
            features=config["model"]["hidden_dim"],
            dtype=config["model"].get("dtype", jnp.float32)
        ),
        "rl_conditional": lambda: RLBasedConditionalLayer(
            features=config["model"]["hidden_dim"],
            dtype=config["model"].get("dtype", jnp.float32)
        )
    }

# Alias TPUMoELayer as MoELayer for compatibility
MoELayer = TPUMoELayer

__all__ = [
    'TPUGEMMLinear',
    'TPULayerNorm', 
    'TPUMultiHeadAttention',
    'TPUMoELayer',
    'MoELayer',
    'DynamicChannelGating',
    'ConditionalInfoGainNode',
    'CIGTLayer',
    'RLBasedConditionalLayer'
]
