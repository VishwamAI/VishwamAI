"""TPU kernel fusion patterns for common operation sequences."""

from typing import List, Tuple, Dict, Any, Optional, Callable
import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass

@dataclass
class FusionPattern:
    """Pattern definition for kernel fusion."""
    name: str
    operations: List[str]
    compute_cost: float
    memory_cost: int
    constraints: Dict[str, Any]

class TPUFusionPatternManager:
    """Manages TPU kernel fusion patterns."""
    
    def __init__(self):
        self.patterns: Dict[str, FusionPattern] = {}
        self._register_default_patterns()
        
    def _register_default_patterns(self):
        """Register common fusion patterns."""
        # Pattern 1: Attention + Dropout + LayerNorm
        self.register_pattern(
            FusionPattern(
                name="attention_dropout_norm",
                operations=["attention", "dropout", "layer_norm"],
                compute_cost=1.0,
                memory_cost=0,  # Computed at runtime
                constraints={
                    "max_sequence_length": 2048,
                    "min_batch_size": 1
                }
            )
        )
        
        # Pattern 2: Linear + GELU + Dropout
        self.register_pattern(
            FusionPattern(
                name="linear_gelu_dropout",
                operations=["linear", "gelu", "dropout"],
                compute_cost=0.8,
                memory_cost=0,
                constraints={
                    "min_hidden_dim": 128,
                    "max_hidden_dim": 8192
                }
            )
        )
        
        # Pattern 3: QKV Projection + Split
        self.register_pattern(
            FusionPattern(
                name="qkv_projection_split",
                operations=["linear", "reshape", "split"],
                compute_cost=0.9,
                memory_cost=0,
                constraints={
                    "head_dim_multiple": 64,
                    "max_heads": 128
                }
            )
        )
        
    def register_pattern(self, pattern: FusionPattern) -> None:
        """Register a new fusion pattern."""
        self.patterns[pattern.name] = pattern
        
    def get_pattern(self, name: str) -> Optional[FusionPattern]:
        """Get registered pattern by name."""
        return self.patterns.get(name)
        
    def create_fused_kernel(
        self,
        pattern_name: str,
        *args: Any,
        **kwargs: Any
    ) -> Callable:
        """Create a fused kernel implementation for a pattern."""
        pattern = self.get_pattern(pattern_name)
        if pattern is None:
            raise ValueError(f"Unknown pattern: {pattern_name}")
            
        # Validate constraints
        self._validate_constraints(pattern, kwargs)
        
        # Create fused implementation
        if pattern_name == "attention_dropout_norm":
            return self._create_attention_dropout_norm(*args, **kwargs)
        elif pattern_name == "linear_gelu_dropout":
            return self._create_linear_gelu_dropout(*args, **kwargs)
        elif pattern_name == "qkv_projection_split":
            return self._create_qkv_projection_split(*args, **kwargs)
        else:
            raise ValueError(f"No implementation for pattern: {pattern_name}")
            
    def _validate_constraints(
        self,
        pattern: FusionPattern,
        params: Dict[str, Any]
    ) -> None:
        """Validate pattern constraints against parameters."""
        for key, constraint in pattern.constraints.items():
            if key not in params:
                continue
                
            value = params[key]
            if isinstance(constraint, (int, float)):
                if value > constraint:
                    raise ValueError(
                        f"Parameter {key}={value} exceeds max value {constraint}"
                    )
            elif isinstance(constraint, dict):
                if "min" in constraint and value < constraint["min"]:
                    raise ValueError(
                        f"Parameter {key}={value} below min value {constraint['min']}"
                    )
                if "max" in constraint and value > constraint["max"]:
                    raise ValueError(
                        f"Parameter {key}={value} exceeds max value {constraint['max']}"
                    )
                    
    def _create_attention_dropout_norm(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        eps: float = 1e-6
    ) -> Callable:
        """Create fused attention + dropout + layer norm kernel."""
        
        @partial(jax.jit, static_argnames=("training",))
        def fused_kernel(
            q: jnp.ndarray,
            k: jnp.ndarray,
            v: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
            training: bool = True,
            rng: Optional[Any] = None
        ) -> jnp.ndarray:
            # Scaled dot-product attention
            scale = 1.0 / jnp.sqrt(hidden_size // num_heads)
            scores = jnp.matmul(q, k.transpose(-2, -1)) * scale
            
            if mask is not None:
                scores = scores + mask
                
            attn_weights = jax.nn.softmax(scores, axis=-1)
            
            if training and dropout_rate > 0:
                if rng is None:
                    rng = jax.random.PRNGKey(0)
                attn_weights = jax.random.dropout(
                    rng,
                    dropout_rate,
                    attn_weights
                )
                
            # Attention output
            attn_output = jnp.matmul(attn_weights, v)
            
            # Layer norm
            mean = jnp.mean(attn_output, axis=-1, keepdims=True)
            variance = jnp.mean(
                jnp.square(attn_output - mean),
                axis=-1,
                keepdims=True
            )
            normed = (attn_output - mean) * jax.lax.rsqrt(variance + eps)
            
            return normed
            
        return fused_kernel
        
    def _create_linear_gelu_dropout(
        self,
        in_features: int,
        out_features: int,
        dropout_rate: float = 0.1
    ) -> Callable:
        """Create fused linear + GELU + dropout kernel."""
        
        @partial(jax.jit, static_argnames=("training",))
        def fused_kernel(
            x: jnp.ndarray,
            weight: jnp.ndarray,
            bias: Optional[jnp.ndarray] = None,
            training: bool = True,
            rng: Optional[Any] = None
        ) -> jnp.ndarray:
            # Linear
            y = jnp.dot(x, weight)
            if bias is not None:
                y = y + bias
                
            # GELU
            y = jax.nn.gelu(y)
            
            # Dropout
            if training and dropout_rate > 0:
                if rng is None:
                    rng = jax.random.PRNGKey(0)
                y = jax.random.dropout(rng, dropout_rate, y)
                
            return y
            
        return fused_kernel
        
    def _create_qkv_projection_split(
        self,
        hidden_size: int,
        num_heads: int,
        head_size: Optional[int] = None
    ) -> Callable:
        """Create fused QKV projection + split kernel."""
        if head_size is None:
            head_size = hidden_size // num_heads
            
        @jax.jit
        def fused_kernel(
            x: jnp.ndarray,
            qkv_weight: jnp.ndarray,
            qkv_bias: Optional[jnp.ndarray] = None
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            # Combined QKV projection
            qkv = jnp.dot(x, qkv_weight)
            if qkv_bias is not None:
                qkv = qkv + qkv_bias
                
            # Split and reshape
            batch_size = x.shape[0]
            seq_len = x.shape[1]
            qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_size)
            qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
            
            # Split Q, K, V
            q, k, v = jnp.split(qkv, 3, axis=0)
            return (
                q.squeeze(0),  # [batch, heads, seq, head_size]
                k.squeeze(0),
                v.squeeze(0)
            )
            
        return fused_kernel