"""TPU kernel fusion manager for optimized training."""

from typing import Dict, List, Optional, Any, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

from .pipeline_config import TPUPipelineConfig
from .flash_attention import TPUFlashAttention
from .gemm import TPUGEMMKernel

class TPUKernelFusionManager:
    """Manages kernel fusion and optimization for TPU training."""
    
    def __init__(self, config: TPUPipelineConfig):
        self.config = config
        self.flash_attention = TPUFlashAttention(
            block_size=config.block_size,
            use_bfloat16=config.use_bfloat16,
            use_fp8=config.use_fp8
        )
        self.gemm = TPUGEMMKernel(
            block_size=config.block_size,
            use_bfloat16=config.use_bfloat16,
            use_fp8=config.use_fp8,
            precision=config.matmul_precision
        )
        
        # Initialize fusion patterns
        self.fusion_patterns = self._create_fusion_patterns()
        
    def _create_fusion_patterns(self) -> Dict[str, Any]:
        """Create optimized fusion patterns for common operations."""
        patterns = {}
        
        # Fused attention + dropout + residual
        def fused_attention_pattern(
            q: jnp.ndarray,
            k: jnp.ndarray,
            v: jnp.ndarray,
            residual: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None,
            dropout_rng: Optional[Any] = None
        ) -> jnp.ndarray:
            # Run flash attention
            attn_output = self.flash_attention(q, k, v, mask=mask)
            
            # Fused dropout + residual
            if dropout_rng is not None and self.config.dropout_rate > 0:
                keep_prob = 1.0 - self.config.dropout_rate
                dropout_mask = jax.random.bernoulli(
                    dropout_rng,
                    p=keep_prob,
                    shape=attn_output.shape
                )
                attn_output = attn_output * dropout_mask / keep_prob
            
            return attn_output + residual
            
        patterns["attention_dropout_residual"] = fused_attention_pattern
        
        # Fused FFN + dropout + residual
        def fused_ffn_pattern(
            x: jnp.ndarray,
            w1: jnp.ndarray,
            w2: jnp.ndarray,
            residual: jnp.ndarray,
            dropout_rng: Optional[Any] = None,
            activation: str = "gelu"
        ) -> jnp.ndarray:
            # First linear layer
            hidden = self.gemm(x, w1)
            
            # Activation
            if activation == "gelu":
                hidden = jax.nn.gelu(hidden)
            elif activation == "relu":
                hidden = jax.nn.relu(hidden)
            else:
                raise ValueError(f"Unsupported activation: {activation}")
                
            # Second linear layer with fused dropout + residual
            output = self.gemm(hidden, w2)
            
            if dropout_rng is not None and self.config.dropout_rate > 0:
                keep_prob = 1.0 - self.config.dropout_rate
                dropout_mask = jax.random.bernoulli(
                    dropout_rng,
                    p=keep_prob,
                    shape=output.shape
                )
                output = output * dropout_mask / keep_prob
                
            return output + residual
            
        patterns["ffn_dropout_residual"] = fused_ffn_pattern
        
        # Fused layer norm
        def fused_layer_norm(
            x: jnp.ndarray,
            weight: Optional[jnp.ndarray] = None,
            bias: Optional[jnp.ndarray] = None,
            eps: float = 1e-6
        ) -> jnp.ndarray:
            mean = jnp.mean(x, axis=-1, keepdims=True)
            variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
            x_norm = (x - mean) * jax.lax.rsqrt(variance + eps)
            
            if weight is not None:
                x_norm = x_norm * weight
            if bias is not None:
                x_norm = x_norm + bias
                
            return x_norm
            
        patterns["layer_norm"] = fused_layer_norm
        
        return patterns
        
    def get_fused_pattern(self, pattern_name: str):
        """Get a specific fusion pattern."""
        if pattern_name not in self.fusion_patterns:
            raise ValueError(f"Unknown fusion pattern: {pattern_name}")
        return self.fusion_patterns[pattern_name]
        
    @partial(jax.jit, static_argnums=(0,))
    def fused_transformer_layer(
        self,
        hidden_states: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        layer_params: Dict[str, jnp.ndarray],
        dropout_rng: Optional[Any] = None
    ) -> jnp.ndarray:
        """Fused transformer layer computation."""
        # Layer norm 1
        ln1_out = self.fusion_patterns["layer_norm"](
            hidden_states,
            layer_params["ln1_weight"],
            layer_params["ln1_bias"]
        )
        
        # Self attention
        q = self.gemm(ln1_out, layer_params["q_weight"])
        k = self.gemm(ln1_out, layer_params["k_weight"]) 
        v = self.gemm(ln1_out, layer_params["v_weight"])
        
        attn_out = self.fusion_patterns["attention_dropout_residual"](
            q, k, v,
            residual=hidden_states,
            mask=attention_mask,
            dropout_rng=dropout_rng
        )
        
        # Layer norm 2
        ln2_out = self.fusion_patterns["layer_norm"](
            attn_out,
            layer_params["ln2_weight"],
            layer_params["ln2_bias"]
        )
        
        # Feed forward
        ffn_out = self.fusion_patterns["ffn_dropout_residual"](
            ln2_out,
            layer_params["ffn_w1"],
            layer_params["ffn_w2"],
            residual=attn_out,
            dropout_rng=dropout_rng
        )
        
        return ffn_out
        
    def create_optimized_forward(self, num_layers: int):
        """Create optimized forward pass function."""
        
        @partial(jax.jit, static_argnums=(0,))
        def optimized_forward(
            self,
            hidden_states: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray],
            params: Dict[str, Dict[str, jnp.ndarray]],
            dropout_rng: Optional[Any] = None
        ) -> jnp.ndarray:
            def layer_fn(state, layer_idx):
                layer_params = params[f"layer_{layer_idx}"]
                return self.fused_transformer_layer(
                    state,
                    attention_mask,
                    layer_params,
                    dropout_rng
                ), None
                
            # Process all layers with automatic rematerialization
            final_state, _ = jax.lax.scan(
                layer_fn,
                hidden_states,
                jnp.arange(num_layers),
                unroll=self.config.remat_granularity
            )
            
            return final_state
            
        return optimized_forward