"""
Core TPU functionality and optimizations using JAX/XLA
"""

import jax
import jax.numpy as jnp
from jax import lax, random, jit, vmap
import numpy as np
from typing import Optional, Dict, List, Tuple, Union, Any
import math
import os

# TPU configuration
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
jax.config.update("jax_enable_x64", False)  # Disable float64 for TPU

def configure_tpu_dtypes():
    """Configure TPU dtype settings"""
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    # Don't set default dtype to bfloat16 globally to avoid affecting embeddings
    return {
        'compute_dtype': jnp.bfloat16,
        'embedding_dtype': jnp.int32,
        'default_float': jnp.float32
    }

def _apply_rotary(x: jnp.ndarray, freqs_cis: jnp.ndarray) -> jnp.ndarray:
    """Internal function to apply rotary embeddings.
    
    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim]
        freqs_cis: Complex rotary embeddings [seq_len, head_dim/2]
    Returns:
        Tensor with rotary embeddings applied
    """
    # Reshape for complex multiplication
    x_complex = x.reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.reshape(1, freqs_cis.shape[0], 1, -1)
    
    # Convert to complex numbers for rotation
    x_complex = jnp.complex64(x_complex[..., 0]) + 1j * jnp.complex64(x_complex[..., 1])
    # Apply rotation through complex multiplication
    x_rotated = x_complex * freqs_cis
    
    # Convert back to real numbers and proper shape
    x_out = jnp.stack([jnp.real(x_rotated), jnp.imag(x_rotated)], axis=-1)
    return x_out.reshape(*x.shape)

@jit
def apply_rotary_embedding(x: jnp.ndarray, freqs_cis: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary position embeddings to input tensor.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, num_heads, head_dim]
        freqs_cis: Complex rotary embeddings [seq_len, head_dim/2]
    Returns:
        Tensor with rotary embeddings applied
    """
    # Ensure compute dtype for rotary embedding
    x = x.astype(jnp.bfloat16)
    freqs_cis = freqs_cis.astype(jnp.bfloat16)
    return _apply_rotary(x, freqs_cis)

def create_causal_mask(seq_len: int, batch_size: Optional[int] = None) -> jnp.ndarray:
    """Create causal attention mask for transformer decoder.
    
    Args:
        seq_len: Length of the sequence
        batch_size: Optional batch size for broadcasting
    Returns:
        Causal mask of shape [batch_size, 1, seq_len, seq_len] if batch_size is provided,
        otherwise [1, 1, seq_len, seq_len]
    """
    # Use float32 for mask to avoid dtype issues
    mask = jnp.triu(jnp.ones((seq_len, seq_len), dtype=jnp.float32), k=1)
    mask = -1e9 * mask
    if batch_size:
        mask = jnp.expand_dims(mask, axis=0)
        mask = jnp.repeat(mask, batch_size, axis=0)
    return mask

class TPUDeviceManager:
    """Manages TPU device configuration and optimization"""
    
    @staticmethod
    def configure_for_tpu():
        """Configure TPU settings"""
        jax.config.update("jax_platforms", "tpu")
        jax.config.update("jax_xla_backend", "tpu")
        return configure_tpu_dtypes()
    
    @staticmethod
    def get_device_count() -> int:
        """Get number of available TPU devices"""
        return len(jax.devices())
        
    @staticmethod
    def get_device_type() -> str:
        """Get TPU device type"""
        return jax.devices()[0].device_kind
        
    @staticmethod
    def get_optimal_batch_size(model_dim: int, seq_len: int) -> int:
        """Calculate optimal batch size for TPU memory"""
        device_mem = 8 * (1024 ** 3)  # Assume 8GB per TPU core
        elem_size = 2  # bfloat16
        overhead = 1.2  # 20% overhead for activations
        return int(device_mem / (model_dim * seq_len * elem_size * overhead))

    @staticmethod
    def get_hardware_capabilities() -> Dict[str, Any]:
        """Get hardware capabilities of the current device"""
        device = jax.devices()[0]
        return {
            "device_type": device.device_kind,
            "platform": device.platform,
            "device_count": len(jax.devices()),
            "memory_per_device": "8GB",  # Standard for TPU v3-8
            "supports_bfloat16": True,
            "supports_tf32": False,
            "supports_mixed_precision": True
        }

class TPUOptimizer:
    """TPU-specific optimizations for model operations"""
    
    @staticmethod
    @jit
    def fused_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                       mask: Optional[jnp.ndarray] = None,
                       dropout_rate: float = 0.0,
                       deterministic: bool = False) -> jnp.ndarray:
        """Fused attention computation optimized for TPU"""
        # Ensure compute dtype
        q = q.astype(jnp.bfloat16)
        k = k.astype(jnp.bfloat16)
        v = v.astype(jnp.bfloat16)
        
        d_k = q.shape[-1]
        scale = jnp.sqrt(d_k).astype(jnp.bfloat16)
        
        # Attention scores with optimized matmul
        scores = lax.dot_general(
            q, k,
            dimension_numbers=(((q.ndim - 1,), (k.ndim - 2,)), ((), ())),
        ) / scale
        
        if mask is not None:
            # Convert mask to same dtype as scores
            mask = mask.astype(scores.dtype)
            scores = jnp.where(mask, scores, -1e9)
            
        # Optimized softmax
        scores = jax.nn.softmax(scores, axis=-1)
        
        if dropout_rate > 0.0 and not deterministic:
            key = random.PRNGKey(0)  # Should be passed from outside in practice
            scores = random.bernoulli(key, 1.0 - dropout_rate, scores.shape) * scores
            scores = scores / (1.0 - dropout_rate)
            
        # Optimized output computation
        return lax.dot_general(
            scores, v,
            dimension_numbers=(((scores.ndim - 1,), (v.ndim - 2,)), ((), ())),
        )

    @staticmethod
    @jit
    def fused_ffn(x: jnp.ndarray, w1: jnp.ndarray, b1: jnp.ndarray,
                  w2: jnp.ndarray, b2: jnp.ndarray) -> jnp.ndarray:
        """Fused feed-forward network computation"""
        # Ensure compute dtype
        x = x.astype(jnp.bfloat16)
        w1 = w1.astype(jnp.bfloat16)
        w2 = w2.astype(jnp.bfloat16)
        b1 = b1.astype(jnp.bfloat16)
        b2 = b2.astype(jnp.bfloat16)
        
        return lax.dot_general(
            jax.nn.gelu(lax.dot_general(x, w1, ((1,), (0,)), ((), ())) + b1),
            w2, ((1,), (0,)), ((), ())
        ) + b2

class TPUDataParallel:
    """Data parallel training utilities for TPU"""
    
    def __init__(self, num_devices: Optional[int] = None):
        self.num_devices = num_devices or len(jax.devices())
        
    def shard_batch(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Shard batch across TPU cores"""
        return jax.tree_map(
            lambda x: x.reshape(self.num_devices, -1, *x.shape[1:]),
            batch
        )
        
    def gather_results(self, sharded_outputs: jnp.ndarray) -> jnp.ndarray:
        """Gather results from TPU cores"""
        return sharded_outputs.reshape(-1, *sharded_outputs.shape[2:])

    @staticmethod
    def cross_replica_sum(x: jnp.ndarray) -> jnp.ndarray:
        """Sum values across TPU replicas"""
        return lax.psum(x, axis_name='batch')

class TPUProfiler:
    """TPU profiling and performance monitoring"""
    
    @staticmethod
    def profile_execution(fn, *args, **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Profile function execution on TPU"""
        with jax.profiler.trace("profile") as trace:
            result = fn(*args, **kwargs)
            stats = {
                "compile_time": trace.compile_time,
                "execution_time": trace.execution_time
            }
        return result, stats
        
    @staticmethod
    def get_memory_usage() -> Dict[str, int]:
        """Get TPU memory usage statistics"""
        usage = {
            "total": 0,
            "used": 0,
            "free": 0
        }
        # Note: Actual memory stats would require TPU driver APIs
        return usage

class TPUModelUtils:
    """Utility functions for TPU model operations"""
    
    @staticmethod
    def get_gradient_checkpoint_policy(memory_threshold: float = 0.9):
        """Get gradient checkpointing policy based on memory usage"""
        return jax.checkpoint_policies.save_anything_except_buffers

# Example usage
if __name__ == "__main__":
    # Initialize TPU configuration
    TPUDeviceManager.configure_for_tpu()
    
    # Test attention computation
    batch_size, seq_len, d_model = 2, 32, 64
    num_heads = 8
    head_dim = d_model // num_heads
    
    # Generate random inputs
    rng = random.PRNGKey(0)
    q = random.normal(rng, (batch_size, seq_len, num_heads, head_dim))
    k = random.normal(rng, (batch_size, seq_len, num_heads, head_dim))
    v = random.normal(rng, (batch_size, seq_len, num_heads, head_dim))
    
    # Test fused attention
    output = TPUOptimizer.fused_attention(q, k, v)
    print("Attention output shape:", output.shape)
    
    # Test data parallel
    dp = TPUDataParallel()
    batch = {"input_ids": jnp.ones((16, seq_len))}
    sharded = dp.shard_batch(batch)
    print("Sharded batch shape:", sharded["input_ids"].shape)

# Initialize TPU dtypes on module import
DTYPE_CONFIG = configure_tpu_dtypes()