"""Hybrid MatMul implementation optimized for both TPU and GPU execution."""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Tuple, Optional, Dict, Any, Union, Literal, List
from vishwamai.kernels.core.kernel import optimize_kernel_layout, act_quant
from vishwamai.kernels.cuda.flash_kv import FlashKVCache
from vishwamai.kernels.ops.tree_matmul import TreeMatMul

DeviceType = Literal["tpu", "gpu"]

class HybridMatMul:
    """Implements optimized matrix multiplication for TPU/GPU."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        device_type: DeviceType = "tpu",
        block_size: int = 128,
        use_fp8: bool = True
    ):
        """Initialize hybrid matmul.
        
        Args:
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            device_type: Target device ("tpu" or "gpu")
            block_size: Block size for tiling
            use_fp8: Whether to use FP8 precision
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.device_type = device_type
        self.block_size = block_size
        self.use_fp8 = use_fp8
        
        # Initialize device-specific components
        if device_type == "tpu":
            # TPU prefers certain block sizes and fp8
            self.block_size = 128  # TPU optimal
            self.use_flash_attn = False  # Not optimal for TPU
        else:
            # GPU benefits from flash attention
            self.block_size = 64   # GPU optimal
            self.use_flash_attn = True
            
        # Set up helper modules
        self.kv_cache = FlashKVCache(
            max_length=2048,  # Configurable
            num_heads=num_heads,
            head_dim=hidden_dim // num_heads,
            block_size=self.block_size,
            use_fp8=self.use_fp8
        )
        
        self.tree_matmul = TreeMatMul(
            num_layers=24,  # Configurable
            hidden_dim=hidden_dim,
            block_size=self.block_size,
            use_fp8=self.use_fp8
        )

    def matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_b: bool = False
    ) -> jnp.ndarray:
        """Optimized matrix multiplication for current device."""
        if self.device_type == "tpu":
            return self.tpu_matmul(a, b, transpose_b)
        else:
            return self.gpu_matmul(a, b, transpose_b)

    def tpu_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_b: bool = False
    ) -> jnp.ndarray:
        """TPU-optimized matrix multiplication."""
        # Cast to optimal precision
        if self.use_fp8:
            a, a_scale = act_quant(a, block_size=self.block_size)
            b, b_scale = act_quant(b, block_size=self.block_size)
        
        # Optimize memory layout
        a = optimize_kernel_layout(a)
        b = optimize_kernel_layout(b)
        
        if transpose_b:
            b = b.transpose(*range(b.ndim - 2), -1, -2)
        
        # Use TPU dot_general for optimal performance
        c = jax.lax.dot_general(
            a, b,
            (((a.ndim - 1,), (b.ndim - 2,)), ((), ())),
            precision=jax.lax.Precision.HIGHEST
        )
        
        # Scale back if using FP8
        if self.use_fp8:
            c = c * (a_scale * b_scale)
            
        return c

    def gpu_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_b: bool = False
    ) -> jnp.ndarray:
        """GPU-optimized matrix multiplication."""
        # For GPU, we use a different blocking strategy
        M, K = a.shape[-2:]
        N = b.shape[-1] if not transpose_b else b.shape[-2]
        
        # Round dimensions up to multiple of block size
        M_pad = (self.block_size - M % self.block_size) % self.block_size
        N_pad = (self.block_size - N % self.block_size) % self.block_size
        K_pad = (self.block_size - K % self.block_size) % self.block_size
        
        # Pad inputs
        a_padded = jnp.pad(a, ((0, 0), (0, M_pad), (0, K_pad)))
        b_padded = jnp.pad(b, ((0, 0), (0, K_pad), (0, N_pad)) if not transpose_b
                          else ((0, 0), (0, N_pad), (0, K_pad)))
        
        if transpose_b:
            b_padded = b_padded.transpose(*range(b_padded.ndim - 2), -1, -2)
        
        # Reshape for efficient GPU execution
        M_blocks = (M + M_pad) // self.block_size
        N_blocks = (N + N_pad) // self.block_size
        K_blocks = (K + K_pad) // self.block_size
        
        a_reshaped = a_padded.reshape(-1, M_blocks, self.block_size,
                                    K_blocks, self.block_size)
        b_reshaped = b_padded.reshape(-1, K_blocks, self.block_size,
                                    N_blocks, self.block_size)
        
        # Compute matrix multiplication
        c_reshaped = jnp.einsum(
            'bmskt,bktnu->bmsnu',
            a_reshaped,
            b_reshaped,
            precision=jax.lax.Precision.HIGHEST
        )
        
        # Reshape result and remove padding
        c = c_reshaped.reshape(-1, M + M_pad, N + N_pad)
        c = c[..., :M, :N]
        
        return c

    def attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Optimized attention computation for current device."""
        if self.device_type == "tpu":
            return self.tpu_attention(q, k, v, mask)
        else:
            return self.gpu_attention(q, k, v, mask)

    def tpu_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """TPU-optimized attention computation."""
        # Compute attention scores with optimal TPU layout
        scores = self.tpu_matmul(q, k, transpose_b=True)
        scores = scores / jnp.sqrt(q.shape[-1])
        
        # Apply mask if provided
        if mask is not None:
            scores = jnp.where(mask, scores, -1e10)
        
        # Apply softmax
        weights = jax.nn.softmax(scores, axis=-1)
        
        # Compute attention output
        output = self.tpu_matmul(weights, v)
        return output

    def gpu_attention(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """GPU-optimized attention using flash attention."""
        if not self.use_flash_attn:
            return self.tpu_attention(q, k, v, mask)
            
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Process in blocks for flash attention
        def process_block(block_start):
            block_size = min(self.block_size, seq_len - block_start)
            
            # Get query block
            q_block = jax.lax.dynamic_slice(
                q,
                (0, 0, block_start, 0),
                (batch_size, num_heads, block_size, head_dim)
            )
            
            # Compute attention scores for this block
            scores = jnp.einsum('bhqd,bhkd->bhqk', q_block, k)
            scores = scores / jnp.sqrt(head_dim)
            
            # Apply mask if provided
            if mask is not None:
                mask_block = jax.lax.dynamic_slice(
                    mask,
                    (0, 0, block_start, 0),
                    (batch_size, num_heads, block_size, mask.shape[-1])
                )
                scores = jnp.where(mask_block, scores, -1e10)
            
            # Compute attention weights
            weights = jax.nn.softmax(scores, axis=-1)
            
            # Apply attention
            return jnp.einsum('bhqk,bhkd->bhqd', weights, v)
            
        # Process all blocks
        outputs = []
        for block_start in range(0, seq_len, self.block_size):
            block_output = process_block(block_start)
            outputs.append(block_output)
            
        return jnp.concatenate(outputs, axis=2)

    def adaptive_compute(
        self,
        x: jnp.ndarray,
        weights: List[jnp.ndarray],
        depth_scales: jnp.ndarray
    ) -> jnp.ndarray:
        """Apply adaptive computation using tree matmul."""
        return self.tree_matmul.forward(x, weights, depth_scales)

    @staticmethod
    def get_optimal_config(device_type: DeviceType) -> Dict[str, Any]:
        """Get optimal configuration for given device type."""
        if device_type == "tpu":
            return {
                "block_size": 128,
                "use_fp8": True,
                "use_flash_attn": False,
                "precision": jax.lax.Precision.HIGHEST
            }
        else:
            return {
                "block_size": 64,
                "use_fp8": False,
                "use_flash_attn": True,
                "precision": jax.lax.Precision.DEFAULT
            }