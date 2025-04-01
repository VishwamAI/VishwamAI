"""TPU-optimized matrix multiplication kernels."""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from typing import Optional, Tuple, Dict, NamedTuple
from functools import partial

from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class MatMulOutput(NamedTuple):
    """Output from matrix multiplication."""
    output: jnp.ndarray
    intermediates: Optional[Dict[str, jnp.ndarray]] = None

class TPUMatMulKernel:
    """
    TPU-optimized matrix multiplication implementation.
    
    Features:
    - Tile-based computation for TPU efficiency
    - Automatic mixed precision
    - Memory layout optimization
    - Support for sparse operations
    """
    
    def __init__(
        self,
        block_size: int = 128,
        use_bias: bool = True,
        precision: str = "highest",
        sparsity_threshold: Optional[float] = None,
        profile: bool = False
    ):
        """
        Initialize TPU matmul kernel.
        
        Args:
            block_size: Block size for tiling (must be multiple of 128)
            use_bias: Whether to use bias
            precision: Precision mode ("highest", "high", or "default")
            sparsity_threshold: Optional threshold for sparse computation
            profile: Whether to collect performance metrics
        """
        if block_size % 128 != 0:
            raise ValueError("Block size must be multiple of 128 for TPU")
            
        self.block_size = block_size
        self.use_bias = use_bias
        self.precision = getattr(lax.Precision, precision.upper())
        self.sparsity_threshold = sparsity_threshold
        self.profile = profile

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        bias: Optional[jnp.ndarray] = None,
        transpose_a: bool = False,
        transpose_b: bool = False
    ) -> MatMulOutput:
        """
        Compute matrix multiplication.
        
        Args:
            a: First input matrix
            b: Second input matrix
            bias: Optional bias
            transpose_a: Whether to transpose first matrix
            transpose_b: Whether to transpose second matrix
            
        Returns:
            MatMulOutput with result and intermediates
        """
        # Get shapes
        M = a.shape[-2] if not transpose_a else a.shape[-1]
        K = a.shape[-1] if not transpose_a else a.shape[-2]
        N = b.shape[-1] if not transpose_b else b.shape[-2]
        
        # Optimize memory layout
        a = optimize_tpu_layout(a, self.block_size)
        b = optimize_tpu_layout(b, self.block_size)
        
        # Handle sparse computation if enabled
        if self.sparsity_threshold is not None:
            return self._sparse_matmul(a, b, bias, transpose_a, transpose_b)
            
        # Pad dimensions to block size
        M_pad = (self.block_size - M % self.block_size) % self.block_size
        N_pad = (self.block_size - N % self.block_size) % self.block_size
        K_pad = (self.block_size - K % self.block_size) % self.block_size
        
        if M_pad > 0 or K_pad > 0:
            a = jnp.pad(a, ((0, M_pad), (0, K_pad)))
        if K_pad > 0 or N_pad > 0:
            b = jnp.pad(b, ((0, K_pad), (0, N_pad)))
            
        # Handle transposes
        if transpose_a:
            a = jnp.transpose(a, (0, 2, 1))
        if transpose_b:
            b = jnp.transpose(b, (0, 2, 1))
            
        # Reshape into blocks for TPU efficiency
        a_blocked = a.reshape(-1, self.block_size, K // self.block_size, self.block_size)
        b_blocked = b.reshape(K // self.block_size, self.block_size, -1, self.block_size)
        
        # Compute blocked matrix multiplication
        result = jax.lax.dot_general(
            a_blocked,
            b_blocked,
            dimension_numbers=(((2, 3), (0, 1)), ((0, 1), (2, 3))),
            precision=self.precision
        )
        
        # Add bias if provided
        if bias is not None and self.use_bias:
            result = result + bias
            
        # Remove padding if needed
        if M_pad > 0 or N_pad > 0:
            result = result[:M, :N]
            
        # Collect metrics if profiling
        intermediates = None
        if self.profile:
            intermediates = {
                "input_shapes": (a.shape, b.shape),
                "padded_shapes": (a_blocked.shape, b_blocked.shape),
                "output_shape": result.shape
            }
            
        return MatMulOutput(result, intermediates)
        
    def _sparse_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        bias: Optional[jnp.ndarray],
        transpose_a: bool,
        transpose_b: bool
    ) -> MatMulOutput:
        """
        Compute sparse matrix multiplication.
        
        Uses block-sparse format optimized for TPU.
        """
        # Create sparsity masks
        a_mask = jnp.abs(a) > self.sparsity_threshold
        b_mask = jnp.abs(b) > self.sparsity_threshold
        
        # Convert to block-sparse format
        def to_block_sparse(x, mask):
            # Reshape into blocks
            x_blocks = x.reshape(-1, self.block_size, x.shape[-1] // self.block_size, self.block_size)
            mask_blocks = mask.reshape(-1, self.block_size, mask.shape[-1] // self.block_size, self.block_size)
            
            # Keep only non-zero blocks
            block_mask = jnp.any(mask_blocks, axis=(1, 3))
            x_sparse = x_blocks[block_mask]
            
            return x_sparse, block_mask
            
        a_sparse, a_block_mask = to_block_sparse(a, a_mask)
        b_sparse, b_block_mask = to_block_sparse(b, b_mask)
        
        # Compute sparse matrix multiplication
        result_sparse = jax.lax.dot_general(
            a_sparse,
            b_sparse,
            dimension_numbers=(((2, 3), (0, 1)), ((0, 1), (2, 3))),
            precision=self.precision
        )
        
        # Convert back to dense
        def to_dense(x_sparse, block_mask, out_shape):
            dense = jnp.zeros(out_shape)
            return dense.at[block_mask].set(x_sparse)
            
        result = to_dense(
            result_sparse,
            a_block_mask & b_block_mask,
            (a.shape[0], b.shape[1])
        )
        
        if bias is not None and self.use_bias:
            result = result + bias
            
        return MatMulOutput(
            result,
            {
                "sparsity": {
                    "a_sparsity": jnp.mean(a_mask),
                    "b_sparsity": jnp.mean(b_mask)
                }
            }
        )
        
    def efficient_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        compute_dtype: Optional[jnp.dtype] = None
    ) -> jnp.ndarray:
        """
        Memory-efficient matrix multiplication optimized for TPU.
        
        Features:
        - Auto mixed precision
        - Progressive computation
        - Memory-optimal tiling
        """
        # Auto mixed precision
        if compute_dtype is None:
            compute_dtype = jnp.bfloat16
            
        # Cast inputs
        a = a.astype(compute_dtype)
        b = b.astype(compute_dtype)
        
        # Get optimal tile size
        tile_size = self._get_optimal_tile_size(a.shape, b.shape)
        
        # Split computation into tiles
        result_parts = []
        
        for i in range(0, a.shape[0], tile_size):
            row_parts = []
            a_tile = jax.lax.dynamic_slice(
                a,
                (i, 0),
                (min(tile_size, a.shape[0] - i), a.shape[1])
            )
            
            for j in range(0, b.shape[1], tile_size):
                b_tile = jax.lax.dynamic_slice(
                    b,
                    (0, j),
                    (b.shape[0], min(tile_size, b.shape[1] - j))
                )
                
                # Compute tile product
                tile_result = jax.lax.dot(
                    a_tile,
                    b_tile,
                    precision=self.precision
                )
                row_parts.append(tile_result)
                
            # Combine row parts
            result_parts.append(jnp.concatenate(row_parts, axis=1))
            
        # Combine all parts
        result = jnp.concatenate(result_parts, axis=0)
        
        # Cast back to original dtype
        return result.astype(a.dtype)
        
    def _get_optimal_tile_size(
        self,
        a_shape: Tuple[int, ...],
        b_shape: Tuple[int, ...]
    ) -> int:
        """Calculate optimal tile size based on TPU memory."""
        # Get TPU memory constraints
        device = jax.devices()[0]
        memory_per_core = device.memory_per_core
        
        # Calculate sizes
        matrix_size = np.prod(a_shape) + np.prod(b_shape)
        element_size = 2  # bfloat16
        
        # Leave room for intermediates
        available_memory = memory_per_core * 0.8
        
        # Calculate maximum tile size
        max_tile = int(np.sqrt(available_memory / (3 * element_size)))
        
        # Round down to multiple of TPU block size
        return (max_tile // self.block_size) * self.block_size