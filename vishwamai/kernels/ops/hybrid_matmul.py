"""Hybrid matrix multiplication strategies optimized for TPU/GPU."""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union, NamedTuple
from functools import partial

from vishwamai.kernels.optimizers.quantization import dynamic_quant, dequantize
from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple
from vishwamai.kernels.ops.tree_matmul import TreeMatMul

class HybridMatMulOutput(NamedTuple):
    """Output from hybrid matrix multiplication."""
    output: jnp.ndarray
    quant_scale: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None

class HybridMatMul:
    """
    Hybrid matrix multiplication that adaptively switches between different
    computation strategies based on input size and hardware. Optimized for
    TPU with support for quantization and tree-based computation.
    """
    
    def __init__(
        self,
        block_size: int = 128,  # Changed to 128 for TPU optimization
        min_parallel_size: int = 512,
        use_tree: bool = True,
        use_quantization: bool = False,
        use_bfloat16: bool = True,
        precision: Optional[jax.lax.Precision] = None,
        tree_config: Optional[dict] = None
    ):
        """
        Initialize hybrid matrix multiplication.
        
        Args:
            block_size: Size of blocks for blocked multiplication (multiple of 128 for TPU)
            min_parallel_size: Minimum matrix size to use parallel strategy
            use_tree: Whether to use tree-based multiplication for large matrices
            use_quantization: Whether to use int8 quantization for TPU
            use_bfloat16: Whether to use bfloat16 precision
            precision: JAX precision setting for computation
            tree_config: Configuration for tree-based multiplication
        """
        if block_size % 128 != 0:
            raise ValueError("Block size must be a multiple of 128 for TPU")
            
        self.block_size = block_size
        self.min_parallel_size = min_parallel_size
        self.use_tree = use_tree
        self.use_quantization = use_quantization
        self.use_bfloat16 = use_bfloat16
        self.precision = precision or jax.lax.Precision.HIGHEST
        
        # Initialize tree-based multiplication if enabled
        if use_tree:
            self.tree_matmul = TreeMatMul(**(tree_config or {}))
    
    def _blocked_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        a_scale: Optional[jnp.ndarray] = None,
        b_scale: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Blocked matrix multiplication optimized for TPU."""
        # Optimize memory layout for TPU
        a = optimize_tpu_layout(a)
        b = optimize_tpu_layout(b)
        
        m, k = a.shape
        k_, n = b.shape
        assert k == k_, f"Incompatible dimensions: {k} != {k_}"

        # Pad matrices to block size for TPU efficiency
        pad_m = (self.block_size - m % self.block_size) % self.block_size
        pad_n = (self.block_size - n % self.block_size) % self.block_size
        pad_k = (self.block_size - k % self.block_size) % self.block_size
        
        a_pad = pad_to_tpu_multiple(a, self.block_size)
        b_pad = pad_to_tpu_multiple(b, self.block_size)
        
        # Use JAX's dot_general which is highly optimized for TPU
        if a_scale is not None and b_scale is not None:
            # Handle quantized multiplication
            result = jax.lax.dot_general(
                a_pad.astype(jnp.float32), 
                b_pad.astype(jnp.float32),
                (((1,), (0,)), ((), ())),
                precision=self.precision
            )
            result = result * (a_scale * b_scale)
        else:
            result = jax.lax.dot_general(
                a_pad, 
                b_pad,
                (((1,), (0,)), ((), ())),
                precision=self.precision
            )
            
        # Remove padding
        if pad_m > 0 or pad_n > 0:
            result = result[:m, :n]
            
        return result
    
    def _parallel_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        a_scale: Optional[jnp.ndarray] = None,
        b_scale: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Parallel matrix multiplication optimized for TPU/GPU."""
        if a_scale is not None and b_scale is not None:
            # Handle quantized multiplication
            result = jax.vmap(lambda x: x @ b.astype(jnp.float32))(a.astype(jnp.float32))
            return result * (a_scale * b_scale)
        else:
            return jax.vmap(lambda x: x @ b)(a)
    
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        return_quant_scale: bool = False
    ) -> Union[jnp.ndarray, HybridMatMulOutput]:
        """
        Perform hybrid matrix multiplication with automatic strategy selection.
        
        Args:
            a: First input matrix
            b: Second input matrix
            transpose_a: Whether to transpose first matrix
            transpose_b: Whether to transpose second matrix
            return_quant_scale: Whether to return quantization scales
            
        Returns:
            Matrix multiplication result, optionally with quantization scales
        """
        # Handle dtype conversion
        orig_dtype = a.dtype
        if self.use_bfloat16:
            a = a.astype(jnp.bfloat16)
            b = b.astype(jnp.bfloat16)
            
        # Handle transposes
        if transpose_a:
            a = jnp.transpose(a)
        if transpose_b:
            b = jnp.transpose(b)
            
        # Quantize inputs if enabled
        a_scale = b_scale = None
        if self.use_quantization:
            a_quant, a_scale = dynamic_quant(a)
            b_quant, b_scale = dynamic_quant(b)
            a, b = a_quant, b_quant
            
        # Choose multiplication strategy based on size
        m = a.shape[0]
        n = b.shape[-1] if b.ndim > 1 else 1
        k = a.shape[-1]
        
        result = None
        
        # Use tree-based approach for very large matrices
        if self.use_tree and max(m, n, k) >= self.min_parallel_size * 4:
            result = self.tree_matmul(a, b)
            if a_scale is not None and b_scale is not None:
                result = result * (a_scale * b_scale)
                
        # Use parallel approach for medium-large matrices
        elif m >= self.min_parallel_size:
            result = self._parallel_matmul(a, b, a_scale, b_scale)
            
        # Use blocked approach for smaller matrices
        else:
            result = self._blocked_matmul(a, b, a_scale, b_scale)
            
        # Convert back to original dtype if needed
        if self.use_bfloat16 and orig_dtype != jnp.bfloat16:
            result = result.astype(orig_dtype)
            
        if return_quant_scale:
            return HybridMatMulOutput(result, (a_scale, b_scale) if a_scale is not None else None)
        return result
