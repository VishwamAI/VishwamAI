"""TPU-optimized GEMM (General Matrix Multiplication) implementation."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Optional, Tuple, Union, Dict, Any
import numpy as np
from functools import partial

from vishwamai.kernels.core.kernel import Kernel, KernelConfig
from vishwamai.kernels.core.kernel_manager import HardwareType
from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout, pad_to_tpu_multiple

class TPUGEMMKernel:
    """TPU-optimized matrix multiplication kernel.
    
    Provides efficient matrix multiplication operations optimized for TPU hardware
    with automatic tiling, memory layout transformation, and mixed precision support.
    """
    
    def __init__(
        self,
        block_size: int = 128,
        precision: Optional[lax.Precision] = None,
        use_fp8: bool = False,
        use_bfloat16: bool = True,
        reuse_workspace: bool = True,
    ):
        """Initialize TPU GEMM kernel.
        
        Args:
            block_size: Size of blocks for tiling (should be multiple of 128 for TPU)
            precision: JAX precision setting for computation
            use_fp8: Whether to use FP8 precision for weights
            use_bfloat16: Whether to use bfloat16 for computation
            reuse_workspace: Whether to reuse workspace memory between operations
        """
        self.block_size = block_size
        self.precision = precision or lax.Precision.HIGHEST
        self.use_fp8 = use_fp8
        self.use_bfloat16 = use_bfloat16
        self.reuse_workspace = reuse_workspace
        
        # Ensure block_size is appropriate for TPU
        if block_size % 128 != 0:
            raise ValueError("Block size must be a multiple of 128 for TPU")
    
    def __call__(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        scale: Optional[float] = None,
    ) -> jnp.ndarray:
        """Perform matrix multiplication optimized for TPU.
        
        Args:
            a: First input matrix
            b: Second input matrix
            transpose_a: Whether to transpose first matrix
            transpose_b: Whether to transpose second matrix
            scale: Optional scaling factor for result
            
        Returns:
            Matrix multiplication result
        """
        return self.forward(a, b, transpose_a, transpose_b, scale)
        
    def forward(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
        scale: Optional[float] = None,
    ) -> jnp.ndarray:
        """Forward pass computation.
        
        Args:
            a: First input matrix
            b: Second input matrix
            transpose_a: Whether to transpose first matrix
            transpose_b: Whether to transpose second matrix
            scale: Optional scaling factor for result
            
        Returns:
            Matrix multiplication result
        """
        # Cast to bfloat16 if specified
        if self.use_bfloat16:
            a = a.astype(jnp.bfloat16)
            b = b.astype(jnp.bfloat16)
            
        # Handle transpose if needed
        if transpose_a:
            a = jnp.transpose(a)
        if transpose_b:
            b = jnp.transpose(b)
            
        # For 2D matrices, use block_tpu_matmul
        if a.ndim == 2 and b.ndim == 2:
            result = self._block_tpu_matmul(a, b)
        else:
            # For batched matmul, use einsum with optimal layout
            a = optimize_tpu_layout(a)
            b = optimize_tpu_layout(b)
            
            if a.ndim == 3 and b.ndim == 3:
                # Batched matrix multiplication
                result = jnp.einsum('bij,bjk->bik', a, b, precision=self.precision)
            elif a.ndim == 4 and b.ndim == 4:
                # Multi-head attention style matmul
                result = jnp.einsum('bhij,bhjk->bhik', a, b, precision=self.precision)
            else:
                raise ValueError(f"Unsupported shapes for TPU GEMM: {a.shape}, {b.shape}")
        
        # Apply scaling if specified
        if scale is not None:
            result = result * scale
            
        return result

    def backward(
        self,
        grad_output: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        transpose_a: bool = False,
        transpose_b: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Backward pass computation for gradients.
        
        Args:
            grad_output: Gradient with respect to output
            a: First input matrix from forward pass
            b: Second input matrix from forward pass
            transpose_a: Whether first matrix was transposed in forward pass
            transpose_b: Whether second matrix was transposed in forward pass
            
        Returns:
            Gradients for a and b
        """
        # Handle transpose states for gradient computation
        if transpose_a:
            a = jnp.transpose(a)
        if transpose_b:
            b = jnp.transpose(b)
            
        # Cast to bfloat16 if specified
        if self.use_bfloat16:
            grad_output = grad_output.astype(jnp.bfloat16)
            a = a.astype(jnp.bfloat16)
            b = b.astype(jnp.bfloat16)
        
        # Optimize layouts for TPU
        grad_output = optimize_tpu_layout(grad_output)
        a = optimize_tpu_layout(a)
        b = optimize_tpu_layout(b)
        
        # Compute gradients
        if grad_output.ndim == 2:
            # For 2D case
            grad_a = self._block_tpu_matmul(grad_output, jnp.transpose(b))
            grad_b = self._block_tpu_matmul(jnp.transpose(a), grad_output)
        elif grad_output.ndim == 3:
            # For batched matrix multiplication
            grad_a = jnp.einsum('bik,bjk->bij', grad_output, b, precision=self.precision)
            grad_b = jnp.einsum('bij,bik->bjk', a, grad_output, precision=self.precision)
        elif grad_output.ndim == 4:
            # For multi-head attention style matmul
            grad_a = jnp.einsum('bhik,bhjk->bhij', grad_output, b, precision=self.precision)
            grad_b = jnp.einsum('bhij,bhik->bhjk', a, grad_output, precision=self.precision)
        else:
            raise ValueError(f"Unsupported shape for gradients: {grad_output.shape}")
            
        # Reverse any transpose operations for returning gradients
        if transpose_a:
            grad_a = jnp.transpose(grad_a)
        if transpose_b:
            grad_b = jnp.transpose(grad_b)
            
        return grad_a, grad_b
    
    def _block_tpu_matmul(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
    ) -> jnp.ndarray:
        """TPU-optimized blocked matrix multiplication for 2D matrices.
        
        Features:
        - Efficient memory access patterns for TPU HBM
        - Block-level fusion of operations
        - Dynamic padding for optimal TPU utilization
        
        Args:
            A: First input matrix
            B: Second input matrix
            
        Returns:
            Matrix multiplication result
        """
        M, K = A.shape
        K_, N = B.shape
        assert K == K_, f"Incompatible dimensions: {K} != {K_}"

        # Pad dimensions to block_size for TPU efficiency
        M_pad = (self.block_size - M % self.block_size) % self.block_size
        N_pad = (self.block_size - N % self.block_size) % self.block_size
        K_pad = (self.block_size - K % self.block_size) % self.block_size
        
        # Pad inputs for optimal TPU execution
        if M_pad > 0 or K_pad > 0:
            A_padded = pad_to_tpu_multiple(A, self.block_size)
        else:
            A_padded = A
            
        if K_pad > 0 or N_pad > 0:
            B_padded = pad_to_tpu_multiple(B, self.block_size)
        else:
            B_padded = B
        
        # Use TPU-optimized matmul (jax.lax.dot_general is highly optimized on TPU)
        result = jax.lax.dot_general(
            A_padded, 
            B_padded,
            (((1,), (0,)), ((), ())),
            precision=self.precision
        )
        
        # Remove padding
        if M_pad > 0 or N_pad > 0:
            result = result[:M, :N]
            
        return result
        
    @partial(jax.jit, static_argnums=(0,))
    def quantized_matmul(
        self,
        a: jnp.ndarray,
        b: jnp.ndarray,
        a_scale: Optional[jnp.ndarray] = None,
        b_scale: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Perform quantized matrix multiplication for TPU.
        
        Uses dynamic quantization for improved performance while maintaining accuracy.
        
        Args:
            a: First input matrix
            b: Second input matrix
            a_scale: Optional scaling factor for first matrix
            b_scale: Optional scaling factor for second matrix
            
        Returns:
            Matrix multiplication result
        """
        # If scales not provided, compute them dynamically
        if a_scale is None:
            a_abs_max = jnp.max(jnp.abs(a), axis=1, keepdims=True)
            a_scale = jnp.ones_like(a_abs_max)
            a_scale = jnp.where(a_abs_max > 0, 127.0 / (a_abs_max + 1e-5), a_scale)
            
        if b_scale is None:
            b_abs_max = jnp.max(jnp.abs(b), axis=0, keepdims=True)
            b_scale = jnp.ones_like(b_abs_max)
            b_scale = jnp.where(b_abs_max > 0, 127.0 / (b_abs_max + 1e-5), b_scale)
        
        # Quantize inputs
        a_quant = jnp.round(a * a_scale).astype(jnp.int8)
        b_quant = jnp.round(b * b_scale).astype(jnp.int8)
        
        # Perform integer matrix multiplication
        result_int = jnp.matmul(a_quant, b_quant)
        
        # Dequantize the result
        result_scale = 1.0 / (a_scale * b_scale)
        result = result_int.astype(jnp.float32) * result_scale
        
        return result