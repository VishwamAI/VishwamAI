"""TPU-optimized Lion optimizer with dynamic quantization."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, NamedTuple, Optional, Tuple
from functools import partial

class QuantizedLionState(NamedTuple):
    """Quantized Lion optimizer state."""
    params: Dict[str, jnp.ndarray]
    momentum: Dict[str, jnp.ndarray]  # INT8 momentum
    scales: Dict[str, jnp.ndarray]    # Momentum scales
    step: jnp.ndarray

class TPUQuantizedLion:
    """TPU-optimized Lion with dynamic block sizing and quantization."""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        use_quantization: bool = True,
        min_block_size: int = 128,
        max_block_size: int = 2048,
        pipeline_depth: int = 4
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.use_quantization = use_quantization
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.pipeline_depth = pipeline_depth
        
    def init(self, params: Dict[str, jnp.ndarray]) -> QuantizedLionState:
        """Initialize quantized state."""
        momentum = {}
        scales = {}
        
        for k, param in params.items():
            if self.use_quantization:
                # Initialize quantized momentum
                scale = jnp.max(jnp.abs(param)) / 127.0
                momentum[k] = jnp.zeros_like(param, dtype=jnp.int8)
                scales[k] = scale
            else:
                momentum[k] = jnp.zeros_like(param)
                scales[k] = jnp.ones(())
                
        return QuantizedLionState(
            params=params,
            momentum=momentum,
            scales=scales,
            step=jnp.zeros((), dtype=jnp.int32)
        )
        
    def _compute_block_size(self, param_size: int) -> int:
        """Compute optimal block size based on parameter size."""
        # Use larger blocks for larger parameters
        block_size = min(
            max(
                self.min_block_size,
                param_size // (8 * self.pipeline_depth)
            ),
            self.max_block_size
        )
        # Round to nearest multiple of 128 for TPU efficiency
        return ((block_size + 127) // 128) * 128
        
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: QuantizedLionState,
        grads: Dict[str, jnp.ndarray],
        rng_key: Optional[jnp.ndarray] = None
    ) -> QuantizedLionState:
        """Update parameters with quantized state."""
        
        step = state.step + 1
        
        def update_param_block(param_state, block_idx):
            param, grad, momentum, scale = param_state
            
            # Get current block
            block_size = self._compute_block_size(param.size)
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, param.size)
            
            param_block = param.reshape(-1)[start_idx:end_idx]
            grad_block = grad.reshape(-1)[start_idx:end_idx]
            
            if self.use_quantization:
                # Dequantize momentum
                momentum_block = (
                    momentum.reshape(-1)[start_idx:end_idx] * scale
                ).astype(param.dtype)
                
                # Update momentum with sign
                update = jnp.sign(
                    self.beta1 * momentum_block + 
                    (1 - self.beta1) * grad_block
                )
                
                # Update momentum
                new_momentum = (
                    self.beta2 * momentum_block +
                    (1 - self.beta2) * grad_block
                )
                
                # Compute new scale for quantization
                new_scale = jnp.max(jnp.abs(new_momentum)) / 127.0
                
                # Quantize new momentum
                new_momentum = jnp.clip(
                    jnp.round(new_momentum / new_scale),
                    -127, 127
                ).astype(jnp.int8)
            else:
                momentum_block = momentum.reshape(-1)[start_idx:end_idx]
                
                # Regular Lion update
                update = jnp.sign(
                    self.beta1 * momentum_block +
                    (1 - self.beta1) * grad_block
                )
                
                new_momentum = (
                    self.beta2 * momentum_block +
                    (1 - self.beta2) * grad_block
                )
                new_scale = scale
            
            # Apply weight decay
            if self.weight_decay > 0:
                param_block = param_block * (1 - self.lr * self.weight_decay)
            
            # Update parameters
            new_param = param_block - self.lr * update
            
            return (new_param, new_momentum, new_scale)
            
        # Process parameters in parallel pipeline
        new_params = {}
        new_momentum = {}
        new_scales = {}
        
        for param_name in state.params:
            param = state.params[param_name]
            grad = grads[param_name]
            momentum = state.momentum[param_name]
            scale = state.scales[param_name]
            
            # Split into blocks for pipeline parallel processing
            block_size = self._compute_block_size(param.size)
            num_blocks = (param.size + block_size - 1) // block_size
            
            # Process blocks in pipeline
            block_results = []
            for pipeline_start in range(0, num_blocks, self.pipeline_depth):
                # Process block range in parallel
                pipeline_end = min(pipeline_start + self.pipeline_depth, num_blocks)
                block_range = jnp.arange(pipeline_start, pipeline_end)
                
                # Update blocks
                results = jax.vmap(
                    partial(update_param_block, (param, grad, momentum, scale))
                )(block_range)
                
                block_results.append(results)
                
            # Combine results
            param_blocks = []
            momentum_blocks = []
            scale_blocks = []
            
            for result in block_results:
                p, m, s = result
                param_blocks.append(p)
                momentum_blocks.append(m)
                scale_blocks.append(s)
                
            # Reshape back to original shape
            new_params[param_name] = jnp.concatenate(param_blocks).reshape(param.shape)
            new_momentum[param_name] = jnp.concatenate(momentum_blocks).reshape(param.shape)
            new_scales[param_name] = jnp.mean(jnp.stack(scale_blocks))
            
        return QuantizedLionState(
            params=new_params,
            momentum=new_momentum,
            scales=new_scales,
            step=step
        )