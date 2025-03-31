"""TPU-optimized quantized AdamW implementation."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, NamedTuple, Optional, Tuple
from functools import partial

class QuantizedState(NamedTuple):
    """Quantized optimizer state."""
    params: Dict[str, jnp.ndarray]  # FP32 master copy
    m: Dict[str, jnp.ndarray]       # INT8 momentum 
    v: Dict[str, jnp.ndarray]       # INT8 velocity
    scales_m: Dict[str, jnp.ndarray] # Scales for m
    scales_v: Dict[str, jnp.ndarray] # Scales for v
    step: jnp.ndarray

class TPUQuantizedAdamW:
    """TPU-optimized AdamW with 8-bit state and pipelined updates."""
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        pipeline_size: int = 4,
        use_quantization: bool = True,
        block_size: int = 128
    ):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.pipeline_size = pipeline_size
        self.use_quantization = use_quantization
        self.block_size = block_size
        
    def init(self, params: Dict[str, jnp.ndarray]) -> QuantizedState:
        """Initialize quantized state."""
        if not self.use_quantization:
            return QuantizedState(
                params=params,
                m={k: jnp.zeros_like(v) for k, v in params.items()},
                v={k: jnp.zeros_like(v) for k, v in params.items()},
                scales_m={k: jnp.ones(()) for k in params},
                scales_v={k: jnp.ones(()) for k in params},
                step=jnp.zeros((), dtype=jnp.int32)
            )
            
        # Initialize quantized state
        m = {}
        v = {}
        scales_m = {}
        scales_v = {}
        
        for k, param in params.items():
            # Calculate initial scaling factors
            abs_max = jnp.max(jnp.abs(param))
            scale_m = abs_max / 127.0
            scale_v = abs_max * abs_max / 127.0
            
            # Initialize quantized momentum/velocity
            m[k] = jnp.zeros_like(param, dtype=jnp.int8)
            v[k] = jnp.zeros_like(param, dtype=jnp.int8)
            scales_m[k] = scale_m
            scales_v[k] = scale_v
            
        return QuantizedState(
            params=params,
            m=m,
            v=v, 
            scales_m=scales_m,
            scales_v=scales_v,
            step=jnp.zeros((), dtype=jnp.int32)
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: QuantizedState,
        grads: Dict[str, jnp.ndarray]
    ) -> QuantizedState:
        """Update parameters with quantized state."""
        
        # Update step count
        step = state.step + 1
        
        # Compute bias correction
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        
        def update_param_block(param_state, block_idx):
            param, grad, m, v, scale_m, scale_v = param_state
            
            # Get current block
            start_idx = block_idx * self.block_size
            end_idx = min(start_idx + self.block_size, param.size)
            param_block = param.reshape(-1)[start_idx:end_idx]
            grad_block = grad.reshape(-1)[start_idx:end_idx]
            
            if self.use_quantization:
                # Dequantize momentum/velocity
                m_block = (m.reshape(-1)[start_idx:end_idx] * scale_m).astype(param.dtype)
                v_block = (v.reshape(-1)[start_idx:end_idx] * scale_v).astype(param.dtype)
                
                # Update momentums
                m_block = self.beta1 * m_block + (1 - self.beta1) * grad_block
                v_block = self.beta2 * v_block + (1 - self.beta2) * (grad_block * grad_block)
                
                # Bias correction
                m_hat = m_block / bias_correction1
                v_hat = v_block / bias_correction2
                
                # Compute update
                update = m_hat / (jnp.sqrt(v_hat) + self.eps)
                
                # Apply weight decay
                if self.weight_decay > 0:
                    update = update + self.weight_decay * param_block
                    
                # Update parameters
                param_block = param_block - self.lr * update
                
                # Requantize momentum/velocity
                new_scale_m = jnp.max(jnp.abs(m_block)) / 127.0
                new_scale_v = jnp.max(jnp.abs(v_block)) / 127.0
                m_block = jnp.clip(
                    jnp.round(m_block / new_scale_m),
                    -127, 127
                ).astype(jnp.int8)
                v_block = jnp.clip(
                    jnp.round(v_block / new_scale_v),
                    -127, 127
                ).astype(jnp.int8)
            else:
                # Regular AdamW update
                m_block = self.beta1 * m + (1 - self.beta1) * grad_block
                v_block = self.beta2 * v + (1 - self.beta2) * (grad_block * grad_block)
                
                # Bias correction
                m_hat = m_block / bias_correction1
                v_hat = v_block / bias_correction2
                
                # Compute update
                update = m_hat / (jnp.sqrt(v_hat) + self.eps)
                
                # Apply weight decay
                if self.weight_decay > 0:
                    update = update + self.weight_decay * param_block
                    
                # Update parameters
                param_block = param_block - self.lr * update
                
                new_scale_m = scale_m
                new_scale_v = scale_v
                
            return (param_block, m_block, v_block, new_scale_m, new_scale_v)
            
        # Process parameters in parallel pipeline
        new_params = {}
        new_m = {}
        new_v = {} 
        new_scales_m = {}
        new_scales_v = {}
        
        for param_name in state.params:
            param = state.params[param_name]
            grad = grads[param_name]
            m = state.m[param_name] 
            v = state.v[param_name]
            scale_m = state.scales_m[param_name]
            scale_v = state.scales_v[param_name]
            
            # Split into blocks for pipeline parallel processing
            num_blocks = (param.size + self.block_size - 1) // self.block_size
            
            # Process blocks in pipeline
            block_results = []
            for block_idx in range(0, num_blocks, self.pipeline_size):
                # Process block range
                end_idx = min(block_idx + self.pipeline_size, num_blocks)
                block_range = jnp.arange(block_idx, end_idx)
                
                # Update blocks in parallel
                results = jax.vmap(
                    partial(update_param_block, (param, grad, m, v, scale_m, scale_v))
                )(block_range)
                
                block_results.append(results)
                
            # Combine results
            param_blocks = []
            m_blocks = []
            v_blocks = []
            scale_m_blocks = []
            scale_v_blocks = []
            
            for result in block_results:
                p, m, v, sm, sv = result
                param_blocks.append(p)
                m_blocks.append(m)
                v_blocks.append(v)
                scale_m_blocks.append(sm)
                scale_v_blocks.append(sv)
                
            # Reshape back to original shape
            new_params[param_name] = jnp.concatenate(param_blocks).reshape(param.shape)
            new_m[param_name] = jnp.concatenate(m_blocks).reshape(param.shape)
            new_v[param_name] = jnp.concatenate(v_blocks).reshape(param.shape)
            new_scales_m[param_name] = jnp.mean(jnp.stack(scale_m_blocks))
            new_scales_v[param_name] = jnp.mean(jnp.stack(scale_v_blocks))
            
        return QuantizedState(
            params=new_params,
            m=new_m,
            v=new_v,
            scales_m=new_scales_m,
            scales_v=new_scales_v,
            step=step
        )