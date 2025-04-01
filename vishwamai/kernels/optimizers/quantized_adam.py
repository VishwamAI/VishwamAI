"""TPU-optimized quantized Adam implementation."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, NamedTuple, Optional, Tuple, List
from functools import partial

from vishwamai.kernels.tpu.tpu_custom_call import optimize_tpu_layout
from vishwamai.kernels.optimizers.quantization import dynamic_quant, dequantize, fp8_quantize

class QuantizedState(NamedTuple):
    """Quantized optimizer state."""
    params: Dict[str, jnp.ndarray]  # FP32 master copy
    m: Dict[str, jnp.ndarray]       # INT8 momentum
    v: Dict[str, jnp.ndarray]       # INT8 velocity 
    scales_m: Dict[str, jnp.ndarray] # Scales for momentum
    scales_v: Dict[str, jnp.ndarray] # Scales for velocity
    step: jnp.ndarray               # Training step counter
    amax_history: Optional[Dict[str, jnp.ndarray]] = None # For FP8

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
        use_fp8: bool = False,
        block_size: int = 128,
        quantization_bits: int = 8,
        amax_history_length: int = 16,
        recompute_scale_freq: int = 100
    ):
        if block_size % 128 != 0:
            raise ValueError("Block size must be multiple of 128 for TPU")
            
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.pipeline_size = pipeline_size
        self.use_quantization = use_quantization
        self.use_fp8 = use_fp8
        self.block_size = block_size
        self.quantization_bits = quantization_bits
        self.amax_history_length = amax_history_length
        self.recompute_scale_freq = recompute_scale_freq
    
    def init(self, params: Dict[str, jnp.ndarray]) -> QuantizedState:
        """Initialize quantized state with TPU optimizations."""
        # Apply TPU memory layout optimization
        optimized_params = {}
        for k, param in params.items():
            optimized_params[k] = optimize_tpu_layout(param, self.block_size)
        
        if not self.use_quantization:
            return QuantizedState(
                params=optimized_params,
                m={k: jnp.zeros_like(v) for k, v in optimized_params.items()},
                v={k: jnp.zeros_like(v) for k, v in optimized_params.items()},
                scales_m={k: jnp.ones(()) for k in optimized_params},
                scales_v={k: jnp.ones(()) for k in optimized_params},
                step=jnp.zeros((), dtype=jnp.int32)
            )
            
        # Initialize quantized state
        m = {}
        v = {}
        scales_m = {}
        scales_v = {}
        amax_history = {} if self.use_fp8 else None
        
        for k, param in optimized_params.items():
            if self.use_fp8:
                # Initialize FP8 state
                m_quant, amax_m = fp8_quantize(
                    jnp.zeros_like(param),
                    block_size=self.block_size
                )
                v_quant, amax_v = fp8_quantize(
                    jnp.zeros_like(param),
                    block_size=self.block_size
                )
                m[k] = m_quant
                v[k] = v_quant
                amax_history[k] = jnp.stack([amax_m, amax_v])
                scales_m[k] = jnp.ones(())  # Will be updated during training
                scales_v[k] = jnp.ones(())
            else:
                # Initialize INT8 state
                m_quant, scale_m = dynamic_quant(
                    jnp.zeros_like(param),
                    block_size=self.block_size,
                    bits=self.quantization_bits
                )
                v_quant, scale_v = dynamic_quant(
                    jnp.zeros_like(param),
                    block_size=self.block_size,
                    bits=self.quantization_bits
                )
                m[k] = m_quant
                v[k] = v_quant
                scales_m[k] = scale_m
                scales_v[k] = scale_v
        
        return QuantizedState(
            params=optimized_params,
            m=m,
            v=v,
            scales_m=scales_m,
            scales_v=scales_v,
            step=jnp.zeros((), dtype=jnp.int32),
            amax_history=amax_history
        )

    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: QuantizedState,
        grads: Dict[str, jnp.ndarray]
    ) -> QuantizedState:
        """Update parameters with quantized state and pipeline parallelism."""
        step = state.step + 1
        
        # Compute bias correction
        bias_correction1 = 1 - self.beta1 ** step
        bias_correction2 = 1 - self.beta2 ** step
        
        def process_block(
            param: jnp.ndarray,
            grad: jnp.ndarray,
            m: jnp.ndarray,
            v: jnp.ndarray,
            scale_m: jnp.ndarray,
            scale_v: jnp.ndarray,
            amax_history: Optional[jnp.ndarray],
            block_idx: Tuple[int, ...]
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
            # Get current block
            def get_block(tensor: jnp.ndarray) -> jnp.ndarray:
                starts = tuple(i * self.block_size for i in block_idx)
                sizes = tuple(min(self.block_size, s - start)
                            for s, start in zip(tensor.shape, starts))
                return lax.dynamic_slice(tensor, starts, sizes)
            
            param_block = get_block(param)
            grad_block = get_block(grad)
            
            if self.use_fp8:
                # FP8 update path
                m_block, amax_m = fp8_quantize(
                    dequantize(get_block(m), scale_m),
                    amax_history=amax_history[0] if amax_history is not None else None,
                    block_size=self.block_size
                )
                v_block, amax_v = fp8_quantize(
                    dequantize(get_block(v), scale_v),
                    amax_history=amax_history[1] if amax_history is not None else None,
                    block_size=self.block_size
                )
                new_amax = jnp.stack([amax_m, amax_v]) if amax_history is not None else None
            else:
                # INT8 update path
                m_block = dequantize(get_block(m), scale_m)
                v_block = dequantize(get_block(v), scale_v)
                new_amax = None
            
            # Update moments
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
                
            # Update parameter
            new_param = param_block - self.lr * update
            
            # Requantize updated moments
            if self.use_fp8:
                new_m, _ = fp8_quantize(m_block, block_size=self.block_size)
                new_v, _ = fp8_quantize(v_block, block_size=self.block_size)
                new_scale_m = scale_m  # Scales updated elsewhere
                new_scale_v = scale_v
            else:
                new_m, new_scale_m = dynamic_quant(
                    m_block,
                    block_size=self.block_size,
                    bits=self.quantization_bits
                )
                new_v, new_scale_v = dynamic_quant(
                    v_block,
                    block_size=self.block_size,
                    bits=self.quantization_bits
                )
            
            return new_param, new_m, new_v, new_scale_m, new_scale_v, new_amax
        
        # Process all parameters with pipeline parallelism
        new_params = {}
        new_m = {}
        new_v = {}
        new_scales_m = {}
        new_scales_v = {}
        new_amax_history = {} if state.amax_history is not None else None
        
        for param_name in state.params:
            param = state.params[param_name]
            grad = grads[param_name]
            m = state.m[param_name]
            v = state.v[param_name]
            scale_m = state.scales_m[param_name]
            scale_v = state.scales_v[param_name]
            amax_history = state.amax_history[param_name] if state.amax_history else None
            
            # Get shape info for blocking
            shape = param.shape
            if len(shape) >= 2:
                # Process matrices and tensors in blocks
                num_blocks = [
                    (s + self.block_size - 1) // self.block_size
                    for s in shape
                ]
                total_blocks = jnp.prod(jnp.array(num_blocks))
                
                # Process blocks in pipeline
                param_blocks = []
                m_blocks = []
                v_blocks = []
                scale_m_blocks = []
                scale_v_blocks = []
                amax_blocks = []
                
                for block_start in range(0, total_blocks, self.pipeline_size):
                    block_end = min(block_start + self.pipeline_size, total_blocks)
                    block_idxs = []
                    
                    # Convert flat indices to nd indices
                    for flat_idx in range(block_start, block_end):
                        nd_idx = []
                        remaining = flat_idx
                        for block_dim in reversed(num_blocks[1:]):
                            nd_idx.append(remaining % block_dim)
                            remaining = remaining // block_dim
                        nd_idx.append(remaining)
                        block_idxs.append(tuple(reversed(nd_idx)))
                    
                    # Process blocks in parallel
                    results = jax.vmap(lambda idx: process_block(
                        param, grad, m, v, scale_m, scale_v, amax_history, idx
                    ))(jnp.array(block_idxs))
                    
                    # Accumulate results
                    for p, m_, v_, sm, sv, am in results:
                        param_blocks.append(p)
                        m_blocks.append(m_)
                        v_blocks.append(v_)
                        scale_m_blocks.append(sm)
                        scale_v_blocks.append(sv)
                        if am is not None:
                            amax_blocks.append(am)
                
                # Combine blocks
                new_params[param_name] = jnp.concatenate(param_blocks).reshape(shape)
                new_m[param_name] = jnp.concatenate(m_blocks).reshape(shape)
                new_v[param_name] = jnp.concatenate(v_blocks).reshape(shape)
                new_scales_m[param_name] = jnp.concatenate(scale_m_blocks).reshape(num_blocks)
                new_scales_v[param_name] = jnp.concatenate(scale_v_blocks).reshape(num_blocks)
                
                if amax_blocks:
                    new_amax_history[param_name] = jnp.mean(jnp.stack(amax_blocks), axis=0)
            
            else:
                # Process vectors directly
                p, m_, v_, sm, sv, am = process_block(
                    param, grad, m, v, scale_m, scale_v, amax_history, (0,)
                )
                new_params[param_name] = p
                new_m[param_name] = m_
                new_v[param_name] = v_
                new_scales_m[param_name] = sm
                new_scales_v[param_name] = sv
                if am is not None:
                    new_amax_history[param_name] = am
        
        return QuantizedState(
            params=new_params,
            m=new_m,
            v=new_v,
            scales_m=new_scales_m,
            scales_v=new_scales_v,
            step=step,
            amax_history=new_amax_history
        )