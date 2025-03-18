import os
import json
from argparse import ArgumentParser
from glob import glob
from typing import Dict, Any
import jax
import jax.numpy as jnp
from flax import serialization
from flax.training import checkpoints
import numpy as np
from tqdm import tqdm
from safetensors import safe_open
from safetensors.flax import save_file

"""FP8 casting and precision optimization utilities for TPU"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Tuple, Optional
import numpy as np

@partial(jax.jit, static_argnums=(1, 2))
def quantize_to_fp8(
    x: jnp.ndarray,
    num_bits: int = 8,
    block_size: int = 128
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Quantize input tensor to FP8 format.
    
    Args:
        x: Input tensor
        num_bits: Number of bits for quantization
        block_size: Block size for tiling
        
    Returns:
        Tuple of (quantized_tensor, scale_factor)
    """
    # Compute per-block scale factors
    def _get_block_scale(block):
        max_abs = jnp.max(jnp.abs(block))
        scale = (2 ** (num_bits - 1) - 1) / (max_abs + 1e-5)
        return scale
    
    # Process input in blocks
    orig_shape = x.shape
    x_reshaped = x.reshape(-1, block_size)
    num_blocks = x_reshaped.shape[0]
    
    # Compute scales for each block
    scales = jax.vmap(_get_block_scale)(x_reshaped)
    scales = scales.reshape(-1, 1)
    
    # Quantize
    x_quant = jnp.clip(
        jnp.round(x_reshaped * scales),
        -2 ** (num_bits - 1),
        2 ** (num_bits - 1) - 1
    )
    
    # Reshape back to original shape
    x_quant = x_quant.reshape(orig_shape)
    scales = scales.reshape(num_blocks)
    
    return x_quant, scales

@partial(jax.jit, static_argnums=(2,))
def dequantize_from_fp8(
    x_quant: jnp.ndarray,
    scales: jnp.ndarray,
    block_size: int = 128
) -> jnp.ndarray:
    """
    Dequantize FP8 tensor back to original format.
    
    Args:
        x_quant: Quantized tensor
        scales: Scale factors
        block_size: Block size used for quantization
        
    Returns:
        Dequantized tensor
    """
    # Reshape for block processing
    orig_shape = x_quant.shape
    x_reshaped = x_quant.reshape(-1, block_size)
    scales = scales.reshape(-1, 1)
    
    # Dequantize
    x_dequant = x_reshaped / (scales + 1e-5)
    
    # Reshape back
    return x_dequant.reshape(orig_shape)

class DynamicFP8Scaler:
    """Dynamic FP8 scaling with automatic margin adjustment"""
    
    def __init__(
        self,
        init_scale: float = 1.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        max_scale: float = 2**15,
        min_scale: float = 2**-15,
        margin: float = 0.1
    ):
        self.current_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.margin = margin
        self.step = 0
    
    def update_scale(self, is_overflow: bool):
        """Update scale factor based on overflow status"""
        if is_overflow:
            self.current_scale = max(
                self.current_scale * self.backoff_factor,
                self.min_scale
            )
        else:
            self.current_scale = min(
                self.current_scale * self.growth_factor,
                self.max_scale
            )
        self.step += 1
    
    def get_scale(self) -> float:
        """Get current scale factor with margin"""
        return self.current_scale * (1.0 - self.margin)

def optimize_kernel_layout(x: jnp.ndarray) -> jnp.ndarray:
    """Optimize tensor layout for TPU memory access patterns"""
    # Determine optimal layout based on shape
    if x.ndim == 4:  # BHQK format for attention
        return x.transpose((0, 2, 1, 3))  # -> BQHK
    elif x.ndim == 3:  # BLD format for embeddings/outputs
        return x.transpose((1, 0, 2))  # -> LBD
    return x

@partial(jax.jit, static_argnums=(3,))
def convert_precision(
    x: jnp.ndarray,
    from_dtype: Any,
    to_dtype: Any,
    dynamic_scale: bool = True
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """
    Convert tensor precision with optional dynamic scaling.
    
    Args:
        x: Input tensor
        from_dtype: Source precision type
        to_dtype: Target precision type
        dynamic_scale: Whether to use dynamic scaling
        
    Returns:
        Tuple of (converted_tensor, scale_factor)
    """
    if not dynamic_scale:
        return x.astype(to_dtype), None
    
    # Compute scale factor to preserve dynamic range
    max_val = jnp.max(jnp.abs(x))
    target_max = {
        jnp.float16: 2**15,
        jnp.bfloat16: 2**7,
        jnp.float32: 2**126
    }.get(to_dtype, 1.0)
    
    scale = jnp.minimum(target_max / (max_val + 1e-5), 1.0)
    x_scaled = x * scale
    
    return x_scaled.astype(to_dtype), scale

def create_mixed_precision_policy(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create mixed precision training policy"""
    return {
        'compute_dtype': jnp.bfloat16 if config['tpu']['use_bfloat16'] else jnp.float32,
        'output_dtype': jnp.float32,
        'param_dtype': jnp.float32,
        'grad_dtype': jnp.bfloat16 if config['tpu']['use_bfloat16'] else jnp.float32,
        'dynamic_scale': config['optimization'].get('dynamic_scale', True),
        'use_fp8': config['optimization'].get('use_fp8', True),
        'block_size': config['optimization'].get('block_size', 128)
    }

def get_optimal_dtypes(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Get optimal dtype configuration for TPU training"""
    policy = create_mixed_precision_policy(config)
    
    return {
        'model_dtype': policy['compute_dtype'],
        'optimizer_dtype': policy['param_dtype'],
        'attention': {
            'qk_dtype': jnp.bfloat16,  # QK attention always in BF16
            'v_dtype': policy['compute_dtype'],
            'output_dtype': policy['compute_dtype']
        },
        'mlp': {
            'intermediate_dtype': policy['compute_dtype'],
            'output_dtype': policy['compute_dtype']
        },
        'embedding': policy['param_dtype'],
        'layernorm': policy['param_dtype']
    }

def check_overflow(
    x: jnp.ndarray,
    max_val: float = 2**15
) -> bool:
    """Check for overflow in tensor values"""
    return bool(jnp.any(jnp.abs(x) > max_val))

class FP8CastManager:
    """Manage FP8 casting and precision conversions"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        dynamic_scale: bool = True
    ):
        self.config = config
        self.dynamic_scale = dynamic_scale
        self.policy = create_mixed_precision_policy(config)
        self.scaler = DynamicFP8Scaler() if dynamic_scale else None
    
    def cast_params(
        self,
        params: Dict[str, Any],
        to_dtype: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Cast model parameters to appropriate precision"""
        if to_dtype is None:
            to_dtype = self.policy['param_dtype']
            
        def _cast(param):
            if param.dtype != to_dtype:
                return param.astype(to_dtype)
            return param
            
        return jax.tree_map(_cast, params)
    
    def cast_for_compute(
        self,
        x: jnp.ndarray,
        compute_type: str = 'attention'
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Cast input for compute operations"""
        target_dtype = {
            'attention': self.policy['compute_dtype'],
            'mlp': self.policy['compute_dtype'],
            'output': self.policy['output_dtype']
        }.get(compute_type, self.policy['compute_dtype'])
        
        if self.policy['use_fp8']:
            return quantize_to_fp8(
                x,
                block_size=self.policy['block_size']
            )
        else:
            return convert_precision(
                x,
                x.dtype,
                target_dtype,
                self.dynamic_scale
            )
    
    def update_scaling(self, tensors: Dict[str, jnp.ndarray]):
        """Update dynamic scaling factors"""
        if not self.dynamic_scale or self.scaler is None:
            return
            
        # Check for overflow in any tensor
        has_overflow = any(
            check_overflow(t) for t in tensors.values()
        )
        
        self.scaler.update_scale(has_overflow)

def dynamic_scale_finder(
    x: jnp.ndarray,
    block_size: int = 128,
    margin_factor: float = 0.01
) -> jnp.ndarray:
    """Find optimal dynamic scaling factor for FP8 casting."""
    # Compute max absolute value in blocks
    x_reshape = x.reshape(-1, block_size)
    max_abs = jnp.max(jnp.abs(x_reshape), axis=1, keepdims=True)
    
    # Add margin to avoid overflow
    scale = (127.0 - margin_factor) / (max_abs + 1e-10)
    
    # Broadcast scale back to original shape
    scale = scale.reshape(x.shape[0], 1, 1, 1)
    return scale

def fp8_cast(
    x: jnp.ndarray,
    scale: Optional[jnp.ndarray] = None,
    block_size: int = 128
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Cast input to FP8 format with dynamic scaling."""
    if scale is None:
        scale = dynamic_scale_finder(x, block_size)
        
    # Scale and round to int8
    x_scaled = x * scale
    x_int8 = jnp.clip(jnp.round(x_scaled), -127, 127).astype(jnp.int8)
    
    # Convert back to original dtype
    x_fp8 = (x_int8.astype(x.dtype) / scale).astype(x.dtype)
    
    return x_fp8, scale

def optimize_kernel_layout(
    x: jnp.ndarray,
    mesh: Optional[Any] = None,
    device_strategy: str = 'data_parallel'
) -> jnp.ndarray:
    """Optimize tensor layout for TPU kernel operations."""
    # Reshape for optimal TPU memory layout
    shape = x.shape
    
    if len(shape) == 4:  # BHQK layout for attention
        B, H, Q, K = shape
        x = x.reshape(B, H, Q // 32, 32, K // 32, 32)
        x = x.transpose(0, 2, 4, 1, 3, 5)
        x = x.reshape(B, Q, K, H)
    elif len(shape) == 3:  # BLD layout for embeddings
        B, L, D = shape
        x = x.reshape(B, L // 32, 32, D // 32, 32)
        x = x.transpose(0, 1, 3, 2, 4)
        x = x.reshape(B, L, D)
        
    return x

@partial(jax.jit, static_argnums=(1,))
def matmul_fn8_tpu(
    x: jnp.ndarray,
    block_size: int,
    y: jnp.ndarray,
    scale_x: Optional[jnp.ndarray] = None,
    scale_y: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Optimized matrix multiplication with FP8 casting for TPU."""
    # Cast inputs to FP8
    x_fp8, scale_x = fp8_cast(x, scale_x, block_size)
    y_fp8, scale_y = fp8_cast(y, scale_y, block_size)
    
    # Optimize memory layout
    x_fp8 = optimize_kernel_layout(x_fp8)
    y_fp8 = optimize_kernel_layout(y_fp8)
    
    # Compute matrix multiplication
    out = jax.lax.dot_general(
        x_fp8,
        y_fp8,
        (((x_fp8.ndim - 1,), (0,)), ((), ()))
    )
    
    # Apply scales
    if scale_x is not None and scale_y is not None:
        out = out / (scale_x * scale_y)
        
    return out

def create_fp8_config(
    model_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Create FP8 configuration for model training."""
    return {
        "block_size": model_config.get("block_size", 128),
        "margin_factor": model_config.get("fp8_margin", 0.01),
        "mixed_precision": model_config.get("mixed_precision", True),
        "dynamic_scale": model_config.get("dynamic_scale", True),
        "use_fp8_attention": True,
        "use_fp8_mlp": True
    }

def convert_grads_fp8(
    grads: Dict[str, jnp.ndarray],
    scales: Dict[str, jnp.ndarray],
    config: Dict[str, Any]
) -> Dict[str, jnp.ndarray]:
    """Convert gradients to FP8 for efficient all-reduce."""
    fp8_grads = {}
    for k, g in grads.items():
        if g is not None:
            g_fp8, _ = fp8_cast(
                g,
                scale=scales.get(k),
                block_size=config["block_size"]
            )
            fp8_grads[k] = g_fp8
        else:
            fp8_grads[k] = None
            
    return fp8_grads

def fp8_cast_to_bf16(x: jnp.ndarray) -> jnp.ndarray:
    """Cast FP8 tensor to BF16 format.
    
    Args:
        x: Input tensor in FP8 format
        
    Returns:
        Tensor cast to BF16
    """
    if x.dtype == jnp.bfloat16:
        return x
    
    # First convert to FP32 for better precision in intermediate calculations
    x_fp32 = x.astype(jnp.float32)
    
    # Get the dynamic scale
    scale = dynamic_scale_finder(x_fp32)
    
    # Cast to BF16
    return (x_fp32 * scale).astype(jnp.bfloat16)

class FP8Scaler:
    """Dynamic loss scaler for mixed FP8 training."""
    
    def __init__(
        self,
        initial_scale: float = 2**15,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        max_scale: float = 2**24
    ):
        self.scale = initial_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.max_scale = max_scale
        self.step = 0
        self.growth_tracker = 0
        
    def __call__(self, loss: jnp.ndarray) -> jnp.ndarray:
        """Scale loss for mixed precision training."""
        return loss * self.scale
        
    def update(self, overflow: bool):
        """Update scaling factor based on gradient overflow."""
        if overflow:
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self.growth_tracker = 0
        else:
            self.growth_tracker += 1
            if self.growth_tracker == self.growth_interval:
                self.scale = min(self.scale * self.growth_factor, self.max_scale)
                self.growth_tracker = 0
        self.step += 1
        
def create_fp8_functions(config: Dict[str, Any]):
    """Create optimized FP8 functions for model training."""
    
    # Attention FP8 matmul
    @partial(jax.jit, static_argnums=(3,))
    def attention_fp8_matmul(q, k, v, num_heads):
        # Split heads
        B, L, D = q.shape
        q = q.reshape(B, L, num_heads, -1)
        k = k.reshape(B, L, num_heads, -1)
        v = v.reshape(B, L, num_heads, -1)
        
        # Cast to FP8 and compute attention
        scores = matmul_fn8_tpu(
            q.transpose(0, 2, 1, 3),  # [B, H, L, D]
            k.transpose(0, 2, 3, 1),  # [B, H, D, L]
            config["block_size"]
        )
        
        # Scale and softmax
        scores = scores / jnp.sqrt(D // num_heads)
        attn = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        out = matmul_fn8_tpu(
            attn,
            v.transpose(0, 2, 1, 3),  # [B, H, L, D]
            config["block_size"]
        )
        
        # Restore shape
        return out.transpose(0, 2, 1, 3).reshape(B, L, D)
    
    # MLP FP8 matmul
    @partial(jax.jit, static_argnums=(2,))
    def mlp_fp8_matmul(x, w, block_size):
        return matmul_fn8_tpu(x, w, block_size)
        
    return {
        "attention_matmul": attention_fp8_matmul,
        "mlp_matmul": mlp_fp8_matmul,
        "cast": fp8_cast,
        "optimize_layout": optimize_kernel_layout
    }

def weight_dequant(weight: jnp.ndarray, scale_inv: jnp.ndarray) -> jnp.ndarray:
    """
    Dequantizes FP8 weights using scale inverse.
    
    Args:
        weight (jnp.ndarray): The FP8 weight tensor stored as uint8
        scale_inv (jnp.ndarray): The inverse scale tensor
        
    Returns:
        jnp.ndarray: The dequantized weight tensor in bfloat16
    """
    # Convert uint8 (FP8) to float32 first, then apply scale
    weight_float = weight.astype(jnp.float32)
    # Apply the inverse scale (element-wise division)
    dequantized = weight_float / scale_inv
    # Convert to bfloat16
    return dequantized.astype(jnp.bfloat16)

def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    
    weight_map = model_index["weight_map"]
    
    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            jnp.ndarray: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            # Open the safetensor file in read-only mode
            loaded_files[file_name] = {}
            with safe_open(file_path, framework="flax") as f:
                for key in f.keys():
                    # Load tensor to device
                    loaded_files[file_name][key] = jax.device_put(f.get_tensor(key))
        
        return loaded_files[file_name][tensor_name]

    # Get all safetensor files in the directory
    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        
        # Load the current safetensor file
        current_state_dict = {}
        with safe_open(safetensor_file, framework="flax") as f:
            for key in f.keys():
                current_state_dict[key] = jax.device_put(f.get_tensor(key))
        
        loaded_files[file_name] = current_state_dict
        
        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            # Check if this is an FP8 weight (element size 1 byte)
            elif weight.dtype == jnp.uint8:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(weight_name)
                    new_state_dict[weight_name] = weight_dequant(weight, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
        
        # Save the converted state dict
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            jax.clear_caches()  # Clear JAX caches
    
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True,
                        help="Path to the directory containing FP8 weights")
    parser.add_argument("--output-bf16-hf-path", type=str, required=True,
                        help="Path where converted BF16 weights will be saved")
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)
