"""
Utility functions and helpers for VishwamAI.

Provides common utilities for model configuration, data processing,
hardware detection, and performance optimization.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Dict, Any, Optional, List, Tuple, Union
import chex
import time
import os
import json
from dataclasses import asdict, is_dataclass

from .model import ModelConfig
from .training import TrainingConfig


def get_hardware_info() -> Dict[str, Any]:
    """Get information about available hardware."""
    
    devices = jax.devices()
    
    info = {
        'num_devices': len(devices),
        'device_types': [d.platform for d in devices],
        'device_names': [str(d) for d in devices],
        'has_tpu': any(d.platform == 'tpu' for d in devices),
        'has_gpu': any(d.platform == 'gpu' for d in devices),
        'has_cpu': any(d.platform == 'cpu' for d in devices),
    }
    
    # TPU-specific info
    if info['has_tpu']:
        info['tpu_topology'] = jax.devices()[0].coords if hasattr(jax.devices()[0], 'coords') else None
    
    # GPU-specific info
    if info['has_gpu']:
        try:
            import pynvml
            pynvml.nvmlInit()
            gpu_count = pynvml.nvmlDeviceGetCount()
            gpu_info = []
            
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                gpu_info.append({
                    'name': name,
                    'memory_total': memory_info.total // (1024**3),  # GB
                    'memory_free': memory_info.free // (1024**3),   # GB
                })
            
            info['gpu_details'] = gpu_info
        except ImportError:
            info['gpu_details'] = "pynvml not available"
    
    return info


def estimate_memory_usage(config: ModelConfig, batch_size: int = 1, seq_len: int = 2048) -> Dict[str, float]:
    """Estimate memory usage for a given model configuration."""
    
    # Parameter count estimation
    vocab_params = config.vocab_size * config.dim
    embed_params = config.max_seq_len * config.dim if not config.use_rotary_embeddings else 0
    
    # Transformer parameters per layer
    attn_params = (
        4 * config.dim * config.dim +  # Q, K, V, O projections
        2 * config.dim if config.use_rmsnorm else 2 * config.dim  # Layer norms
    )
    
    if config.use_moe:
        ff_params = config.expert_count * (2 * config.dim * config.dim * 4)  # MoE FFN
        ff_params += config.dim * config.expert_count  # Router
    else:
        ff_params = 2 * config.dim * config.dim * 4  # Standard FFN
    
    layer_params = attn_params + ff_params
    total_params = vocab_params + embed_params + (config.depth * layer_params)
    
    # Memory usage (in GB)
    param_memory = total_params * 4 / (1024**3)  # FP32 parameters
    if config.use_bfloat16:
        param_memory = total_params * 2 / (1024**3)  # BF16 parameters
    
    # Activation memory (rough estimate)
    activation_memory = batch_size * seq_len * config.dim * config.depth * 8 / (1024**3)  # FP32 activations
    if config.use_bfloat16:
        activation_memory /= 2  # BF16 activations
    
    # Gradient memory (same as parameters for full fine-tuning)
    gradient_memory = param_memory
    
    # Optimizer state memory (AdamW: 2x parameters)
    optimizer_memory = param_memory * 2
    
    return {
        'parameters_gb': param_memory,
        'activations_gb': activation_memory,
        'gradients_gb': gradient_memory,
        'optimizer_gb': optimizer_memory,
        'total_gb': param_memory + activation_memory + gradient_memory + optimizer_memory,
        'inference_gb': param_memory + activation_memory,
        'parameter_count': total_params
    }


def create_optimizer(
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    gradient_clip_norm: float = 1.0,
    optimizer_type: str = "adamw"
) -> optax.GradientTransformation:
    """Create an optimizer with common configurations."""
    
    # Learning rate schedule
    if warmup_steps > 0:
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=total_steps - warmup_steps,
            end_value=learning_rate * 0.1
        )
    else:
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps,
            alpha=0.1
        )
    
    # Optimizer choice
    if optimizer_type == "adamw":
        optimizer = optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=weight_decay
        )
    elif optimizer_type == "adam":
        optimizer = optax.adam(
            learning_rate=schedule,
            b1=0.9,
            b2=0.999,
            eps=1e-8
        )
    elif optimizer_type == "sgd":
        optimizer = optax.sgd(
            learning_rate=schedule,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Add gradient clipping
    if gradient_clip_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(gradient_clip_norm),
            optimizer
        )
    
    return optimizer


def setup_mixed_precision(hardware_type: Optional[str] = None) -> Dict[str, Any]:
    """Setup mixed precision configuration based on hardware."""
    
    if hardware_type is None:
        hardware_info = get_hardware_info()
        if hardware_info['has_tpu']:
            hardware_type = 'tpu'
        elif hardware_info['has_gpu']:
            hardware_type = 'gpu'
        else:
            hardware_type = 'cpu'
    
    config = {
        'hardware_type': hardware_type,
        'use_mixed_precision': False,
        'dtype': jnp.float32,
        'param_dtype': jnp.float32,
        'loss_scale': 1.0,
    }
    
    if hardware_type == 'tpu':
        config.update({
            'use_mixed_precision': True,
            'dtype': jnp.bfloat16,
            'param_dtype': jnp.float32,  # Keep params in FP32
            'loss_scale': 1.0,  # No loss scaling needed for bfloat16
        })
    elif hardware_type == 'gpu':
        config.update({
            'use_mixed_precision': True,
            'dtype': jnp.float16,
            'param_dtype': jnp.float32,  # Keep params in FP32
            'loss_scale': 2.0**15,  # Loss scaling for FP16
        })
    
    return config


def count_parameters(params: Dict[str, Any]) -> int:
    """Count the total number of parameters in a model."""
    
    def count_nested(x):
        if isinstance(x, dict):
            return sum(count_nested(v) for v in x.values())
        elif isinstance(x, (list, tuple)):
            return sum(count_nested(v) for v in x)
        elif hasattr(x, 'size'):
            return x.size
        else:
            return 0
    
    return count_nested(params)


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """Format large numbers with appropriate suffixes."""
    
    if num >= 1e12:
        return f"{num/1e12:.{precision}f}T"
    elif num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def benchmark_model(
    model: Any,
    params: Dict[str, Any],
    input_shape: Tuple[int, ...],
    num_runs: int = 10,
    warmup_runs: int = 3
) -> Dict[str, float]:
    """Benchmark model inference performance."""
    
    # Create dummy input
    dummy_input = jnp.ones(input_shape, dtype=jnp.int32)
    
    # JIT compile
    @jax.jit
    def forward_fn(params, x):
        return model.apply(params, x, training=False)
    
    # Warmup
    for _ in range(warmup_runs):
        _ = forward_fn(params, dummy_input)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        output = forward_fn(params, dummy_input)
        output.block_until_ready()  # Ensure computation is complete
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': float(jnp.mean(jnp.array(times))),
        'std_time': float(jnp.std(jnp.array(times))),
        'min_time': float(jnp.min(jnp.array(times))),
        'max_time': float(jnp.max(jnp.array(times))),
        'throughput_samples_per_sec': input_shape[0] / float(jnp.mean(jnp.array(times))),
    }


def save_config(config: Union[ModelConfig, TrainingConfig], filepath: str):
    """Save configuration to JSON file."""
    
    if is_dataclass(config):
        config_dict = asdict(config)
    else:
        config_dict = config.__dict__
    
    # Handle non-JSON serializable objects
    def make_serializable(obj):
        if isinstance(obj, (jnp.ndarray, jnp.DeviceArray)):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif callable(obj):
            return str(obj)
        else:
            return obj
    
    # Convert nested structures
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, (list, tuple)):
            return [convert_dict(v) for v in d]
        else:
            return make_serializable(d)
    
    serializable_config = convert_dict(config_dict)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(serializable_config, f, indent=2)


def load_config(filepath: str, config_class: type) -> Union[ModelConfig, TrainingConfig]:
    """Load configuration from JSON file."""
    
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    
    # Convert back to dataclass if applicable
    if is_dataclass(config_class):
        return config_class(**config_dict)
    else:
        config = config_class()
        for key, value in config_dict.items():
            setattr(config, key, value)
        return config


def create_attention_mask(
    input_ids: chex.Array,
    pad_token_id: int = 0,
    causal: bool = True
) -> chex.Array:
    """Create attention mask for input sequences."""
    
    batch_size, seq_len = input_ids.shape
    
    # Padding mask (1 for real tokens, 0 for padding)
    padding_mask = (input_ids != pad_token_id).astype(jnp.float32)
    
    if causal:
        # Causal mask (lower triangular)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        
        # Combine masks
        mask = padding_mask[:, None, None, :] * causal_mask[None, None, :, :]
    else:
        mask = padding_mask[:, None, None, :]
    
    return mask


def get_model_size_recommendation(available_memory_gb: float) -> Dict[str, Any]:
    """Get model size recommendations based on available memory."""
    
    recommendations = []
    
    # Small model (1-2B parameters)
    small_config = ModelConfig(
        dim=2048,
        depth=24,
        heads=32,
        vocab_size=50304
    )
    small_memory = estimate_memory_usage(small_config)
    
    if available_memory_gb >= small_memory['inference_gb']:
        recommendations.append({
            'name': 'Small (1.5B parameters)',
            'config': small_config,
            'memory_usage': small_memory,
            'description': 'Good for experimentation and fast inference'
        })
    
    # Medium model (7B parameters)
    medium_config = ModelConfig(
        dim=4096,
        depth=32,
        heads=32,
        vocab_size=50304
    )
    medium_memory = estimate_memory_usage(medium_config)
    
    if available_memory_gb >= medium_memory['inference_gb']:
        recommendations.append({
            'name': 'Medium (7B parameters)',
            'config': medium_config,
            'memory_usage': medium_memory,
            'description': 'Balanced performance and efficiency'
        })
    
    # Large model (20B parameters)
    large_config = ModelConfig(
        dim=8192,
        depth=40,
        heads=64,
        vocab_size=50304
    )
    large_memory = estimate_memory_usage(large_config)
    
    if available_memory_gb >= large_memory['inference_gb']:
        recommendations.append({
            'name': 'Large (20B parameters)',
            'config': large_config,
            'memory_usage': large_memory,
            'description': 'High performance for demanding tasks'
        })
    
    return {
        'available_memory_gb': available_memory_gb,
        'recommendations': recommendations
    }


def print_model_info(config: ModelConfig):
    """Print detailed model information."""
    
    memory_usage = estimate_memory_usage(config)
    param_count = memory_usage['parameter_count']
    
    print("=" * 60)
    print("VishwamAI Model Configuration")
    print("=" * 60)
    print(f"Model Dimensions:")
    print(f"  Hidden Size: {config.dim}")
    print(f"  Number of Layers: {config.depth}")
    print(f"  Attention Heads: {config.heads}")
    print(f"  Head Dimension: {config.head_dim}")
    print(f"  Vocabulary Size: {format_number(config.vocab_size)}")
    print(f"  Max Sequence Length: {config.max_seq_len}")
    
    print(f"\nArchitecture Features:")
    print(f"  Flash Attention: {config.use_flash_attention}")
    print(f"  Grouped Query Attention: {config.use_grouped_query_attention}")
    print(f"  RMS Normalization: {config.use_rmsnorm}")
    print(f"  Rotary Embeddings: {config.use_rotary_embeddings}")
    print(f"  Mixture of Experts: {config.use_moe}")
    print(f"  Gradient Checkpointing: {config.gradient_checkpointing}")
    
    print(f"\nModel Size:")
    print(f"  Total Parameters: {format_number(param_count)}")
    print(f"  Inference Memory: {memory_usage['inference_gb']:.2f} GB")
    print(f"  Training Memory: {memory_usage['total_gb']:.2f} GB")
    
    print("=" * 60)


def validate_config(config: ModelConfig) -> List[str]:
    """Validate model configuration and return list of issues."""
    
    issues = []
    
    # Check basic constraints
    if config.dim % config.heads != 0:
        issues.append(f"dim ({config.dim}) must be divisible by heads ({config.heads})")
    
    if config.use_grouped_query_attention and config.heads % config.gqa_groups != 0:
        issues.append(f"heads ({config.heads}) must be divisible by gqa_groups ({config.gqa_groups})")
    
    if config.max_seq_len <= 0:
        issues.append(f"max_seq_len must be positive, got {config.max_seq_len}")
    
    if config.dropout_rate < 0 or config.dropout_rate >= 1:
        issues.append(f"dropout_rate must be in [0, 1), got {config.dropout_rate}")
    
    # Check MoE constraints
    if config.use_moe:
        if config.expert_count <= 0:
            issues.append(f"expert_count must be positive when using MoE, got {config.expert_count}")
        
        if config.expert_capacity <= 0:
            issues.append(f"expert_capacity must be positive when using MoE, got {config.expert_capacity}")
    
    return issues
