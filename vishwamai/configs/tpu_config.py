"""
TPU configuration and initialization utilities with explicit dtype management
"""

import jax
import jax.numpy as jnp
import os
from typing import Dict, Any, Optional, Union
import warnings

class TPUConfig:
    """TPU configuration management with fallback to CPU/GPU"""
    
    _instance = None
    _initialized = False
    _platform = None
    _compute_dtype = jnp.bfloat16
    _embedding_dtype = jnp.int32  # Explicit dtype for embeddings
    
    @classmethod
    def initialize(cls, force_cpu: bool = False) -> None:
        """Initialize TPU configuration with proper flags and fallback"""
        if not cls._initialized:
            # Configure JAX for TPU efficiency
            jax.config.update("jax_enable_x64", False)
            jax.config.update("jax_default_matmul_precision", "bfloat16")
            
            try:
                if not force_cpu:
                    jax.config.update("jax_platforms", "tpu")
                    jax.config.update("jax_xla_backend", "tpu")
                    # Test TPU availability
                    devices = jax.devices()
                    if any('TPU' in device.device_kind for device in devices):
                        cls._platform = "tpu"
                    else:
                        raise RuntimeError("No TPU devices found")
                else:
                    cls._platform = "cpu"
            except Exception as e:
                warnings.warn(f"TPU initialization failed: {str(e)}. Falling back to available platform.")
                jax.config.update("jax_platforms", "")
                devices = jax.devices()
                cls._platform = devices[0].platform.lower()
                
            cls._initialized = True
    
    @classmethod
    def get_platform(cls) -> str:
        """Get the current execution platform"""
        if not cls._initialized:
            cls.initialize()
        return cls._platform
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get information about current device configuration"""
        if not cls._initialized:
            cls.initialize()
            
        device = jax.devices()[0]
        return {
            "device_type": device.device_kind,
            "platform": device.platform,
            "device_count": len(jax.devices()),
            "memory_per_device": "8GB" if "TPU" in device.device_kind else "Unknown",
            "supports_bfloat16": True,
            "supports_tf32": "GPU" in device.device_kind,
            "supports_mixed_precision": True,
            "compute_dtype": str(cls._compute_dtype),
            "embedding_dtype": str(cls._embedding_dtype)
        }
    
    @classmethod
    def get_compute_dtype(cls) -> Any:
        """Get the default compute dtype (bfloat16 for TPU)"""
        if not cls._initialized:
            cls.initialize()
        return cls._compute_dtype
    
    @classmethod 
    def get_embedding_dtype(cls) -> Any:
        """Get the dtype for embedding layers (always int32)"""
        return cls._embedding_dtype

    @classmethod
    def convert_dtype(cls, x: jnp.ndarray, target_dtype: Any) -> jnp.ndarray:
        """Convert array to target dtype safely"""
        if x.dtype != target_dtype:
            return x.astype(target_dtype)
        return x
    
    @staticmethod
    def get_optimal_batch_size(model_dim: int, seq_len: int, dtype_bytes: int = 2) -> int:
        """Calculate optimal batch size for available hardware"""
        if TPUConfig._platform == "tpu":
            device_mem = 8 * (1024 ** 3)  # 8GB for TPU
        else:
            device_mem = 4 * (1024 ** 3)  # 4GB for CPU/GPU
            
        overhead = 1.2  # 20% overhead for activations
        return int(device_mem / (model_dim * seq_len * dtype_bytes * overhead))
    
    @staticmethod
    def get_optimal_config(hidden_size: int, seq_len: int, vocab_size: int,
                          num_layers: Optional[int] = None) -> Dict[str, Any]:
        """Get optimal configuration for model parameters"""
        if not TPUConfig._initialized:
            TPUConfig.initialize()
            
        batch_size = TPUConfig.get_optimal_batch_size(hidden_size, seq_len)
        
        config = {
            "hidden_size": hidden_size,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "compute_dtype": TPUConfig._compute_dtype,
            "embedding_dtype": TPUConfig._embedding_dtype,
            "hardware": {
                "platform": TPUConfig.get_platform(),
                "cores_per_replica": len(jax.devices()),
                "replicas": 1
            },
            "training": {
                "optimizer": {
                    "learning_rate": 1e-4,
                    "warmup_steps": 1000,
                    "grad_clip_norm": 1.0
                },
                "precision": {
                    "policy": "mixed_bfloat16",
                    "compute_type": "bfloat16",
                    "embed_type": "int32",
                    "output_type": "float32"
                }
            }
        }
        
        if num_layers is not None:
            config["num_layers"] = num_layers
            
        return config
    
    @staticmethod
    def print_configuration() -> None:
        """Print current hardware configuration"""
        if not TPUConfig._initialized:
            TPUConfig.initialize()
            
        print("\nHardware Configuration:")
        print("-" * 50)
        print(f"Platform: {TPUConfig.get_platform()}")
        print(f"Available devices: {len(jax.devices())}")
        print(f"Device type: {jax.devices()[0].device_kind}")
        print(f"Using bfloat16: {jax.config.jax_enable_x64 == False}")
        print(f"Default matmul precision: {jax.config.jax_default_matmul_precision}")
        print("-" * 50)

if __name__ == "__main__":
    # Initialize hardware configuration
    TPUConfig.initialize()
    
    # Print current configuration
    TPUConfig.print_configuration()
    
    # Get optimal configuration example
    config = TPUConfig.get_optimal_config(
        hidden_size=768,
        seq_len=512,
        vocab_size=32000,
        num_layers=12
    )
    
    print("\nOptimal configuration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)