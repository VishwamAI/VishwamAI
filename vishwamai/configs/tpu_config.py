"""
TPU configuration and initialization utilities
"""

import jax
import os
from typing import Dict, Any, Optional

class TPUConfig:
    """TPU configuration management"""
    
    _instance = None
    _initialized = False
    
    @classmethod
    def initialize(cls, force_cpu: bool = False) -> None:
        """Initialize TPU configuration with proper flags"""
        if not cls._initialized:
            # Disable float64 for TPU efficiency
            jax.config.update("jax_enable_x64", False)
            
            # Use bfloat16 for matrix operations
            jax.config.update("jax_default_matmul_precision", "bfloat16")
            
            if not force_cpu:
                # Force TPU platform and backend if available
                jax.config.update("jax_platforms", "tpu")
                jax.config.update("jax_xla_backend", "tpu")
            
            cls._initialized = True
    
    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get information about the current device configuration"""
        device = jax.devices()[0]
        return {
            "device_type": device.device_kind,
            "platform": device.platform,
            "device_count": len(jax.devices()),
            "memory_per_device": "8GB",  # Standard for TPU v3-8
            "supports_bfloat16": True,
            "supports_tf32": False,
            "supports_mixed_precision": True
        }
    
    @staticmethod
    def get_optimal_batch_size(model_dim: int, seq_len: int, dtype_bytes: int = 2) -> int:
        """Calculate optimal batch size for TPU memory
        
        Args:
            model_dim: Model hidden dimension size
            seq_len: Sequence length
            dtype_bytes: Bytes per element (2 for bfloat16/float16, 4 for float32)
        
        Returns:
            Optimal batch size for TPU memory
        """
        device_mem = 8 * (1024 ** 3)  # Assume 8GB per TPU core
        overhead = 1.2  # 20% overhead for activations
        return int(device_mem / (model_dim * seq_len * dtype_bytes * overhead))
    
    @staticmethod
    def get_optimal_config(hidden_size: int, seq_len: int, vocab_size: int,
                          num_layers: Optional[int] = None) -> Dict[str, Any]:
        """Get optimal TPU configuration for model parameters
        
        Args:
            hidden_size: Model hidden dimension
            seq_len: Maximum sequence length
            vocab_size: Vocabulary size
            num_layers: Number of transformer layers (optional)
            
        Returns:
            Dictionary with optimal configuration parameters
        """
        batch_size = TPUConfig.get_optimal_batch_size(hidden_size, seq_len)
        
        config = {
            "hidden_size": hidden_size,
            "seq_len": seq_len,
            "vocab_size": vocab_size,
            "batch_size": batch_size,
            "dtype": "bfloat16",
            "optimizer": {
                "learning_rate": 1e-4,
                "warmup_steps": 1000,
                "grad_clip_norm": 1.0
            },
            "hardware": {
                "cores_per_replica": len(jax.devices()),
                "replicas": 1,
                "precision": "bfloat16"
            }
        }
        
        if num_layers is not None:
            config["num_layers"] = num_layers
            
        return config
    
    @staticmethod
    def print_configuration() -> None:
        """Print current TPU configuration"""
        print("\nTPU Configuration:")
        print("-" * 50)
        print(f"Available devices: {len(jax.devices())}")
        print(f"Device type: {jax.devices()[0].device_kind}")
        print(f"Platform: {jax.devices()[0].platform}")
        print(f"Using bfloat16: {jax.config.jax_enable_x64 == False}")
        print(f"Default matmul precision: {jax.config.jax_default_matmul_precision}")
        print("-" * 50)

if __name__ == "__main__":
    # Initialize TPU configuration
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