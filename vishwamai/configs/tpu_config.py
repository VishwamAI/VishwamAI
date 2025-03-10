# VishwamAI/configs/tpu_config.py
"""
TPU optimization parameters for VishwamAI, including precision,
batching, and parallelism settings.
"""

import jax
import jax.numpy as jnp
import os

class TPUConfig:
    # TPU hardware settings
    TPU_VERSION = "v2-8"  # Default TPU type (e.g., "v2-8", "v3-32")
    NUM_TPU_CORES = 8  # Number of TPU cores (adjust based on TPU type)

    # Precision settings
    DEFAULT_PRECISION = "bf16"  # Options: "fp8", "bf16", "fp32"
    MIXED_PRECISION = True  # Enable mixed precision training
    LOSS_SCALE = 128.0  # Loss scaling for numerical stability

    # Batching and parallelism
    GLOBAL_BATCH_SIZE = 1024  # Total batch size across all TPU cores
    PER_CORE_BATCH_SIZE = GLOBAL_BATCH_SIZE // NUM_TPU_CORES  # Per-core batch size
    DATA_PARALLELISM = True  # Enable data parallelism across cores
    MODEL_PARALLELISM = False  # Enable model parallelism (if needed)

    # Optimization settings
    GRADIENT_ACCUMULATION_STEPS = 4  # Steps for gradient accumulation
    MAX_GRAD_NORM = 1.0  # Gradient clipping threshold
    OPTIMIZER = "adamw"  # Options: "adam", "adamw", "sgd"

    @classmethod
    def initialize(cls):
        """Initialize TPU configuration and validate settings."""
        # Configure TPU environment
        os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
        jax.config.update("jax_enable_x64", False)
        jax.config.update("jax_default_matmul_precision", "bfloat16")
        jax.config.update("jax_platforms", "tpu")
        jax.config.update("jax_xla_backend", "tpu")

        # Validate TPU availability
        devices = jax.devices()
        if not devices or "TPU" not in devices[0].device_kind:
            raise RuntimeError("No TPU devices found")

        # Update core count based on actual devices
        cls.NUM_TPU_CORES = len(devices)
        cls.PER_CORE_BATCH_SIZE = cls.GLOBAL_BATCH_SIZE // cls.NUM_TPU_CORES

        return cls

    @classmethod
    def update_batch_size(cls, new_global_batch_size):
        """Update batch size dynamically and recalculate per-core size."""
        cls.GLOBAL_BATCH_SIZE = new_global_batch_size
        cls.PER_CORE_BATCH_SIZE = new_global_batch_size // cls.NUM_TPU_CORES
        print(f"Updated: Global Batch Size = {cls.GLOBAL_BATCH_SIZE}, "
              f"Per-Core Batch Size = {cls.PER_CORE_BATCH_SIZE}")

    @classmethod
    def get_device_config(cls) -> dict:
        """Get current TPU device configuration."""
        return {
            "device_type": jax.devices()[0].device_kind,
            "num_devices": cls.NUM_TPU_CORES,
            "precision": cls.DEFAULT_PRECISION,
            "mixed_precision": cls.MIXED_PRECISION,
            "global_batch_size": cls.GLOBAL_BATCH_SIZE,
            "per_core_batch_size": cls.PER_CORE_BATCH_SIZE
        }

if __name__ == "__main__":
    # Test the configuration
    config = TPUConfig.initialize()
    print("TPU Config:", config.get_device_config())
    config.update_batch_size(2048)  # Example update