# VishwamAI/configs/tpu_config.py
"""
TPU optimization parameters for VishwamAI, including precision,
batching, and parallelism settings.
"""

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

    @staticmethod
    def update_batch_size(new_global_batch_size):
        """Update batch size dynamically and recalculate per-core size."""
        TPUConfig.GLOBAL_BATCH_SIZE = new_global_batch_size
        TPUConfig.PER_CORE_BATCH_SIZE = new_global_batch_size // TPUConfig.NUM_TPU_CORES
        print(f"Updated: Global Batch Size = {TPUConfig.GLOBAL_BATCH_SIZE}, "
              f"Per-Core Batch Size = {TPUConfig.PER_CORE_BATCH_SIZE}")

if __name__ == "__main__":
    # Test the configuration
    print("TPU Config:", vars(TPUConfig))
    TPUConfig.update_batch_size(2048)  # Example update