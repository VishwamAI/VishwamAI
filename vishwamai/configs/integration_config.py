# VishwamAI/configs/integration_config.py
"""
Configuration for integrating external optimization tools into VishwamAI.
Includes settings for DeepGEMM, 3FS, DeepEP, DualPipe, FlashMLA, and EPLB.
"""

class IntegrationConfig:
    # General settings
    ENABLE_INTEGRATIONS = True  # Master toggle for all integrations
    TPU_MODE = True  # Set to True for TPU compatibility, False for GPU

    # DeepGEMM (Optimized GEMM for matrix operations)
    DEEPGEMM_ENABLED = False  # Disabled by default for TPUs (GPU-specific)
    DEEPGEMM_PRECISION = "fp16"  # Options: "fp16", "fp32" (if enabled on GPU)
    DEEPGEMM_ALTERNATIVE = "tpu_gemm"  # TPU-specific GEMM replacement

    # 3FS (FFT-based operations)
    THREEFS_ENABLED = True
    THREEFS_FFT_SIZE = 1024  # FFT size for fast convolution operations
    THREEFS_PRECISION = "bf16"  # Bfloat16 for TPU efficiency

    # DeepEP (Expert Parallelism)
    DEEPEP_ENABLED = True
    DEEPEP_NUM_EXPERTS = 8  # Number of experts for parallelism
    DEEPEP_TOP_K = 2  # Top-k gating for expert selection

    # DualPipe (Dual Pipeline Processing)
    DUALPIPE_ENABLED = True
    DUALPIPE_STAGES = 2  # Number of pipeline stages
    DUALPIPE_BUFFER_SIZE = 128  # Buffer size for pipeline efficiency

    # FlashMLA (Matrix-Vector Kernels)
    FLASHMLA_ENABLED = True
    FLASHMLA_BLOCK_SIZE = 256  # Block size for matrix-vector ops
    FLASHMLA_PRECISION = "fp8"  # FP8 for ultra-fast TPU computation

    # EPLB (Expert Parallelism Load Balancing)
    EPLB_ENABLED = True
    EPLB_BALANCE_THRESHOLD = 0.1  # Load imbalance threshold (0 to 1)
    EPLB_UPDATE_FREQ = 100  # Steps between load rebalancing

    @staticmethod
    def validate_config():
        """Validate integration settings for compatibility."""
        if IntegrationConfig.TPU_MODE and IntegrationConfig.DEEPGEMM_ENABLED:
            print("Warning: DeepGEMM is GPU-specific and disabled in TPU mode.")
            IntegrationConfig.DEEPGEMM_ENABLED = False

if __name__ == "__main__":
    # Test the configuration
    IntegrationConfig.validate_config()
    print("Integration Config Loaded:", vars(IntegrationConfig))