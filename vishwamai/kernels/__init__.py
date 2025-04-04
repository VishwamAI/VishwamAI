"""VishwamAI TPU-optimized kernel implementations."""

from vishwamai.kernels.core import (
    Kernel,  # Import the aliased AbstractKernel
    KernelConfig,
    HardwareType,
    KernelType,
    KernelManager,
    get_kernel_manager,
    MemoryLayout,
    MemoryBlock,
    MemoryManager,
    TuningConfig,
    TuningResult,
    KernelTuner,
    AutotuneManager
)

from vishwamai.kernels.tpu.optimizers import (
    TPUOptimizer,
    TPUAdam,
    TPULion,
    TPUAdafactor
)

from vishwamai.kernels.tpu.layer_optimizers import (
    LayerOptConfig,
    AdaptiveLayerOptimizer,
    TPULayerNorm,
    FFNOptimizer,
    AdaptiveMoEOptimizer
)

from vishwamai.kernels.tpu.kernels import (
    TPULayerNormKernel,
    act_quant,
    optimize_kernel_layout,
    block_tpu_matmul,
    fp8_gemm_optimized
)

from vishwamai.kernels.tpu.kernel_fusion import (
    FusionConfig,
    FusionPattern
)

from vishwamai.kernels.tpu.flash_attention import TPUFlashAttention
from vishwamai.kernels.tpu.distillation_kernels import (
    DistillationKernelConfig,
    DistillationOutput,
    DistillationKernelManager
)

from vishwamai.kernels.optimizers.quantized_adam import QuantizedState
from vishwamai.kernels.optimizers.quantized_lion import TPUQuantizedLion

# Optional CUDA kernels if available
try:
    from vishwamai.kernels.cuda.flashmla_cuda import FlashMLACUDA
    from vishwamai.kernels.cuda.flash_kv import FlashKVCache
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

__all__ = [
    # Core Classes
    'Kernel',
    'KernelConfig',
    'HardwareType',
    'KernelType',
    'KernelManager',
    'get_kernel_manager',
    
    # TPU Optimizers
    'TPUOptimizer',
    'TPUAdam', 
    'TPULion',
    'TPUAdafactor',
    'TPUQuantizedLion',
    
    # Layer Optimizations
    'LayerOptConfig',
    'AdaptiveLayerOptimizer',
    'TPULayerNorm',
    'FFNOptimizer',
    'AdaptiveMoEOptimizer',
    
    # Core Kernels
    'TPULayerNormKernel',
    'act_quant',
    'optimize_kernel_layout',
    'block_tpu_matmul',
    'fp8_gemm_optimized',
    
    # Kernel Fusion
    'FusionConfig',
    'FusionPattern',
    
    # Attention & Distillation
    'TPUFlashAttention',
    'DistillationKernelConfig',
    'DistillationOutput',
    'DistillationKernelManager',
    
    # Memory Management
    'MemoryLayout',
    'MemoryBlock',
    'MemoryManager',
    
    # Tuning & Optimization
    'TuningConfig',
    'TuningResult',
    'KernelTuner',
    'AutotuneManager',
    
    # Quantization
    'QuantizedState',
    
    # Optional CUDA kernels
    *(
        ['FlashMLACUDA', 'FlashKVCache']
        if CUDA_AVAILABLE else []
    )
]