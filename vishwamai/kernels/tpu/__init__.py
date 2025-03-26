"""TPU-optimized kernels for VishwamAI."""

from .tpu_custom_call import (
    tpu_custom_call,
    compile_tpu_kernel,
    optimize_tpu_layout,
    pad_to_tpu_multiple,
    get_optimal_tpu_layout
)

from .gemm import TPUGEMMKernel
from .attention import TPUAttentionKernel
from .layer_norm import TPULayerNormKernel
from .flash_attention import TPUFlashAttention

__all__ = [
    # TPU custom call functions
    "tpu_custom_call",
    "compile_tpu_kernel",
    "optimize_tpu_layout",
    "pad_to_tpu_multiple",
    "get_optimal_tpu_layout",
    
    # TPU kernels
    "TPUGEMMKernel",
    "TPUAttentionKernel",
    "TPULayerNormKernel",
    "TPUFlashAttention",
]