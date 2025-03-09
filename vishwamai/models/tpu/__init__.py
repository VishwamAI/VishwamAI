"""
TPU-optimized model components initialization using JAX/Haiku
"""

from .attention import (
    BaseAttentionTPU,
    FlashMLAttentionTPU,
    MultiModalAttentionTPU,
    TemporalAttentionTPU,
    SonnetFlashAttentionTPU
)
from .cot_model import (
    CoTModelTPU,
    train_cot_model,
    extract_answer
)
from .moe import (
    OptimizedMoE,
    ExpertModule,
    ExpertRouter,
    compute_load_balancing_loss
)
from .tot_model import (
    ToTModelTPU,
    ThoughtNodeTPU
)
from .transformer import (
    TransformerComputeLayerTPU,
    TransformerMemoryLayerTPU,
    HybridThoughtAwareAttentionTPU
)
from .core import (
    TPUDeviceManager,
    TPUOptimizer,
    TPUDataParallel,
    TPUProfiler,
    TPUModelUtils
)
from .kernel_layers import (
    TPUGEMMLinear,
    TPUGroupedGEMMLinear,
    TPULayerNorm,
    get_optimal_tpu_config,
    benchmark_matmul,
    compute_numerical_error
)

__all__ = [
    # Attention mechanisms
    "BaseAttentionTPU",
    "FlashMLAttentionTPU",
    "MultiModalAttentionTPU",
    "TemporalAttentionTPU",
    "SonnetFlashAttentionTPU",
    
    # Core models
    "CoTModelTPU",
    "OptimizedMoE",
    "ToTModelTPU",
    "ThoughtNodeTPU",
    
    # Model components
    "TransformerComputeLayerTPU",
    "TransformerMemoryLayerTPU",
    "HybridThoughtAwareAttentionTPU",
    
    # Expert components
    "ExpertModule",
    "ExpertRouter",
    
    # TPU core utilities
    "TPUDeviceManager",
    "TPUOptimizer",
    "TPUDataParallel",
    "TPUProfiler",
    "TPUModelUtils",
    
    # Kernel layers
    "TPUGEMMLinear",
    "TPUGroupedGEMMLinear",
    "TPULayerNorm",
    
    # Utility functions
    "train_cot_model",
    "extract_answer",
    "compute_load_balancing_loss",
    "get_optimal_tpu_config",
    "benchmark_matmul",
    "compute_numerical_error"
]