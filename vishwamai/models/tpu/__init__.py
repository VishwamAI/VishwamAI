"""
TPU-optimized model components initialization using JAX/Haiku/XLA
Provides high-performance implementations of attention mechanisms, 
transformer layers, and specialized optimizations for TPU hardware.
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
    generate_cot
)

from .moe import (
    OptimizedMoE,
    ExpertModule,
    ExpertRouter,
    compute_load_balancing_loss,
    ExpertGating
)

from .tot_model import (
    ToTModelTPU,
    ThoughtNodeTPU,
    generate_tot
)

from .transformer import (
    TransformerComputeLayerTPU,
    TransformerMemoryLayerTPU,
    HybridThoughtAwareAttentionTPU,
    PositionalEncoding,
    TokenEmbedding,
    FeedForward
)

from .core import (
    TPUDeviceManager,
    TPUOptimizer,
    TPUDataParallel,
    TPUProfiler,
    TPUModelUtils,
    apply_rotary_embedding,
    create_causal_mask
)

from .kernel_layers import (
    TPUGEMMLinear,
    TPUGroupedGEMMLinear,
    TPULayerNorm,
    get_optimal_tpu_config,
    benchmark_matmul,
    compute_numerical_error,
    DeepGEMMLinear,
    DeepGEMMLayerNorm,
    DeepGEMMGroupedLinear,
    gelu_kernel
)

__all__ = [
    # Attention mechanisms
    "BaseAttentionTPU",
    "FlashMLAttentionTPU",
    "MultiModalAttentionTPU",
    "TemporalAttentionTPU", 
    "SonnetFlashAttentionTPU",
    
    # Core models and generation
    "CoTModelTPU",
    "generate_cot",
    "ToTModelTPU",
    "ThoughtNodeTPU",
    "generate_tot",
    
    # Mixture of Experts
    "OptimizedMoE",
    "ExpertModule",
    "ExpertRouter",
    "ExpertGating",
    "compute_load_balancing_loss",
    
    # Transformer components
    "TransformerComputeLayerTPU",
    "TransformerMemoryLayerTPU", 
    "HybridThoughtAwareAttentionTPU",
    "PositionalEncoding",
    "TokenEmbedding",
    "FeedForward",
    
    # TPU infrastructure
    "TPUDeviceManager",
    "TPUOptimizer",
    "TPUDataParallel",
    "TPUProfiler",
    "TPUModelUtils",
    
    # Core utilities
    "apply_rotary_embedding",
    "create_causal_mask",
    
    # Optimized kernel layers 
    "TPUGEMMLinear",
    "TPUGroupedGEMMLinear",
    "TPULayerNorm",
    "DeepGEMMLinear",
    "DeepGEMMLayerNorm",
    "DeepGEMMGroupedLinear",
    "gelu_kernel",
    
    # Performance utilities
    "get_optimal_tpu_config",
    "benchmark_matmul",
    "compute_numerical_error"
]

# Version
__version__ = "0.1.0"