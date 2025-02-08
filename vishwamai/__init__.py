"""
VishwamAI: Advanced Language Model with Conceptual Understanding
=============================================================

This package implements an advanced language model with:
- Conceptual understanding and reasoning capabilities
- Memory-efficient implementation
- Advanced attention mechanisms
- Transformer-style architecture with residual connections
"""

from .architecture import (
    VishwamaiModel,
    VishwamaiConfig,
    TrainingConfig,
    GenerationConfig,
    init_model,
    MultiHeadAttention
)

from .toknizer import (
    ConceptualTokenizer,
    ConceptualTokenizerConfig
)

from .generate import (
    generate,
    sample_top_p,
    sample_temperature
)

from .kernel import (
    act_quant,
    weight_dequant,
    fp8_gemm
)

from .convert import (
    convert_to_fp8,
    get_tensor_format
)

__version__ = "0.1.0"
__author__ = "VishwamAI Team"

__all__ = [
    # Main model and config
    "VishwamaiModel",
    "VishwamaiConfig",
    "TrainingConfig",
    "GenerationConfig",
    "init_model",
    "MultiHeadAttention",
    
    # Tokenizer
    "ConceptualTokenizer",
    "ConceptualTokenizerConfig",
    
    # Generation utilities
    "generate",
    "sample_top_p",
    "sample_temperature",
    
    # Kernel operations
    "act_quant",
    "weight_dequant",
    "fp8_gemm",
    
    # Conversion utilities
    "convert_to_fp8",
    "get_tensor_format"
]
