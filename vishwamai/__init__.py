"""
VishwamAI: TPU-optimized generative model framework
==================================================

This package provides TPU-optimized implementations for transformer models
with support for training, inference, and specialized features like
Tree of Thoughts reasoning.
"""

# Version information
__version__ = '0.1.0'

# Make sure the subpackages are correctly initialized
import os
import sys

# Add the parent directory to sys.path to ensure correct imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import commonly used modules
from . import kernels
from . import layers
from . import thoughts

# Core components
from .model import VishwamAI
from .tokenizer import VishwamAITokenizer
from .training.training import TPUTrainingConfig, VishwamAITrainer
from .transformer import EnhancedTransformerModel, create_vishwamai_transformer

# Specialized layers
from .layers.layers import (
    TPUGEMMLinear,
    TPULayerNorm,
    TPUMultiHeadAttention,
    TPUMoELayer,
    create_layer_factory
)

# Flash attention optimizations
from .flash_attention import FlashAttention

# Knowledge distillation
from .distill import (
    compute_distillation_loss,
    create_student_model,
    initialize_from_teacher,
    DistillationTrainer
)

# Pipeline and infrastructure
from .pipeline import VishwamAIPipeline
from .device_mesh import TPUMeshContext
from .profiler import TPUProfiler
from .logger import DuckDBLogger

# Advanced reasoning capabilities
from .thoughts.tot import TreeOfThoughts, ThoughtNode
from .thoughts.cot import ChainOfThoughtPrompting

def pipeline(
    task: str,
    model: str = None,
    **kwargs
) -> 'VishwamAIPipeline':
    """Create a pipeline for the specified task."""
    return VishwamAIPipeline(model, task=task, **kwargs)

__all__ = [
    # Core components
    "VishwamAI",
    "VishwamAITokenizer",
    "EnhancedTransformerModel",
    "create_vishwamai_transformer",
    
    # Training and optimization
    "TPUTrainingConfig",
    "VishwamAITrainer",
    "TPUMeshContext",
    "TPUProfiler",
    "DuckDBLogger",
    
    # Distillation features
    "compute_distillation_loss",
    "create_student_model",
    "initialize_from_teacher",
    "DistillationTrainer",
    
    # Layer components
    "TPUGEMMLinear",
    "TPULayerNorm", 
    "TPUMultiHeadAttention",
    "TPUMoELayer",
    "create_layer_factory",
    "FlashAttention",
    
    # Pipeline
    "VishwamAIPipeline",
    "pipeline",
    
    # Advanced reasoning
    "TreeOfThoughts",
    "ThoughtNode",
    "ChainOfThoughtPrompting",
]
