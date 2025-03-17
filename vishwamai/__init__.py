"""VishwamAI: TPU-optimized text-to-text generation model."""

from .model import VishwamAI
from .tokenizer import VishwamAITokenizer
from .training import TPUTrainingConfig, VishwamAITrainer
from .distill import DistillationTrainer, LinearPathDistillation,IntermediateLayerDistillation, ProgressiveLayerDropout, create_layer_mapping, compute_attention_distillation_loss
from .flash_attention import flash_attention, FlashAttentionLayer
from .layers import (
    TPUGEMMLinear,
    TPULayerNorm,
    TPUMultiHeadAttention,
    TPUMoELayer,
    create_layer_factory
)
from .pipeline import VishwamAIPipeline
from .device_mesh import TPUMeshContext
from .profiler import TPUProfiler
from .logger import DuckDBLogger
from .tot import TreeOfThoughts, ThoughtNode, evaluate_tot_solution
from .cot import ChainOfThoughtPrompting

__version__ = "0.1.0"
__all__ = [
    # Core Model
    "VishwamAI",
    "VishwamAITokenizer",
    
    # Training
    "TPUTrainingConfig",
    "VishwamAITrainer",
    "DistillationTrainer",
    "LinearPathDistillation",
    "IntermediateLayerDistillation",
    "ProgressiveLayerDropout",
    "create_layer_mapping",
    "compute_attention_distillation_loss",
    
    # Attention and Layers
    "flash_attention",
    "FlashAttentionLayer",
    "TPUGEMMLinear",
    "TPULayerNorm",
    "TPUMultiHeadAttention",
    "TPUMoELayer",
    "create_layer_factory",
    
    # Pipeline and Infrastructure
    "VishwamAIPipeline",
    "TPUMeshContext",
    "TPUProfiler",
    "DuckDBLogger",
    
    # Advanced Features
    "TreeOfThoughts",
    "ThoughtNode",
    "evaluate_tot_solution",
    "ChainOfThoughtPrompting"
]
