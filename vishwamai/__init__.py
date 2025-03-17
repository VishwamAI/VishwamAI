"""VishwamAI: TPU-optimized text-to-text generation model."""

from .model import VishwamAI
from .tokenizer import VishwamAITokenizer
from .training import TPUTrainingConfig, VishwamAITrainer
from .distill import DistillationTrainer, compute_distillation_loss
from .flash_attention import flash_attention, FlashAttentionLayer
from .layers import (
    TPUGEMMLinear,
    TPULayerNorm,
    TPUMultiHeadAttention,
    TPUMoELayer,
    create_layer_factory
)
from .pipeline import VishwamAIPipeline
from .device_mesh import configure_device_mesh, DeviceMesh
from .profiler import TPUProfiler, profile_memory_usage
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
    "compute_distillation_loss",
    
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
    "configure_device_mesh",
    "DeviceMesh",
    "TPUProfiler",
    "profile_memory_usage",
    "DuckDBLogger",
    
    # Advanced Features
    "TreeOfThoughts",
    "ThoughtNode",
    "evaluate_tot_solution",
    "ChainOfThoughtPrompting"
]
