"""VishwamAI: TPU-optimized language model with advanced reasoning capabilities."""

from typing import Dict, Any, Optional, Tuple
from .model import VishwamAI, VishwamAIConfig
from .transformer import TPUTrainingState, EnhancedTransformerModel
from .device_mesh import TPUMeshContext
from .pipeline import TPUDataPipeline, DistillationDataPipeline, VishwamAIPipeline
from .distill import (
    compute_distillation_loss,
    create_student_model,
    initialize_from_teacher,
    DistillationTrainer,
    DistillationConfig
)
from .thoughts.tot import TreeOfThoughts, ThoughtNode
from .thoughts.cot import ChainOfThoughtPrompting
from .logger import DuckDBLogger
from .profiler import TPUProfiler

__version__ = "0.1.0"

def pipeline(
    task: str,
    model: str = None,
    **kwargs
) -> 'VishwamAIPipeline':
    """Create a pipeline for the specified task."""
    return VishwamAIPipeline(model, task=task, **kwargs)

def init_thinking_components(
    model: VishwamAI,
    params: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None,
    thinking_config: Optional[Dict[str, Any]] = None
) -> Tuple[TreeOfThoughts, ChainOfThoughtPrompting]:
    """Initialize thinking components (Tree of Thoughts and Chain of Thought).
    
    Args:
        model: VishwamAI model instance
        params: Optional model parameters
        tokenizer: Optional tokenizer
        thinking_config: Optional thinking configuration
        
    Returns:
        Tuple of (TreeOfThoughts, ChainOfThoughtPrompting) instances
    """
    config = thinking_config or {
        "max_branches": 3,
        "max_depth": 3,
        "beam_width": 5,
        "temperature": 0.7,
        "max_length": 512,
        "num_samples": 3
    }
    
    tot = TreeOfThoughts(
        model=model,
        params=params,
        tokenizer=tokenizer,
        max_branches=config.get("max_branches", 3),
        max_depth=config.get("max_depth", 3),
        beam_width=config.get("beam_width", 5),
        temperature=config.get("temperature", 0.7)
    )
    
    cot = ChainOfThoughtPrompting(
        model=model,
        params=params,
        tokenizer=tokenizer,
        temperature=config.get("temperature", 0.7),
        max_length=config.get("max_length", 512)
    )
    
    return tot, cot

__all__ = [
    'VishwamAI',
    'VishwamAIConfig',
    'TPUTrainingState',
    'EnhancedTransformerModel',
    'TPUMeshContext',
    'TPUDataPipeline',
    'DistillationDataPipeline',
    'DistillationTrainer',
    'create_student_model',
    'initialize_from_teacher',
    'compute_distillation_loss',
    'DistillationConfig',
    'TreeOfThoughts',
    'ThoughtNode',
    'ChainOfThoughtPrompting',
    'DuckDBLogger',
    'TPUProfiler'
]
