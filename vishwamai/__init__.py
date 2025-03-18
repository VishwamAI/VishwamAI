"""VishwamAI: A TPU-optimized transformer model with multimodal capabilities."""

from .model import VishwamAI
from .tokenizer import VishwamAITokenizer
from .training import TPUTrainingConfig, VishwamAITrainer
from .distill import DistillationTrainer, LinearPathDistillation,IntermediateLayerDistillation, ProgressiveLayerDropout, create_layer_mapping, compute_attention_distillation_loss
from .flash_attention import flash_attention, FlashAttentionLayer
from .layers.layers import (
    TPUGEMMLinear,
    TPULayerNorm,
    TPUMultiHeadAttention,
    TPUMoELayer,
    create_layer_factory
)
from .pipeline import VishwamAIPipeline, load_multimodal_pipeline
from .device_mesh import TPUMeshContext
from .profiler import TPUProfiler
from .logger import DuckDBLogger
from .thoughts.tot import TreeOfThoughts, ThoughtNode, evaluate_tot_solution
from .thoughts.cot import ChainOfThoughtPrompting
from .multimodal import (
    MultimodalEncoder,
    MultimodalProcessor,
    MultimodalConfig,
    ImageCaptioningPipeline,
    VisualQuestionAnswering, 
    AudioCaptioningPipeline,
    MultimodalChatPipeline,
    pipeline
)

__version__ = "0.1.0"

def pipeline(
    task: str,
    model: str = None,
    **kwargs
) -> VishwamAIPipeline:
    """Create a pipeline for the specified task.
    
    Args:
        task: Task to create pipeline for. One of:
            - "text-generation"
            - "chat"
            - "image-captioning"
            - "visual-qa"
            - "audio-captioning"
            - "multimodal-chat"
        model: Model name or path
        **kwargs: Additional arguments passed to pipeline
        
    Returns:
        Appropriate pipeline for the task
    """
    if task in ["image-captioning", "visual-qa", "audio-captioning", "multimodal-chat"]:
        return load_multimodal_pipeline(model, task=task, **kwargs)
    else:
        # Standard text pipeline
        return VishwamAIPipeline(model, **kwargs)

__all__ = [
    # Model and Core Components
    "VishwamAI",
    "VishwamAITokenizer",
    "TPUTrainingConfig",
    "VishwamAITrainer",
    
    # Distillation Features
    "DistillationTrainer",
    "LinearPathDistillation",
    "IntermediateLayerDistillation",
    "ProgressiveLayerDropout",
    "create_layer_mapping",
    "compute_attention_distillation_loss",
    
    # Layer Components
    "TPUGEMMLinear",
    "TPULayerNorm", 
    "TPUMultiHeadAttention",
    "TPUMoELayer",
    "create_layer_factory",
    "flash_attention",
    "FlashAttentionLayer",
    
    # Pipeline and Infrastructure
    "VishwamAIPipeline",
    "TPUMeshContext",
    "TPUProfiler",
    "DuckDBLogger",
    
    # Advanced Features
    "TreeOfThoughts",
    "ThoughtNode",
    "evaluate_tot_solution",
    "ChainOfThoughtPrompting",
    
    # Multimodal components
    'MultimodalEncoder',
    'MultimodalProcessor',
    'MultimodalConfig',
    'ImageCaptioningPipeline',
    'VisualQuestionAnswering', 
    'AudioCaptioningPipeline',
    'MultimodalChatPipeline',
    'pipeline'
]
