"""
VishwamAI: Advanced Multimodal AI Framework

A comprehensive framework for building efficient multimodal AI systems
with curriculum learning support, featuring Gemma-inspired architectures
and state-of-the-art attention mechanisms.
"""

from .model import VishwamAIModel, ModelConfig, create_integrated_model
from .training import TrainingConfig, CurriculumTrainer
from .kernels import TPUKernels, GPUKernels
from .pipeline import pipeline, TextGenerator, MultimodalGenerator
from .multimodal import MultimodalProcessor, VisionEncoder, AudioEncoder
from .attention import FlashAttention, OptimizedAttention
from .utils import (
    create_optimizer, setup_mixed_precision, get_hardware_info, 
    print_model_info, estimate_memory_usage
)

# New advanced multimodal components inspired by Gemma
from .advanced_multimodal import (
    GemmaInspiredMultimodalTransformer,
    MultimodalConfig,
    SigLIPVisionEncoder,
    AttentionPoolingHead,
    AdaptiveModalityFusion,
    GroupedQueryAttention,
    GEMMA_4B_MULTIMODAL_CONFIG,
    GEMMA_12B_MULTIMODAL_CONFIG,
    create_multimodal_model
)

# Advanced attention mechanisms
from .gemma_attention import (
    FlashAttention2,
    CrossModalAttention,
    MultiScaleAttention,
    SparseAttention,
    AdvancedRoPE,
    QueryKeyNormalization,
    AttentionType,
    QueryPreAttentionNorm,
    create_attention_layer,
    GEMMA_ATTENTION_CONFIG
)

# Advanced training pipeline
from .multimodal_training import (
    MultimodalTrainer,
    TrainingConfig as AdvancedTrainingConfig,
    CurriculumStage,
    AdaptiveLearningRateSchedule,
    MultimodalLoss,
    main_training_pipeline,
    SMALL_MODEL_TRAINING_CONFIG,
    MEDIUM_MODEL_TRAINING_CONFIG,
    LARGE_MODEL_TRAINING_CONFIG
)

__version__ = "0.2.0"
__author__ = "VishwamAI Team"

__all__ = [
    # Core model components
    "VishwamAIModel",
    "ModelConfig", 
    "create_integrated_model",
    "TrainingConfig",
    "CurriculumTrainer",
    
    # Hardware optimization
    "TPUKernels",
    "GPUKernels",
    
    # Generation pipelines
    "pipeline",
    "TextGenerator",
    "MultimodalGenerator",
    
    # Multimodal components
    "MultimodalProcessor",
    "VisionEncoder",
    "AudioEncoder",
    
    # Attention mechanisms
    "FlashAttention",
    "OptimizedAttention",
    
    # Utility functions
    "create_optimizer",
    "setup_mixed_precision",
    "get_hardware_info",
    "print_model_info",
    "estimate_memory_usage",
    
    # Advanced Gemma-inspired components
    "GemmaInspiredMultimodalTransformer",
    "MultimodalConfig",
    "SigLIPVisionEncoder",
    "AttentionPoolingHead",
    "AdaptiveModalityFusion",
    "GEMMA_4B_MULTIMODAL_CONFIG",
    "GEMMA_12B_MULTIMODAL_CONFIG",
    "create_multimodal_model",
    
    # Advanced attention mechanisms
    "FlashAttention2",
    "GroupedQueryAttention", 
    "CrossModalAttention",
    "MultiScaleAttention",
    "SparseAttention",
    "AdvancedRoPE",
    "QueryKeyNormalization",
    "AttentionType",
    "QueryPreAttentionNorm",
    "create_attention_layer",
    "GEMMA_ATTENTION_CONFIG",
    
    # Advanced training pipeline
    "MultimodalTrainer",
    "AdvancedTrainingConfig",
    "CurriculumStage", 
    "AdaptiveLearningRateSchedule",
    "MultimodalLoss",
    "main_training_pipeline",
    "SMALL_MODEL_TRAINING_CONFIG",
    "MEDIUM_MODEL_TRAINING_CONFIG",
    "LARGE_MODEL_TRAINING_CONFIG",
]
