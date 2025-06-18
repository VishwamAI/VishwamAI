"""
VishwamAI: Efficient multimodal AI framework with curriculum learning support.

This package provides a comprehensive framework for building and training
multimodal AI models optimized for resource-constrained environments.
"""

from .model import VishwamAIModel, ModelConfig, create_integrated_model
from .training import TrainingConfig, CurriculumTrainer
from .kernels import TPUKernels, GPUKernels
from .pipeline import pipeline
from .multimodal import MultimodalProcessor, VisionEncoder, AudioEncoder
from .attention import FlashAttention, OptimizedAttention
from .utils import create_optimizer, setup_mixed_precision

__version__ = "0.1.0"
__author__ = "VishwamAI Team"

__all__ = [
    "VishwamAIModel",
    "ModelConfig", 
    "create_integrated_model",
    "TrainingConfig",
    "CurriculumTrainer",
    "TPUKernels",
    "GPUKernels",
    "pipeline",
    "MultimodalProcessor",
    "VisionEncoder",
    "AudioEncoder",
    "FlashAttention",
    "OptimizedAttention",
    "create_optimizer",
    "setup_mixed_precision",
]
