"""
VishwamAI initialization module.
"""

import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Version info
__version__ = "0.1.0"

# Import core components
from .model import VishwamAIModel, ModelConfig
from .tokenizer import VishwamAITokenizer

# Import error correction components
from .error_correction import (
    ErrorCorrectionModule,
    ErrorCorrectionTrainer,
    compute_error_metrics
)

# Import distillation components
from .distillation import VishwamaiShaalaTrainer

# Import text generation components
from .generate import generate

__all__ = [
    "VishwamAIModel",
    "ModelConfig",
    "VishwamAITokenizer",
    "ErrorCorrectionModule",
    "ErrorCorrectionTrainer",
    "compute_error_metrics",
    "VishwamaiShaalaTrainer",
    "generate"
]
