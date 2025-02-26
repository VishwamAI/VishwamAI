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
from .training import train, create_train_state

# Import error correction components
from .error_correction import (
    ErrorCorrectionState,
    ErrorCorrectionOutput,
    ErrorMetrics,
    ErrorCorrectionModule,
    MixtureDensityNetwork,
    ErrorCorrectionTrainer,
    compute_error_metrics,
    create_error_correction_state,
    create_error_corrected_train_step,
    create_error_corrected_eval_step
)

# Import other components
from .tot import TreeOfThoughts

__all__ = [
    "VishwamAIModel",
    "ModelConfig",
    "VishwamAITokenizer",
    "train",
    "create_train_state",
    "ErrorCorrectionState",
    "ErrorCorrectionOutput",
    "ErrorMetrics",
    "ErrorCorrectionModule",
    "MixtureDensityNetwork",
    "ErrorCorrectionTrainer",
    "compute_error_metrics",
    "create_error_correction_state",
    "create_error_corrected_train_step",
    "create_error_corrected_eval_step",
    "TreeOfThoughts",
]
