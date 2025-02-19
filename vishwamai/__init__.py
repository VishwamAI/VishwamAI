"""
VishwamAI: Advanced Language Model Framework
===========================================

VishwamAI is a modular implementation of advanced machine learning components,
focusing on transformer-based models with specialized architectures and training methods.

Main Components:
- Models: Transformer architectures and specialized neural networks
- Training: Advanced training routines and initialization methods
- Utils: Configuration, tokenization, and helper functions
- Extensions: Specialized modules for behavior analysis, ethics, and memory
"""

from . import models
from . import training
from . import utils
from . import extensions

# Version information
__version__ = "0.1.0"
__author__ = "VishwamAI Team"

# Expose key functionality at package level
from .models import Transformer, create_model, load_model
from .training import train_model, initialize_model
from .utils import load_config
from .extensions import generate_text

__all__ = [
    # Main packages
    'models',
    'training',
    'utils',
    'extensions',
    
    # Key functionality
    'Transformer',
    'create_model',
    'load_model',
    'train_model',
    'initialize_model',
    'load_config',
    'generate_text',
    
    # Version info
    '__version__',
    '__author__',
]
