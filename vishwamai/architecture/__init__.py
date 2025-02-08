"""
VishwamAI Architecture Module
============================

This module implements the core neural network architecture components for VishwamAI.
"""

from .transformer import VishwamaiModel, TransformerBlock
from .attention import MultiHeadAttention, SelfAttention, CrossAttention
from .mlp import MLP, FFN, GatedMLP
from .init import init_model, load_checkpoint
from .config import VishwamaiConfig, TrainingConfig, GenerationConfig

__all__ = [
    # Main model
    'VishwamaiModel',
    'TransformerBlock',
    
    # Attention mechanisms
    'MultiHeadAttention',
    'SelfAttention',
    'CrossAttention',
    
    # MLP variants
    'MLP',
    'FFN',
    'GatedMLP',
    
    # Model initialization
    'init_model',
    'load_checkpoint',
    
    # Configurations
    'VishwamaiConfig',
    'TrainingConfig',
    'GenerationConfig'
]

# Version info
__version__ = '0.1.0'
