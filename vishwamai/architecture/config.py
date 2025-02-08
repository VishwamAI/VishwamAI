"""
Model Configuration
=================

This module defines the configuration classes for VishwamAI model architecture.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class VishwamaiConfig:
    """Configuration class for VishwamAI model architecture."""
    
    # Model architecture - Optimized for 4GB VRAM
    vocab_size: int = 32000
    max_seq_length: int = 2048  # Reduced from 8192
    dim: int = 2048  # Reduced from 4096
    depth: int = 24  # Reduced from 32
    num_heads: int = 16  # Reduced from 32
    mlp_ratio: float = 2.67  # Reduced from 4.0
    dropout: float = 0.1
    
    # Optional parameters
    num_key_value_heads: Optional[int] = None  # For grouped-query attention
    intermediate_size: Optional[int] = None  # For custom MLP sizes
    
    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    mask_token_id: int = 3
    
    # Tokenizer settings
    vocab_file: Optional[str] = None
    merges_file: Optional[str] = None
    
    # Training settings
    weight_decay: float = 0.01
    learning_rate: float = 1e-4
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_heads
        if self.intermediate_size is None:
            self.intermediate_size = int(self.dim * self.mlp_ratio)
            
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    batch_size: int = 4  # Reduced from 32 for memory efficiency
    gradient_accumulation_steps: int = 4  # Increased for stable training
    max_steps: int = 100000
    save_steps: int = 1000
    eval_steps: int = 1000
    
    # Optimizer settings
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    
    max_length: int = 200
    min_length: int = 0
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
