"""
VishwamAI Transformer Implementation
"""

# Core Model Components
from .transformer import (
    create_vishwamai_transformer,
    create_train_state,
    TransformerModel,
    EnhancedTransformerModel,
    FlashAttention,
    RMSNorm,
    TPURotaryEmbedding
)

from .cot import ChainOfThoughtPrompting
from .tot import TreeOfThoughts, ThoughtNode, evaluate_tot_solution
from .distill import compute_distillation_loss, create_student_model, initialize_from_teacher

# Optimized TPU Kernel Operations
from .kernel import (
    fp8_cast_transpose,
    fp8_gemm_optimized,
    block_tpu_matmul,
    act_quant,
    multi_head_attention_kernel,
    flash_attention,
    rope_embedding,
    apply_rotary_pos_emb,
    weight_dequant,
    batch_norm_tpu,
    fp8_gemm
)

# Advanced Layers & Architectures
from .layers import MLABlock, MoELayer
from .pipeline import VishwamAIPipeline
from .training import VishwamAITrainer, create_trainer

# Logging & Monitoring
from .logger import DuckDBLogger

__version__ = "0.1.0"

DEFAULT_CONFIG = {
    'model_config': {
        'vocab_size': 32000,
        'num_layers': 12,
        'num_heads': 12,
        'head_dim': 64,  # Fixed head dimension
        'hidden_dim': 768,
        'mlp_dim': 3072,
        'max_seq_len': 2048,
        'dropout_rate': 0.1,
        'attention_dropout_rate': 0.1,
        'use_enhanced': True,
        'use_rotary': True,
        'use_flash_attn': True,
        'use_rms_norm': True,
        'dtype': 'bfloat16',
        'param_dtype': 'bfloat16',
        'compute_dtype': 'float32'
    },
    'training': {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'warmup_steps': 2000,
        'decay_steps': 50000,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.95,
        'epsilon': 1e-8,
        'gradient_checkpointing': True,
        'gradient_accumulation_steps': 4,
        'mixed_precision': True,
        'tpu_iterations_per_loop': 100
    }
}

__all__ = [
    # Core Models
    'create_vishwamai_transformer',
    'create_train_state',
    'TransformerModel',
    'EnhancedTransformerModel',
    'FlashAttention',
    'RMSNorm',
    'TPURotaryEmbedding',
    
    # Reasoning Components
    'ChainOfThoughtPrompting',
    'TreeOfThoughts',
    'ThoughtNode',
    'evaluate_tot_solution',
    
    # Training Components
    'compute_distillation_loss',
    'create_student_model',
    'initialize_from_teacher',
    
    # TPU Optimizations
    'fp8_cast_transpose',
    'fp8_gemm_optimized',
    'block_tpu_matmul',
    'act_quant',
    'multi_head_attention_kernel',
    'flash_attention',
    'rope_embedding',
    'apply_rotary_pos_emb',
    'weight_dequant',
    'batch_norm_tpu',
    'fp8_gemm',
    
    # Advanced Components
    'MLABlock',
    'MoELayer',
    'VishwamAIPipeline',
    'VishwamAITrainer',
    'create_trainer',
    'DuckDBLogger',
    
    # Configuration
    'DEFAULT_CONFIG'
]
