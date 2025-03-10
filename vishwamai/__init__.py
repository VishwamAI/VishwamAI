"""
VishwamAI Transformer Implementation
"""

# Core Model Components
from .cot import ChainOfThoughtPrompting
from .tot import TreeOfThoughts, ThoughtNode, evaluate_tot_solution
from .distill import compute_distillation_loss, create_student_model, initialize_from_teacher
from .transformer import create_vishwamai_transformer, create_train_state

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

__all__ = [
    # Core Components
    'ChainOfThoughtPrompting',
    'TreeOfThoughts',
    'ThoughtNode',
    'evaluate_tot_solution',
    'compute_distillation_loss',
    'create_student_model',
    'initialize_from_teacher',
    'create_vishwamai_transformer',
    'create_train_state',

    # TPU Optimized Kernels
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

    # Advanced Layers
    'MLABlock',
    'MoELayer',

    # Pipeline & Training
    'VishwamAIPipeline',
    'VishwamAITrainer',
    'create_trainer',

    # Logging
    'DuckDBLogger'
]
