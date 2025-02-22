"""A100-optimized configurations for Vishwamai"""
from dataclasses import dataclass
from typing import Optional

from ..model import ModelArgs, UnifiedConfig

@dataclass
class A100TrainingConfig:
    """Configuration optimized for A100 GPU training"""
    batch_size: int = 256
    gradient_accumulation_steps: int = 4
    sequence_length: int = 2048
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_steps: int = 50000
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    fp8_training: bool = True
    gradient_checkpointing: bool = True
    flash_attention: bool = True

VISHWAMAI_A100 = ModelArgs(
    hidden_size=4096,
    intermediate_size=16384,
    num_attention_heads=32,
    num_hidden_layers=32,
    max_position_embeddings=8192,
    dtype="bf16",  # Use BF16 for A100
    use_flash_attention=True,
    use_kernel_optimizations=True,
    unified=UnifiedConfig(
        transformer=dict(
            fused_qkv=True,
            fused_mlp=True,
            use_memory_efficient_attention=True
        ),
        parallel=dict(
            tensor_parallel_size=1,  # Will be adjusted based on available GPUs
            sequence_parallel=True
        )
    )
)