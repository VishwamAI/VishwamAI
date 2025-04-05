"""Budget-optimized model configuration for TPU v3-8."""

from dataclasses import dataclass
from typing import Dict, Any

class BudgetModelConfig:
    """Credit-efficient configuration for budget model."""
    
    def __init__(self):
        # Model configuration optimized for efficiency
        self.model_config = {
            "vocab_size": 32000,
            "hidden_size": 512,
            "intermediate_size": 2048,
            "num_attention_heads": 8,
            "num_hidden_layers": 8,
            "max_position_embeddings": 512,
            "attention_dropout": 0.1,
            "hidden_dropout": 0.1,
            "use_flash_attention": True,
            "use_fp8": True,
            "block_size": 128
        }

        # Training configuration for maximum efficiency
        self.training_config = {
            "batch_size": 16,        # Reduced from 32 
            "gradient_accumulation_steps": 8,  # Adjusted for effective batch size <= 256
            "learning_rate": 5e-4,
            "warmup_steps": 500,
            "max_steps": 20000,
            "save_steps": 2000,
            "eval_steps": 500
        }

        # Memory optimizations 
        self.memory_config = {
            "use_gradient_checkpointing": True,
            "use_fp8_kv_cache": True,
            "use_flash_attention": True
        }

    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return (self.training_config["batch_size"] * 
                self.training_config["gradient_accumulation_steps"])