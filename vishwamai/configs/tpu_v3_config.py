"""TPU v3-8 optimized training configuration."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TPUV3Config:
    """Credit-optimized configuration for TPU v3-8."""
    
    # Model configuration optimized for 8GB HBM per core
    model_config = {
        "vocab_size": 32000,
        "hidden_size": 768,  # Multiple of 128 for TPU efficiency
        "intermediate_size": 3072,  # 4x hidden_size
        "num_attention_heads": 12,  # Multiple of 4 for TPU efficiency
        "num_hidden_layers": 12,  # Reduced for initial training
        "max_position_embeddings": 1024,  # Start smaller, increase gradually
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1
    }
    
    # Training configuration for credit optimization
    training_config = {
        "batch_size": 32,  # Per TPU core
        "grad_accum_steps": 4,  # Effective batch size = 1024
        "gradient_accumulation_steps": 4,  # Added for compatibility
        "learning_rate": 1e-4,
        "warmup_steps": 1000,
        "max_steps": 50000,
        "save_steps": 5000,
        "eval_steps": 1000
    }
    
    # TPU-specific optimizations
    tpu_config = {
        "mesh_shape": (8,),  # Pure data parallel for credit efficiency
        "precision": "bfloat16",
        "opt_level": 3,  # Maximum XLA optimization
        "dynamic_batch_size": True,  # Adjust batch size based on memory
        "rematerialization": True,  # Trade compute for memory
        "attention_layout": "memory_efficient"
    }
    
    # Memory optimizations
    memory_config = {
        "use_gradient_checkpointing": True,
        "checkpoint_every_n_layers": 3,
        "use_memory_efficient_attention": True,
        "use_fp8_kv_cache": True,
        "activation_dtype": "bfloat16",
        "param_dtype": "bfloat16"
    }
    
    # Monitoring configuration
    monitoring_config = {
        "log_steps": 100,
        "profile_steps": 1000,
        "memory_profile_steps": 500,
        "track_credit_usage": True
    }
    
    def get_effective_batch_size(self) -> int:
        """Get effective global batch size."""
        return (self.training_config["batch_size"] * 
                self.training_config["grad_accum_steps"] * 
                8)  # 8 TPU cores
    
    def get_estimated_credits_per_step(self) -> float:
        """Estimate credit usage per training step."""
        # TPU v3 pricing ~$0.35/core hour
        hours_per_step = 1 / (3600 * self.get_steps_per_hour())
        return hours_per_step * 8 * 0.35  # 8 cores
    
    def get_steps_per_hour(self) -> float:
        """Estimate training steps per hour based on configuration."""
        # Conservative estimate based on model size and batch size
        base_tokens_per_sec = 20000  # Base throughput for 768-hidden model
        hidden_size_factor = 768 / self.model_config["hidden_size"]
        batch_size_factor = self.training_config["batch_size"] / 32
        
        steps_per_sec = (base_tokens_per_sec * hidden_size_factor * batch_size_factor) / (
            self.training_config["batch_size"] * 
            self.model_config["max_position_embeddings"]
        )
        return steps_per_sec * 3600  # Convert to steps/hour