"""Budget-optimized model configuration for TPU v3-8."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class BudgetModelConfig:
    """Credit-efficient configuration for budget model."""
    
    # Model configuration optimized for efficiency
    model_config = {
        "vocab_size": 32000,
        "hidden_size": 512,      # Reduced from 768 for efficiency
        "intermediate_size": 2048, # 4x hidden_size
        "num_attention_heads": 8,  # Reduced for efficiency
        "num_hidden_layers": 8,    # Reduced layer count
        "max_position_embeddings": 512,  # Shorter sequences for efficiency
        "attention_dropout": 0.1,
        "hidden_dropout": 0.1,
        "use_flash_attention": True,
        "use_fp8": True,
        "block_size": 128
    }
    
    # Training configuration for maximum efficiency
    training_config = {
        "batch_size": 32,     # Per TPU core
        "grad_accum_steps": 2,  # Effective batch size = 512
        "learning_rate": 5e-4,
        "warmup_steps": 500,
        "max_steps": 20000,
        "save_steps": 2000,
        "eval_steps": 500
    }
    
    # TPU-specific optimizations
    tpu_config = {
        "mesh_shape": (8,),  # Pure data parallel
        "precision": "bfloat16",
        "dynamic_batch_size": True,
        "rematerialization": True,
        "attention_layout": "memory_efficient"
    }
    
    # Aggressive memory optimizations
    memory_config = {
        "use_gradient_checkpointing": True,
        "checkpoint_every_n_layers": 2,
        "use_memory_efficient_attention": True,
        "use_fp8_kv_cache": True,
        "param_dtype": "bfloat16",
        "compute_dtype": "bfloat16"
    }
    
    # Credit usage monitoring
    monitoring_config = {
        "log_steps": 50,
        "profile_steps": 500,
        "memory_profile_steps": 200,
        "track_credit_usage": True,
        "automatic_resource_scaling": True
    }
    
    def get_parameter_count(self) -> int:
        """Get approximate parameter count."""
        h = self.model_config["hidden_size"]
        v = self.model_config["vocab_size"]
        l = self.model_config["num_hidden_layers"]
        i = self.model_config["intermediate_size"]
        
        # Embedding parameters
        emb_params = h * v
        
        # Transformer layer parameters
        layer_params = (
            # Self-attention
            4 * h * h +  # QKV projection + output
            # FFN
            2 * h * i +  # Two linear layers
            # Layer norms
            4 * h  # Two layer norms with scale and bias
        )
        
        # Total parameters
        total = emb_params + (layer_params * l)
        return total
    
    def get_theoretical_speedup(self) -> float:
        """Get theoretical speedup compared to base model."""
        base_params = 125_000_000  # Base model parameters
        budget_params = self.get_parameter_count()
        return base_params / budget_params
    
    def get_estimated_memory(self) -> float:
        """Get estimated memory usage in GB."""
        params = self.get_parameter_count()
        bytes_per_param = 2  # bfloat16
        activation_memory = (
            self.model_config["hidden_size"] *
            self.model_config["max_position_embeddings"] *
            self.training_config["batch_size"] *
            2  # Forward + backward pass
        )
        total_bytes = (params * bytes_per_param) + (activation_memory * bytes_per_param)
        return total_bytes / (1024 ** 3)  # Convert to GB