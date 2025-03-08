# VishwamAI/configs/model_config.py
"""
Hyperparameters for VishwamAI models: Standard Transformer, ToT, CoT,
and teacher models for knowledge distillation (Phi-4, QwQ-32B).
"""

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

@dataclass
class GPUConfig:
    """GPU-specific configuration settings"""
    use_tensor_cores: bool = True
    use_triton_kernels: bool = True
    memory_fraction: float = 0.95  # Reserve 5% for system
    enforce_deterministic: bool = False
    optimize_conv_ops: bool = True
    dynamic_memory_management: bool = True
    preferred_precision: Literal["fp32", "fp16", "bf16"] = "fp16"
    triton_optimization_level: int = 3
    block_sizes: Dict[str, int] = {
        "layernorm": 1024,
        "gelu": 1024,
        "matmul": 128
    }

@dataclass
class ModelConfig:
    """Core model configuration including GPU settings"""
    BATCH_SIZE: int = 128  # Large batch size for TPU efficiency
    LEARNING_RATE: float = 2e-5  # Optimized for transformer training
    NUM_EPOCHS: int = 10  # Default training epochs

    # Standard Transformer settings
    TRANSFORMER_LAYERS: int = 12  # Number of transformer layers
    TRANSFORMER_HEADS: int = 8  # Number of attention heads
    TRANSFORMER_DIM: int = 768  # Hidden dimension
    TRANSFORMER_DROPOUT: float = 0.1  # Dropout rate

    # Tree of Thoughts (ToT) settings
    TOT_LAYERS: int = 10  # Fewer layers for efficiency in exploration
    TOT_HEADS: int = 6  # Fewer heads for diverse reasoning paths
    TOT_DIM: int = 512  # Smaller dimension for ToT
    TOT_MAX_PATHS: int = 5  # Maximum reasoning paths to explore
    TOT_THRESHOLD: float = 0.7  # Confidence threshold for path selection

    # Chain of Thought (CoT) settings
    COT_LAYERS: int = 14  # More layers for sequential reasoning
    COT_HEADS: int = 10  # More heads for detailed step generation
    COT_DIM: int = 1024  # Larger dimension for CoT complexity
    COT_MAX_STEPS: int = 8  # Maximum reasoning steps

    # Teacher models for distillation
    TEACHER_MODELS: Dict[str, Dict[str, Any]] = {
        "phi-4": {
            "path": "Microsoft/phi-4",
            "layers": 32,  # Predefined by Phi-4
            "dim": 4096,  # Predefined by Phi-4
        },
        "qwq-32b": {
            "path": "Qwen/QwQ-32B-Preview",
            "layers": 40,  # Predefined by QwQ-32B
            "dim": 6144,  # Predefined by QwQ-32B
        }
    }
    DISTILLATION_ALPHA: float = 0.5  # Weight for teacher loss in distillation

    # Hardware optimization settings
    use_flash_attention: bool = True
    use_triton: bool = True
    use_tensor_cores: bool = True
    gpu_config: GPUConfig = GPUConfig()
    
    # Architecture settings
    use_geglu: bool = True
    use_rmsnorm: bool = True
    use_moe_layers: bool = True
    moe_config: Dict[str, Any] = {
        "num_experts": 8,
        "expert_capacity": 1.25,
        "min_expert_capacity": 4
    }

    @staticmethod
    def get_model_params(model_type: str) -> Dict[str, Any]:
        """Retrieve parameters for a specific model type."""
        if model_type == "transformer":
            return {
                "layers": ModelConfig.TRANSFORMER_LAYERS,
                "heads": ModelConfig.TRANSFORMER_HEADS,
                "dim": ModelConfig.TRANSFORMER_DIM,
                "dropout": ModelConfig.TRANSFORMER_DROPOUT
            }
        elif model_type == "tot":
            return {
                "layers": ModelConfig.TOT_LAYERS,
                "heads": ModelConfig.TOT_HEADS,
                "dim": ModelConfig.TOT_DIM,
                "max_paths": ModelConfig.TOT_MAX_PATHS,
                "threshold": ModelConfig.TOT_THRESHOLD
            }
        elif model_type == "cot":
            return {
                "layers": ModelConfig.COT_LAYERS,
                "heads": ModelConfig.COT_HEADS,
                "dim": ModelConfig.COT_DIM,
                "max_steps": ModelConfig.COT_MAX_STEPS
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware-specific configuration based on detected capabilities"""
        from vishwamai.models.kernel_layers import HardwareCapabilityDetector
        
        capabilities = HardwareCapabilityDetector.get_gpu_capabilities()
        config = {
            "use_tensor_cores": (
                self.use_tensor_cores and 
                capabilities.get("has_tensor_cores", False)
            ),
            "use_triton": (
                self.use_triton and
                capabilities.get("has_gpu", False)
            ),
            "preferred_precision": (
                "fp16" if capabilities.get("has_gpu", False) else "fp32"
            ),
            "memory_fraction": min(
                self.gpu_config.memory_fraction,
                0.95  # Cap at 95% for safety
            )
        }
        
        if capabilities.get("has_gpu"):
            # Adjust block sizes based on GPU memory
            total_mem = capabilities.get("total_memory", 0)
            if total_mem > 0:
                mem_gb = total_mem / (1024 ** 3)  # Convert to GB
                if mem_gb < 8:  # Low memory GPUs
                    config["block_sizes"] = {
                        "layernorm": 512,
                        "gelu": 512,
                        "matmul": 64
                    }
                elif mem_gb < 16:  # Mid-range GPUs
                    config["block_sizes"] = {
                        "layernorm": 1024,
                        "gelu": 1024,
                        "matmul": 128
                    }
                else:  # High-end GPUs
                    config["block_sizes"] = {
                        "layernorm": 2048,
                        "gelu": 2048,
                        "matmul": 256
                    }
        
        return config

    def optimize_for_hardware(self) -> None:
        """Update configuration based on available hardware"""
        hw_config = self.get_hardware_config()
        self.gpu_config.use_tensor_cores = hw_config["use_tensor_cores"]
        self.gpu_config.use_triton_kernels = hw_config["use_triton"]
        self.gpu_config.block_sizes = hw_config["block_sizes"]
        self.gpu_config.preferred_precision = hw_config["preferred_precision"]
        self.gpu_config.memory_fraction = hw_config["memory_fraction"]

if __name__ == "__main__":
    # Test the configuration
    print("Transformer Params:", ModelConfig.get_model_params("transformer"))
    print("ToT Params:", ModelConfig.get_model_params("tot"))
    print("CoT Params:", ModelConfig.get_model_params("cot"))