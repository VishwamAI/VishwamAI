"""
Model arguments and configuration classes for Vishwamai
"""
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any
import math
import warnings

from .expert_config import (
    UnifiedConfig,
    ExpertConfig,
    ParallelConfig,
    AdvancedMLPConfig,
    AdvancedTransformerConfig,
    ParallelLinearConfig,
    UNIFIED_BASE,
    UNIFIED_EXPERT,
    UNIFIED_PARALLEL
)

@dataclass
class ModelArgs:
    """
    Model arguments defining the architecture and behavior of Vishwamai models.

    Attributes:
        # Model Architecture
        max_batch_size: int = 32
        max_seq_len: int = 4096
        dtype: Literal["fp16", "fp32", "fp64", "bf16", "fp8"] = "fp16"
        vocab_size: int = 32000
        hidden_size: int = 2048
        intermediate_size: int = 8192
        num_attention_heads: int = 32
        num_hidden_layers: int = 32
        
        # Advanced Components
        unified: UnifiedConfig = field(default_factory=UnifiedConfig)
        
        # Attention Configuration
        attention_dropout: float = 0.0
        hidden_dropout: float = 0.0
        attention_bias: bool = False
        position_embedding_type: Literal["learned", "rope", "alibi"] = "rope"
        max_position_embeddings: int = 4096
        
        # Training Configuration
        gradient_checkpointing: bool = False
        use_flash_attention: bool = True
        use_mixed_precision: bool = True
        gradient_accumulation_steps: int = 1
        
        # Optimization
        optimizer: Literal["adam", "adamw", "lion", "adafactor"] = "adamw"
        learning_rate: float = 1e-4
        weight_decay: float = 0.01
        beta1: float = 0.9
        beta2: float = 0.999
        epsilon: float = 1e-8
        max_grad_norm: float = 1.0
        
        # Expert Configuration (shorthand for unified.expert)
        num_experts: int = 0
        num_experts_per_token: int = 0
        expert_capacity: int = 0
        expert_dtype: Optional[Literal["fp16", "fp32", "bf16"]] = None
        
        # RoPE Configuration
        rope_scaling: Optional[Dict[str, Any]] = None
        rope_theta: float = 10000.0
        rope_scaling_factor: float = 1.0
        
        # Extra Features
        use_bias_in_mlp: bool = True
        use_parallel_attention: bool = False
        use_kernel_optimizations: bool = True
        tie_word_embeddings: bool = True
        
        # Initialization
        initializer_range: float = 0.02
        layer_norm_epsilon: float = 1e-6
        
        # Cache and Memory Management
        use_cache: bool = True
        prealloc_cache: bool = False
        max_cache_size: Optional[int] = None
    """
    
    def __post_init__(self):
        """Validate and compute derived attributes after initialization"""
        # Compute derived values
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # Sync expert config with unified config
        if self.num_experts > 0:
            self.unified.expert.num_experts = self.num_experts
            self.unified.expert.num_experts_per_token = self.num_experts_per_token
            self.unified.expert.expert_capacity = self.expert_capacity
            self.unified.expert.expert_dtype = self.expert_dtype
            
        # Set platform-specific defaults
        if self.rope_scaling is None:
            self.rope_scaling = {
                "type": "linear",
                "factor": 1.0
            }
            
        # Validate configuration
        self.validate()
            
    def validate(self):
        """Validate configuration values"""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"Hidden size {self.hidden_size} must be divisible by number of attention heads {self.num_attention_heads}"
            
        assert self.max_seq_len <= self.max_position_embeddings, \
            f"Max sequence length {self.max_seq_len} cannot exceed max position embeddings {self.max_position_embeddings}"
            
        # Validate expert configuration
        if self.num_experts > 0:
            assert self.num_experts_per_token > 0, \
                "num_experts_per_token must be positive when using experts"
            assert self.expert_capacity > 0, \
                "expert_capacity must be positive when using experts"
                
        # Validate parallel configuration
        if self.unified.parallel.tensor_parallel_size > 1:
            try:
                import torch
                assert self.unified.parallel.tensor_parallel_size <= torch.cuda.device_count(), \
                    "tensor_parallel_size cannot exceed available GPU count"
            except ImportError:
                warnings.warn("PyTorch not available, skipping parallel validation")
                
        # Validate advanced features
        if self.use_flash_attention:
            try:
                import flash_attn
            except ImportError:
                warnings.warn("flash_attn not available, disabling flash attention")
                self.use_flash_attention = False
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        base_config = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_') and k != 'unified'
        }
        base_config['unified'] = self.unified.to_dict()
        return base_config
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelArgs':
        """Create configuration from dictionary"""
        unified_dict = config_dict.pop('unified', {})
        config = cls(**config_dict)
        config.unified = UnifiedConfig.from_dict(unified_dict)
        return config
        
    def update(self, **kwargs) -> 'ModelArgs':
        """Update configuration with new values"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif '.' in k:
                # Handle nested unified config updates
                section, param = k.split('.', 1)
                if section == 'unified':
                    config_dict = self.unified.to_dict()
                    if '.' in param:
                        subsection, subparam = param.split('.', 1)
                        if subsection in config_dict:
                            config_dict[subsection][subparam] = v
                    else:
                        for section_dict in config_dict.values():
                            if param in section_dict:
                                section_dict[param] = v
                    self.unified = UnifiedConfig.from_dict(config_dict)
            else:
                raise ValueError(f"Unknown configuration parameter: {k}")
                
        self.validate()
        return self
        
    def get_attention_config(self) -> Dict[str, Any]:
        """Get attention-specific configuration"""
        config = {
            "num_attention_heads": self.num_attention_heads,
            "hidden_size": self.hidden_size,
            "attention_dropout": self.attention_dropout,
            "attention_bias": self.attention_bias,
            "head_dim": self.head_dim,
            "use_flash_attention": self.use_flash_attention
        }
        config.update(self.unified.transformer.to_dict())
        return config
        
    def get_mlp_config(self) -> Dict[str, Any]:
        """Get MLP-specific configuration"""
        config = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "hidden_dropout": self.hidden_dropout,
            "use_bias": self.use_bias_in_mlp
        }
        config.update(self.unified.mlp.to_dict())
        return config
        
    def get_position_config(self) -> Dict[str, Any]:
        """Get position embedding configuration"""
        config = {
            "max_position_embeddings": self.max_position_embeddings,
            "position_embedding_type": self.position_embedding_type,
        }
        if self.position_embedding_type == "rope":
            config.update({
                "rope_theta": self.rope_theta,
                "rope_scaling": self.rope_scaling,
                "rope_scaling_factor": self.rope_scaling_factor
            })
        return config

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> 'ModelArgs':
        """Create configuration from pretrained model name"""
        # TODO: Implement model hub loading
        raise NotImplementedError("Loading from pretrained models not yet implemented")

# Model size presets
VISHWAMAI_TINY = ModelArgs(
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=6,
    unified=UNIFIED_BASE
)

VISHWAMAI_BASE = ModelArgs(
    hidden_size=768,
    intermediate_size=3072,
    num_attention_heads=12,
    num_hidden_layers=12,
    unified=UNIFIED_BASE
)

VISHWAMAI_LARGE = ModelArgs(
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=16,
    num_hidden_layers=24,
    unified=UNIFIED_BASE
)

# Specialized variants
VISHWAMAI_EXPERT = ModelArgs(
    hidden_size=1024,
    intermediate_size=4096,
    num_attention_heads=16,
    num_hidden_layers=24,
    num_experts=8,
    num_experts_per_token=2,
    expert_capacity=32,
    unified=UNIFIED_EXPERT
)

VISHWAMAI_PARALLEL = ModelArgs(
    hidden_size=2048,
    intermediate_size=8192,
    num_attention_heads=32,
    num_hidden_layers=32,
    unified=UNIFIED_PARALLEL
)
