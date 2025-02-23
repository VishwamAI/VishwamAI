"""Configuration classes for transformer models."""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union

@dataclass
class TransformerConfig:
    """Base configuration for transformer models."""
    
    # Architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    
    # Regularization
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    classifier_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Training
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_cache: bool = True
    
    # Positional embeddings
    max_position_embeddings: int = 2048
    position_embedding_type: str = "rotary"
    rotary_dim: Optional[int] = None
    
    # Data processing
    vocab_size: int = 50257
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Additional fields
    model_type: str = "transformer"
    architectures: List[str] = field(default_factory=lambda: ["TransformerModel"])

@dataclass
class MoEMLAConfig(TransformerConfig):
    """Configuration for MoE-MLA transformer models."""
    
    # MoE settings
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    router_jitter_noise: float = 0.1
    router_dropout_prob: float = 0.1
    expert_dropout_prob: float = 0.1
    moe_layer_position: str = "post_attention"
    use_expert_choice: bool = True
    share_expert_params: bool = False
    router_z_loss_coef: float = 0.001
    router_aux_loss_coef: float = 0.001
    
    # MLA settings
    num_attention_levels: int = 3
    level_scale_factor: float = 0.5
    min_layers_per_level: int = 1
    use_adaptive_computation: bool = True
    computation_threshold: float = 0.1
    layer_temperature: float = 1.0
    
    # Residual settings
    use_adaptive_residual: bool = True
    use_layer_scale: bool = True
    layer_scale_init: float = 0.1
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Update model type and architectures
        self.model_type = "moe_mla_transformer"
        self.architectures = ["MoEMLATransformerModel"]
        
        # Validate MoE settings
        if self.num_experts_per_token > self.num_experts:
            raise ValueError(
                f"Number of experts per token ({self.num_experts_per_token}) "
                f"cannot exceed total number of experts ({self.num_experts})"
            )
            
        # Adjust rotary dimension if not specified
        if self.rotary_dim is None:
            self.rotary_dim = self.hidden_size // self.num_attention_heads
            
        # Validate MLA settings
        if self.num_attention_levels > self.num_hidden_layers:
            raise ValueError(
                f"Number of attention levels ({self.num_attention_levels}) "
                f"cannot exceed number of layers ({self.num_hidden_layers})"
            )
            
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            key: getattr(self, key)
            for key in self.__annotations__
            if hasattr(self, key)
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MoEMLAConfig":
        """Create configuration from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            MoEMLAConfig instance
        """
        return cls(**{
            key: value 
            for key, value in config_dict.items()
            if key in cls.__annotations__
        })

def create_config_from_file(config_path: str) -> Union[TransformerConfig, MoEMLAConfig]:
    """Create configuration from JSON/YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration instance
    """
    import json
    import yaml
    
    # Load configuration file
    if config_path.endswith('.json'):
        with open(config_path) as f:
            config_dict = json.load(f)
    elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path}")
        
    # Create configuration based on model type
    model_type = config_dict.get('model_type', 'transformer')
    if model_type == 'moe_mla_transformer':
        return MoEMLAConfig.from_dict(config_dict)
    else:
        return TransformerConfig(**config_dict)

# Common configurations
SMALL_CONFIG = {
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "intermediate_size": 3072,
}

BASE_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
}

LARGE_CONFIG = {
    "hidden_size": 1536,
    "num_hidden_layers": 36,
    "num_attention_heads": 24,
    "intermediate_size": 6144,
}

XL_CONFIG = {
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "intermediate_size": 8192,
}
