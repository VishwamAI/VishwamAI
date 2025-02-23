"""Configuration management for transformer models."""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass
class TransformerConfig:
    """Configuration class for transformer models."""
    
    # Model Architecture
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_hidden_layers: int = 24
    intermediate_size: Optional[int] = None
    head_dim: Optional[int] = None
    
    # MoE Configuration
    num_moe_layers: int = 4
    moe_layer_frequency: int = 2
    num_experts: int = 8
    expert_capacity_factor: float = 1.25
    num_experts_per_token: int = 2
    expert_hidden_size: Optional[int] = None
    expert_dropout: float = 0.1
    
    # MLA Configuration
    use_mla: bool = True
    num_prev_layers: int = 4
    attention_window: int = 4
    normalize_cached: bool = True
    use_compressed_cache: bool = False
    compression_dim: Optional[int] = None
    
    # Attention Configuration
    use_flash_attention: bool = True
    use_rope: bool = True
    max_sequence_length: int = 2048
    attention_dropout: float = 0.1
    rope_scaling: Optional[float] = None
    
    # General Configuration
    hidden_dropout: float = 0.1
    drop_path: float = 0.0
    activation: str = "swiglu"
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_bias: bool = False
    dtype: str = "bfloat16"
    param_dtype: str = "bfloat16"
    
    # Router Configuration
    router_jitter_noise: float = 0.1
    router_dtype: str = "float32"
    gate_type: str = "top_k"
    gate_temperature: float = 0.1
    gate_noise_type: str = "multiplicative"
    gate_noise_scale: float = 1.0
    z_loss_scale: float = 0.01
    load_balance_scale: float = 0.01
    
    # Additional Parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TransformerConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            TransformerConfig instance
        """
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            
        # Extract known parameters
        known_params = {}
        extra_params = {}
        
        for key, value in config_dict.items():
            if key in cls.__dataclass_fields__:
                known_params[key] = value
            else:
                extra_params[key] = value
                
        # Create instance with known parameters
        config = cls(**known_params)
        config.extra_params = extra_params
        
        return config
        
    @classmethod
    def from_yaml_files(cls, config_files: List[str]) -> "TransformerConfig":
        """Load and merge multiple YAML configurations.
        
        Args:
            config_files: List of paths to YAML configuration files
            
        Returns:
            TransformerConfig instance
        """
        # Start with empty config
        merged_dict = {}
        
        # Load and merge all files
        for config_path in config_files:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
                merged_dict.update(config_dict)
                
        # Create config instance
        known_params = {}
        extra_params = {}
        
        for key, value in merged_dict.items():
            if key in cls.__dataclass_fields__:
                known_params[key] = value
            else:
                extra_params[key] = value
                
        config = cls(**known_params)
        config.extra_params = extra_params
        
        return config
        
    def save(self, output_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
        """
        # Convert to dictionary
        config_dict = {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if key != "extra_params"
        }
        config_dict.update(self.extra_params)
        
        # Save to file
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f)
            
    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Configuration dictionary
        """
        config_dict = {
            key: getattr(self, key)
            for key in self.__dataclass_fields__
            if key != "extra_params"
        }
        config_dict.update(self.extra_params)
        return config_dict
        
    def update(self, **kwargs) -> "TransformerConfig":
        """Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
            
        Returns:
            Updated configuration
        """
        for key, value in kwargs.items():
            if key in self.__dataclass_fields__:
                setattr(self, key, value)
            else:
                self.extra_params[key] = value
        return self
        
    @property
    def num_parameters(self) -> int:
        """Calculate approximate number of model parameters.
        
        Returns:
            Number of parameters
        """
        params = 0
        
        # Embedding parameters
        params += self.hidden_size * self.max_sequence_length  # Position embeddings
        
        # Standard transformer layers
        num_standard_layers = self.num_hidden_layers - self.num_moe_layers
        params_per_layer = (
            # Self-attention
            4 * self.hidden_size * self.hidden_size +  # Q,K,V,O
            # Feed-forward
            2 * self.hidden_size * (self.intermediate_size or self.hidden_size * 4)
        )
        params += num_standard_layers * params_per_layer
        
        # MoE layers
        expert_params = (
            2 * self.hidden_size * (self.expert_hidden_size or self.hidden_size * 4)
        )
        params += self.num_moe_layers * self.num_experts * expert_params
        
        return params
