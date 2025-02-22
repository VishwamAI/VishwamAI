"""
Model configuration classes for Vishwamai
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, Literal

class PrecisionMode(str, Enum):
    """Available precision modes"""
    FP16 = "fp16"
    FP32 = "fp32"
    FP64 = "fp64"
    BF16 = "bf16"
    TF32 = "tf32"

@dataclass
class ConfigMixin:
    """Mixin class for common configuration functionality"""
    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access"""
        return getattr(self, key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style setting"""
        setattr(self, key, value)
        
    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get with default"""
        return getattr(self, key, default)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigMixin':
        """Create from dictionary"""
        return cls(**config_dict)

@dataclass
class PrecisionConfig(ConfigMixin):
    """Precision configuration"""
    mode: PrecisionMode = PrecisionMode.FP16
    mixed_precision: bool = True
    gradient_precision: Literal["fp16", "fp32", "fp64"] = "fp32"
    static_loss_scaling: Optional[float] = None

@dataclass
class T4Config(ConfigMixin):
    """T4-specific model configuration"""
    hidden_size: int = 768
    num_layers: int = 8
    num_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)

@dataclass
class TreePlannerConfig(ConfigMixin):
    """Tree planner configuration"""
    enabled: bool = True
    num_tree_layers: int = 6
    tree_hidden_size: int = 512
    max_tree_depth: int = 5

@dataclass
class InformationRetrievalConfig(ConfigMixin):
    """Information retrieval configuration"""
    enabled: bool = True
    search_query_generator: str = "transformer"
    max_queries_per_response: int = 3
    cache_size: int = 1000

@dataclass
class ModelConfig(ConfigMixin):
    """Base model configuration"""
    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    intermediate_size: int = 4096
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    
    # Advanced features
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    attention_type: str = "multi_head"
    position_embedding_type: str = "RoPE"
    normalization_type: str = "RMSNorm"
    
    # Component configurations
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    tree_planner: TreePlannerConfig = field(default_factory=TreePlannerConfig)
    information_retrieval: InformationRetrievalConfig = field(default_factory=InformationRetrievalConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        config_dict = {}
        for k, v in asdict(self).items():
            if k.startswith('_'):
                continue
            if isinstance(v, (PrecisionConfig, TreePlannerConfig, InformationRetrievalConfig)):
                config_dict[k] = v.to_dict()
            elif isinstance(v, PrecisionMode):
                config_dict[k] = v.value
            else:
                config_dict[k] = v
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary"""
        # Handle nested configs
        if "precision" in config_dict:
            config_dict["precision"] = PrecisionConfig.from_dict(config_dict["precision"])
        if "tree_planner" in config_dict:
            config_dict["tree_planner"] = TreePlannerConfig.from_dict(config_dict["tree_planner"])
        if "information_retrieval" in config_dict:
            config_dict["information_retrieval"] = InformationRetrievalConfig.from_dict(config_dict["information_retrieval"])
            
        return cls(**config_dict)

@dataclass
class TrainingConfig(ConfigMixin):
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Optimizer settings
    optimizer: str = "adamw"
    lr_scheduler: str = "linear"
    warmup_ratio: Optional[float] = None
    
    # Logging and saving
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
