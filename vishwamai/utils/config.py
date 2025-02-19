"""
Configuration classes for VishwamAI components.
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """Configuration for the main VishwamAI model."""
    vocab_size: int = 50257
    hidden_size: int = 2048
    num_layers: int = 24
    num_heads: int = 16
    intermediate_size: int = 8192
    max_position_embeddings: int = 2048
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_moe: bool = False
    num_experts: int = 8
    expert_capacity: int = 128
    use_mla: bool = True
    use_memory: bool = False
    memory_size: int = 1024
    use_ethical_framework: bool = True
    enable_emergent: bool = True
    tree_search_depth: int = 3
    eos_token_id: int = 50256
    extra_configs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MLAConfig:
    """Configuration for Multi-Level Attention."""
    hidden_size: int = 2048
    num_heads: int = 16
    dropout: float = 0.1
    num_levels: int = 3
    combine_method: str = "concat"
    use_bias: bool = True

@dataclass
class EthicalConfig:
    """Configuration for ethical framework."""
    safety_threshold: float = 0.8
    content_filtering: bool = True
    bias_detection: bool = True
    harmful_content_threshold: float = 0.7
    ethical_guidelines: Dict[str, float] = field(default_factory=lambda: {
        "hate_speech": 0.9,
        "bias": 0.8,
        "toxicity": 0.85,
        "harm": 0.9
    })

@dataclass
class EmergentConfig:
    """Configuration for emergent behavior analysis."""
    pattern_detection_threshold: float = 0.6
    complexity_measure: str = "integrated_information"
    monitoring_interval: int = 100
    adaptation_rate: float = 0.1

@dataclass
class TreeConfig:
    """Configuration for Tree of Thoughts search."""
    max_depth: int = 3
    beam_width: int = 5
    max_steps_per_thought: int = 3
    temperature: float = 0.7
    reward_threshold: float = 0.5

@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    coherence_weight: float = 0.4
    novelty_weight: float = 0.3
    relevance_weight: float = 0.3
    diversity_bonus: float = 0.1
    length_penalty: float = 0.5

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    bf16: bool = False
    local_rank: int = -1
    seed: int = 42
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 1000

@dataclass
class OpenEndedConfig:
    """Configuration for open-ended learning."""
    exploration_rate: float = 0.2
    curiosity_weight: float = 0.3
    novelty_threshold: float = 0.5
    adaptation_steps: int = 1000

def load_config(config_path: str) -> ModelConfig:
    """Load configuration from file."""
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ModelConfig(**config_dict)

def save_config(config: ModelConfig, config_path: str):
    """Save configuration to file."""
    import yaml
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith('_')
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)

def update_config(config: ModelConfig, updates: Dict[str, Any]) -> ModelConfig:
    """Update configuration with new values."""
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith('_')
    }
    config_dict.update(updates)
    return ModelConfig(**config_dict)

def merge_configs(base_config: ModelConfig, override_config: Dict[str, Any]) -> ModelConfig:
    """Merge base configuration with override values."""
    config_dict = {
        k: v for k, v in base_config.__dict__.items()
        if not k.startswith('_')
    }
    config_dict.update(override_config)
    return ModelConfig(**config_dict)

def validate_config(config: ModelConfig) -> bool:
    """Validate configuration values."""
    try:
        assert config.hidden_size > 0, "hidden_size must be positive"
        assert config.num_layers > 0, "num_layers must be positive"
        assert config.num_heads > 0, "num_heads must be positive"
        assert config.hidden_size % config.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert 0 <= config.dropout < 1, "dropout must be between 0 and 1"
        if config.use_moe:
            assert config.num_experts > 0, "num_experts must be positive when use_moe is True"
        return True
    except AssertionError as e:
        print(f"Configuration validation failed: {str(e)}")
        return False
