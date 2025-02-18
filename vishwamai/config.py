from dataclasses import dataclass, field
from typing import Literal, Any, Dict, Optional, Union

@dataclass
class ModelArgs:
    """Model configuration arguments."""
    
    # Basic model configuration
    max_batch_size: int = field(default=8)
    max_seq_len: int = field(default=4096 * 4)
    dtype: Literal["bf16", "fp8"] = field(default="bf16")
    vocab_size: int = field(default=102400)
    dim: int = field(default=2048)
    inter_dim: int = field(default=10944)
    
    # Layer configuration
    n_layers: int = field(default=27)
    n_dense_layers: int = field(default=1)
    n_heads: int = field(default=16)
    
    # MoE configuration
    moe_inter_dim: int = field(default=1408)
    n_routed_experts: int = field(default=64)
    n_shared_experts: int = field(default=2)
    n_activated_experts: int = field(default=6)
    n_expert_groups: int = field(default=1)
    n_limited_groups: int = field(default=1)
    score_func: Literal["softmax", "sigmoid"] = field(default="softmax")
    route_scale: float = field(default=1.0)
    
    # Attention configuration
    q_lora_rank: int = field(default=0)
    kv_lora_rank: int = field(default=512)
    qk_nope_head_dim: int = field(default=128)
    qk_rope_head_dim: int = field(default=64)
    v_head_dim: int = field(default=128)
    original_seq_len: int = field(default=4096)
    
    # Scaling parameters
    rope_theta: float = field(default=10000.0)
    rope_factor: float = field(default=40.0)
    beta_fast: int = field(default=32)
    beta_slow: int = field(default=1)
    mscale: float = field(default=1.0)
    
    # Training features
    use_alibi: bool = field(default=True)
    use_rope_scaling: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
    parallel_attn: bool = field(default=True)
    rope_condense_ratio: float = field(default=1.0)
    
    # Additional features
    ethical_framework_enabled: bool = field(default=True)
    emergent_behavior_enabled: bool = field(default=True)
    curriculum_learning_enabled: bool = field(default=True)
    cache_augmentation_enabled: bool = field(default=True)
    integrated_information_enabled: bool = field(default=True)
    
    # Training configuration
    max_steps: int = field(default=100000)
    
    # Training specific parameters
    early_stop_patience: int = 1000
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-5
    warmup_steps: int = 2000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    batch_size: int = 32
    accumulation_steps: int = 1
    
    # Add gradient checkpointing configuration
    enable_gradient_checkpointing: bool = True
    gradient_checkpointing_ratio: float = 0.5
    use_memory_efficient_attention: bool = True

    def __post_init__(self):
        """Validate and convert attributes after initialization."""
        # Ensure integer types
        self.max_seq_len = int(self.max_seq_len)
        self.dim = int(self.dim)
        self.n_layers = int(self.n_layers)
        self.n_heads = int(self.n_heads)
        self.vocab_size = int(self.vocab_size)
        self.max_steps = int(self.max_steps)
        
        # Validate values
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.dim > 0, "dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.vocab_size > 0, "vocab_size must be positive"
    
    # Override dictionary-style access
    def get(self, key: str, default: Any = None) -> Any:
        """Support dictionary-style access with defaults."""
        return getattr(self, key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access."""
        return getattr(self, key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelArgs to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    # Optional: Add a helper to check if attribute exists
    def has(self, key):
        return hasattr(self, key)

    def validate(self):
        """Validate configuration parameters."""
        # Validate gradient checkpointing settings
        if not isinstance(self.enable_gradient_checkpointing, bool):
            raise ValueError("enable_gradient_checkpointing must be a boolean")
        if not 0.0 <= self.gradient_checkpointing_ratio <= 1.0:
            raise ValueError("gradient_checkpointing_ratio must be between 0 and 1")
