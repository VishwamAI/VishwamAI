from dataclasses import dataclass

@dataclass
class VishwamaiConfig:
    """Configuration class for Vishwamai model"""
    # Required parameters
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_hidden_layers: int
    
    # Optional parameters with defaults
    layer_norm_eps: float = 1e-5  # Default value for layer normalization epsilon
    max_position_embeddings: int = 512  # Default value for testing
    num_key_value_heads: int = None  # Default to None, will use num_attention_heads if not set
    rope_theta: float = 10000.0  # Default RoPE theta value
    max_seq_len: int = 2048  # Default max sequence length
    intermediate_size: int = 512  # Default intermediate size for MLP
    n_routed_experts: int = 4  # Default number of experts for routing
    n_activated_experts: int = 2  # Default number of activated experts per token
    route_scale: float = 1.0  # Default routing scale factor for expert routing
    device: str = "cpu"  # Default device for testing
    beta: float = 0.1  # Default beta value for GRPO
    group_size: int = 4  # Default group size for batch processing
    max_length: int = 512  # Maximum sequence length for processing
    
    # Device property for consistent access
    @property
    def torch_device(self) -> str:
        """Get the torch device string"""
        return self.device
    
    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
    
    @classmethod
    def get_test_config(cls):
        """Get a small config suitable for testing"""
        return cls(
            hidden_size=128,
            vocab_size=1000,
            num_attention_heads=4,
            num_hidden_layers=2,
            device="cpu",
            route_scale=1.0,
            beta=0.1,
            group_size=4,
            max_length=512,
            num_key_value_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
            max_seq_len=2048,
            n_routed_experts=4,
            n_activated_experts=2,
            rope_theta=10000.0,
            layer_norm_eps=1e-5
        )

@dataclass
class TokenizerConfig:
    """Configuration class for tokenizer"""
    max_length: int = 512
    vocab_size: int = 1000  # Add vocab_size with default matching test config
    pad_token_id: int = 0
    eos_token_id: int = 2
    unk_token_id: int = 3
