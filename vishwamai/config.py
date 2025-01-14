from dataclasses import dataclass
from typing import Optional, List
from transformers import PretrainedConfig

@dataclass
class ConceptualModelConfig:
    concept_dim: int = 512
    n_concepts: int = 1000
    concept_dropout: float = 0.1
    use_concept_attention: bool = True
    concept_layer_norm: bool = True
    num_concept_layers: int = 2

@dataclass
class VishwamaiConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `VishwamaiModel`."""
    model_type: str = "vishwamai"
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 102400
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    qk_dim: int = 128
    v_dim: int = 128
    rope_scaling: Optional[float] = None

    def __post_init__(self):
        """Validate and set up config after initialization."""
        self.n_kv_heads = self.n_kv_heads or self.n_heads

@dataclass
class GenerationConfig:
    max_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1
