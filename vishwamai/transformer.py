import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 128
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    use_flash_attention: bool = True
    use_rope: bool = True
    use_alibi: bool = False
    use_gqa: bool = True
    num_key_value_heads: int = 4
    dtype: str = "bfloat16"
    quantization: Optional[str] = None
    use_mod: bool = False  # Added to fix error

    def __post_init__(self):
        if self.use_gqa:
            assert self.num_attention_heads % self.num_key_value_heads == 0, \
                "num_attention_heads must be divisible by num_key_value_heads for GQA"

class VishwamAIModel(nn.Module):
    config: ModelConfig

    def setup(self):
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.dtype(self.config.dtype)
        )
        self.lm_head = nn.Dense(features=self.config.vocab_size, use_bias=False)

    def __call__(self, input_ids, attention_mask=None, deterministic=True, use_tot=False, tot_rng_key=None):
        hidden_states = self.embeddings(input_ids)
        outputs = {'hidden_states': hidden_states, 'logits': self.lm_head(hidden_states)}
        if use_tot and hasattr(self, 'tot_model') and tot_rng_key is not None:
            thought = self.tot_model(hidden_states, tot_rng_key)
            outputs['tot_outputs'] = {'thought': thought}
        return outputs

    def init(self, rng, input_ids):
        return self.init_weights(rng, input_ids.shape)