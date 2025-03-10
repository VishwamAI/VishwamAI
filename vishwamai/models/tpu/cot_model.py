"""
TPU-optimized Chain of Thought (CoT) model using JAX/XLA
"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Optional, Dict, Tuple

from .transformer import TransformerComputeLayerTPU, TokenEmbedding
from .kernel_layers import TPUGEMMLinear, TPULayerNorm
from .core import DTYPE_CONFIG

def generate_cot(model: 'CoTModelTPU', input_ids: jnp.ndarray, max_length: int = 512,
                temperature: float = 0.6, top_p: float = 0.95) -> jnp.ndarray:
    """Generate CoT output with nucleus sampling"""
    # Ensure input is int32 for embedding
    input_ids = input_ids.astype(DTYPE_CONFIG['embedding_dtype'])
    return model.generate_cot(input_ids, max_length, temperature, top_p)

class CoTModelTPU(hk.Module):
    def __init__(self, embed_dim: int = 512, num_layers: int = 12,
                 num_heads: int = 8, ff_dim: int = 2048,
                 vocab_size: int = 50000, max_seq_len: int = 512,
                 num_experts: int = 7, dropout_rate: float = 0.1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.num_experts = num_experts
        self.dropout_rate = dropout_rate

    def __call__(self, input_ids: jnp.ndarray, target_ids: Optional[jnp.ndarray] = None,
                 is_training: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        # Ensure input is int32 for embedding
        input_ids = input_ids.astype(DTYPE_CONFIG['embedding_dtype'])
        
        # Token embeddings using TPU-optimized embedding
        embeddings = TokenEmbedding(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim
        )(input_ids)

        # Positional encoding
        positions = jnp.arange(input_ids.shape[1])[None, :]
        pos_encoding = self._create_positional_encoding(
            positions, self.embed_dim
        )
        pos_encoding = pos_encoding.astype(DTYPE_CONFIG['compute_dtype'])
        x = embeddings + pos_encoding

        # Dropout during training
        if is_training:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        # Process through transformer layers
        attention_mask = self._create_attention_mask(input_ids)
        attention_mask = attention_mask.astype(DTYPE_CONFIG['compute_dtype'])
        
        for _ in range(self.num_layers):
            layer = TransformerComputeLayerTPU(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout_rate=self.dropout_rate
            )
            x = layer(x, mask=attention_mask, is_training=is_training)

        # Output projection
        logits = TPUGEMMLinear(self.vocab_size)(x)

        # If training, compute loss
        loss = None
        if target_ids is not None and is_training:
            target_ids = target_ids.astype(DTYPE_CONFIG['embedding_dtype'])
            loss = self._compute_loss(logits, target_ids)

        return logits, loss

    def _create_attention_mask(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        # Create causal mask for auto-regressive decoding
        seq_len = input_ids.shape[1]
        mask = jnp.triu(jnp.ones((seq_len, seq_len)), k=1)
        return jnp.where(mask == 0, 1.0, 0.0)

    def _compute_loss(self, logits: jnp.ndarray, target_ids: jnp.ndarray) -> jnp.ndarray:
        # Compute cross entropy loss
        targets_onehot = jax.nn.one_hot(target_ids, self.vocab_size)
        targets_onehot = targets_onehot.astype(DTYPE_CONFIG['compute_dtype'])
        loss = optax.softmax_cross_entropy(logits, targets_onehot)
        return jnp.mean(loss)

    def _create_positional_encoding(self, positions: jnp.ndarray, d_model: int) -> jnp.ndarray:
        # Create sinusoidal position encoding
        angle_rads = self._get_angles(
            positions,
            jnp.arange(d_model)[None, :],
            d_model
        )

        angle_rads = angle_rads.at[:, 0::2].set(jnp.sin(angle_rads[:, 0::2]))
        angle_rads = angle_rads.at[:, 1::2].set(jnp.cos(angle_rads[:, 1::2]))

        return angle_rads

    def _get_angles(self, pos: jnp.ndarray, i: jnp.ndarray, d_model: int) -> jnp.ndarray:
        angle_rates = 1 / jnp.power(10000, (2 * (i // 2)) / d_model)
        return pos * angle_rates

    def generate_cot(self, input_ids: jnp.ndarray, max_length: int = 512,
                    temperature: float = 0.6, top_p: float = 0.95) -> jnp.ndarray:
        """Generate CoT output with nucleus sampling"""
        # Ensure input is int32 for embedding
        input_ids = input_ids.astype(DTYPE_CONFIG['embedding_dtype'])
        batch_size = input_ids.shape[0]
        generated = input_ids

        @jax.jit
        def sample_step(state: jnp.ndarray, _):
            logits, _ = self(state, is_training=False)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Compute cumulative probabilities for nucleus sampling
            sorted_logits, sorted_indices = jax.lax.top_k(
                next_token_logits, k=self.vocab_size
            )
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
            
            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove = jnp.concatenate([
                jnp.zeros_like(sorted_indices_to_remove[:, :1]),
                sorted_indices_to_remove[:, :-1]
            ], axis=-1)
            
            # Sample from filtered distribution
            next_token_logits = jnp.where(
                sorted_indices_to_remove,
                -1e10,
                sorted_logits
            )
            probs = jax.nn.softmax(next_token_logits)
            next_token = jax.random.categorical(
                hk.next_rng_key(),
                next_token_logits,
                shape=(batch_size,)
            )
            next_token = next_token.astype(DTYPE_CONFIG['embedding_dtype'])
            
            # Update state
            return jnp.concatenate([state, next_token[:, None]], axis=1)

        # Generate tokens
        generated = jax.lax.fori_loop(
            0,
            max_length - input_ids.shape[1],
            sample_step,
            generated
        )

        return generated

# Example usage and test
if __name__ == "__main__":
    def run_cot(x: jnp.ndarray) -> jnp.ndarray:
        model = CoTModelTPU()
        return model(x)[0]

    # Initialize with proper dtype
    batch_size, seq_len = 2, 64
    rng = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 50000, dtype=jnp.int32)

    # Transform and initialize
    transformed = hk.transform(run_cot)
    params = transformed.init(rng, input_ids)

    # Forward pass
    logits = transformed.apply(params, rng, input_ids)
    print("CoT Output shape:", logits.shape)