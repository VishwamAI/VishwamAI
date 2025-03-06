import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np
import logging
from functools import partial
from .model import ModelConfig, VishwamAIModel

logger = logging.getLogger(__name__)

def create_causal_attention_mask(seq_length: int) -> jnp.ndarray:
    """Create causal attention mask optimized for TPU."""
    # Use bfloat16 for TPU efficiency
    mask = jnp.triu(jnp.ones((seq_length, seq_length)), k=1).astype(jnp.bfloat16)
    return -1e9 * mask[None, None, :, :]

class VishwamaiTransformer32B(VishwamAIModel):
    """32B parameter QwQ Transformer model with TPU optimizations."""
    
    @staticmethod
    def get_default_config():
        return ModelConfig(
            vocab_size=151936,     # QwQ-32B vocab size
            hidden_size=7168,      # QwQ-32B hidden size
            num_layers=60,         # QwQ-32B layers
            num_attention_heads=56, # QwQ-32B attention heads
            intermediate_size=28672,
            hidden_dropout_prob=0.1,
            attention_dropout_prob=0.1,
            max_position_embeddings=2048,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=True,
            gradient_checkpointing=True,
            use_flash_attention=True,
            use_rope=True,
            use_alibi=False,
            use_gqa=True,
            num_key_value_heads=8,  # QwQ-32B GQA heads
            dtype="bfloat16",  # Default to bfloat16 for TPU
            use_dualpipe=True  # Enable dualpipe-style pipelining
        )
    
    def setup(self):
        config = self.get_default_config()
        self.config = config
        
        # Initialize layers with TPU optimization
        self.embeddings = nn.Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=jnp.dtype(config.dtype)
        )
        
        # Position embeddings with TPU optimization
        self.position_embeddings = nn.Embed(
            num_embeddings=config.max_position_embeddings,
            features=config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=config.initializer_range),
            dtype=jnp.dtype(config.dtype)
        )
        
        # Transformer layers with dualpipe support
        num_partitions = min(config.num_layers, jax.device_count())
        layers_per_partition = config.num_layers // num_partitions
        
        self.layers = []
        for i in range(config.num_layers):
            partition_idx = i // layers_per_partition
            self.layers.append(
                TransformerBlock(
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    intermediate_size=config.intermediate_size,
                    dropout_rate=config.hidden_dropout_prob,
                    attention_dropout=config.attention_dropout_prob,
                    dtype=config.dtype,
                    use_dualpipe=config.use_dualpipe,
                    partition_idx=partition_idx,
                    name=f'layer_{i}'
                )
            )
        
        # Layer normalization
        self.ln_f = nn.LayerNorm(
            epsilon=config.layer_norm_eps,
            dtype=config.dtype,
            name='ln_f'
        )
        
        # Output head
        self.lm_head = nn.Dense(
            features=config.vocab_size,
            use_bias=False,
            dtype=config.dtype,
            name='lm_head'
        )

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, input_ids, attention_mask=None, deterministic=True):
        b, s = input_ids.shape
        
        # Get embeddings (TPU-optimized with bfloat16)
        hidden_states = self.embeddings(input_ids)
        
        # Add position embeddings
        position_ids = jnp.arange(s)[None, :].repeat(b, axis=0)
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        
        # Create attention mask for TPU
        if attention_mask is None:
            attention_mask = create_causal_attention_mask(s)
        
        # Process through transformer layers with dualpipe
        if self.config.use_dualpipe:
            # Split the batch for forward and backward passes
            batch_size = hidden_states.shape[0]
            split_point = batch_size // 2
            
            # Forward pass batch
            forward_states = hidden_states[:split_point]
            forward_mask = attention_mask[:split_point] if attention_mask is not None else None
            
            # Backward pass batch (processed simultaneously)
            backward_states = hidden_states[split_point:]
            backward_mask = attention_mask[split_point:] if attention_mask is not None else None
            
            # Process both passes through layers
            for layer in self.layers:
                forward_states = layer(forward_states, forward_mask, deterministic)
                backward_states = layer(backward_states, backward_mask, deterministic)
            
            # Combine results
            hidden_states = jnp.concatenate([forward_states, backward_states], axis=0)
        else:
            # Standard processing
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask, deterministic)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        outputs = {'hidden_states': hidden_states, 'logits': logits}
        return outputs

class TransformerBlock(nn.Module):
    """Transformer block with TPU and dualpipe optimizations."""
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    dropout_rate: float
    attention_dropout: float
    dtype: str = "bfloat16"
    use_dualpipe: bool = True
    partition_idx: int = 0
    
    def setup(self):
        self.attention = SelfAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            dropout_rate=self.attention_dropout,
            dtype=self.dtype
        )
        self.ln_1 = nn.LayerNorm(dtype=self.dtype)
        self.ln_2 = nn.LayerNorm(dtype=self.dtype)
        self.mlp = FeedForward(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            dtype=self.dtype
        )
        
        # Dualpipe buffers
        if self.use_dualpipe:
            self.forward_buffer = None
            self.backward_buffer = None
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x, attention_mask=None, deterministic=True):
        # Place computation on specific TPU core
        device = jax.devices("tpu")[self.partition_idx % jax.device_count()]
        with jax.default_device(device):
            # Self attention with TPU optimization
            attn_output = self.attention(
                self.ln_1(x),
                attention_mask,
                deterministic=deterministic
            )
            x = x + attn_output
            
            # Feed-forward with dualpipe if enabled
            if self.use_dualpipe:
                mlp_output = self._dualpipe_mlp(x, deterministic)
            else:
                mlp_output = self.mlp(
                    self.ln_2(x),
                    deterministic=deterministic
                )
            
            return x + mlp_output
    
    def _dualpipe_mlp(self, x, deterministic):
        """Implement dualpipe-style processing for MLP."""
        # Split input for forward/backward processing
        batch_size = x.shape[0]
        split_point = batch_size // 2
        
        # Process both halves simultaneously
        forward_x = self.ln_2(x[:split_point])
        backward_x = self.ln_2(x[split_point:])
        
        # Run MLP on both streams
        forward_output = self.mlp(forward_x, deterministic)
        backward_output = self.mlp(backward_x, deterministic)
        
        # Combine results
        return jnp.concatenate([forward_output, backward_output], axis=0)

class SelfAttention(nn.Module):
    """Multi-head self attention with TPU optimizations."""
    hidden_size: int
    num_attention_heads: int
    dropout_rate: float
    dtype: str = "bfloat16"
    
    def setup(self):
        self.head_size = self.hidden_size // self.num_attention_heads
        # Initialize QKV as a single projection for TPU efficiency
        self.qkv = nn.Dense(features=3 * self.hidden_size, dtype=self.dtype)
        self.o = nn.Dense(features=self.hidden_size, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x, attention_mask=None, deterministic=True):
        b, s, h = x.shape
        
        # Fused QKV projection for TPU efficiency
        qkv = self.qkv(x)
        qkv = qkv.reshape(b, s, 3, self.num_attention_heads, self.head_size)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = q.squeeze(2)  # Remove the split dimension
        k = k.squeeze(2)
        v = v.squeeze(2)
        
        # Compute attention scores with TPU optimization
        scale = 1.0 / jnp.sqrt(self.head_size)
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k).astype(self.dtype) * scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        # Get attention weights
        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, deterministic=deterministic)
        
        # Apply attention to values with TPU optimization
        attn = jnp.einsum('bhqk,bkhd->bqhd', weights, v).astype(self.dtype)
        attn = attn.reshape(b, s, h)
        
        return self.o(attn)

class FeedForward(nn.Module):
    """Feed-forward neural network with TPU optimizations."""
    hidden_size: int
    intermediate_size: int
    dropout_rate: float
    dtype: str = "bfloat16"
    
    def setup(self):
        self.fc1 = nn.Dense(features=self.intermediate_size, dtype=self.dtype)
        self.fc2 = nn.Dense(features=self.hidden_size, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        
    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, x, deterministic=True):
        # Forward pass with TPU optimization
        x = self.fc1(x)
        x = jax.nn.gelu(x)
        x = self.dropout(x, deterministic=deterministic)
        x = self.fc2(x)
        x = self.dropout(x, deterministic=deterministic)
        return x
