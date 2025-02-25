import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
from typing import Any, Dict, Optional, Tuple, Union, List
import safetensors.flax as stf
import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for VishwamAI transformer model."""
    vocab_size: int = 129280
    hidden_size: int = 7168
    num_layers: int = 61
    num_attention_heads: int = 128
    intermediate_size: int = 18432
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 163840
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    use_flash_attention: bool = True
    use_rope: bool = True
    use_alibi: bool = False
    use_gqa: bool = True
    num_key_value_heads: int = 128
    dtype: str = "bfloat16"
    
    def __post_init__(self):
        self.dtype = jnp.bfloat16 if self.dtype == "bfloat16" else jnp.float32

class FlaxAttention(nn.Module):
    """Multi-head attention mechanism with optional grouped query attention."""
    config: ModelConfig
    
    def setup(self):
        config = self.config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads 
        
        self.query = nn.Dense(self.hidden_size, dtype=config.dtype)
        self.key = nn.Dense(self.num_key_value_heads * self.head_dim, dtype=config.dtype)
        self.value = nn.Dense(self.num_key_value_heads * self.head_dim, dtype=config.dtype)
        self.output = nn.Dense(self.hidden_size, dtype=config.dtype)
        self.dropout = nn.Dropout(rate=config.attention_dropout_prob)
    
    def __call__(
        self, 
        hidden_states, 
        attention_mask=None, 
        deterministic=True, 
        output_attentions=False
    ):
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
        key_states = key_states.reshape(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        value_states = value_states.reshape(batch_size, seq_length, self.num_key_value_heads, self.head_dim)
        
        # Implement GQA when num_kv_heads != num_heads
        if self.num_key_value_heads != self.num_heads:
            key_states = jnp.repeat(
                key_states, 
                repeats=self.num_heads // self.num_key_value_heads, 
                axis=2
            )
            value_states = jnp.repeat(
                value_states,
                repeats=self.num_heads // self.num_key_value_heads,
                axis=2
            )
        
        # Transpose for attention computation
        query_states = jnp.transpose(query_states, axes=(0, 2, 1, 3))
        key_states = jnp.transpose(key_states, axes=(0, 2, 1, 3))
        value_states = jnp.transpose(value_states, axes=(0, 2, 1, 3))
        
        # Compute attention scores
        attention_scores = jnp.matmul(query_states, jnp.transpose(key_states, axes=(0, 1, 3, 2)))
        attention_scores = attention_scores / jnp.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax normalization
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, deterministic=deterministic)
        
        # Apply attention weights to values
        attention_output = jnp.matmul(attention_weights, value_states)
        
        # Reshape and project back to hidden size
        attention_output = jnp.transpose(attention_output, axes=(0, 2, 1, 3))
        attention_output = attention_output.reshape(batch_size, seq_length, self.hidden_size)
        attention_output = self.output(attention_output)
        
        outputs = (attention_output,)
        if output_attentions:
            outputs = outputs + (attention_weights,)
        
        return outputs

class FlaxFeedForward(nn.Module):
    """Feed-forward layer implementation."""
    config: ModelConfig
    
    def setup(self):
        config = self.config
        self.intermediate = nn.Dense(config.intermediate_size, dtype=config.dtype)
        self.output = nn.Dense(config.hidden_size, dtype=config.dtype)
        self.dropout = nn.Dropout(rate=config.hidden_dropout_prob)
        
    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.intermediate(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states

class FlaxTransformerLayer(nn.Module):
    """Single transformer layer with attention, feed-forward, residual connections, and layer norm."""
    config: ModelConfig
    
    def setup(self):
        config = self.config
        self.attention = FlaxAttention(config)
        self.feed_forward = FlaxFeedForward(config)
        self.attention_layernorm = nn.LayerNorm(epsilon=config.layer_norm_eps, dtype=config.dtype)
        self.ffn_layernorm = nn.LayerNorm(epsilon=config.layer_norm_eps, dtype=config.dtype)
        self.dropout = nn.Dropout(rate=config.hidden_dropout_prob)
        
    def __call__(
        self, 
        hidden_states, 
        attention_mask=None, 
        deterministic=True, 
        output_attentions=False
    ):
        # Self-attention block
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        attn_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        
        # Feed-forward block
        residual = hidden_states
        hidden_states = self.ffn_layernorm(hidden_states)
        hidden_states = self.feed_forward(hidden_states, deterministic=deterministic)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (attn_outputs[1],)
        
        return outputs

class FlaxTransformerEncoder(nn.Module):
    """Stack of transformer layers."""
    config: ModelConfig
    
    def setup(self):
        self.layers = [
            FlaxTransformerLayer(self.config) 
            for _ in range(self.config.num_layers)
        ]
        
    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                deterministic=deterministic,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)

class VishwamAIModel(nn.Module):
    """VishwamAI transformer model implementation."""
    config: ModelConfig
    
    def setup(self):
        config = self.config
        self.embed_dim = config.hidden_size
        
        # Token embeddings
        self.word_embeddings = nn.Embed(
            num_embeddings=config.vocab_size,
            features=self.embed_dim,
            dtype=config.dtype
        )
        
        # Positional embeddings - RoPE is applied in attention
        if config.use_rope:
            # In RoPE, we don't need a separate embedding table but rather
            # compute the positional encoding inside the attention calculation
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embed(
                num_embeddings=config.max_position_embeddings,
                features=self.embed_dim,
                dtype=config.dtype
            )
            
        self.encoder = FlaxTransformerEncoder(config)
        self.final_layernorm = nn.LayerNorm(epsilon=config.layer_norm_eps, dtype=config.dtype)
        self.dropout = nn.Dropout(rate=config.hidden_dropout_prob)
        
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        deterministic=True,
        output_attentions=False,
        output_hidden_states=False
    ):
        input_shape = input_ids.shape
        batch_size, seq_length = input_shape
        
        # Compute embeddings
        inputs_embeds = self.word_embeddings(input_ids)
        
        # Add positional embeddings if not using RoPE
        if self.position_embeddings is not None:
            if position_ids is None:
                position_ids = jnp.arange(seq_length)[None, :].repeat(batch_size, axis=0)
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = inputs_embeds + position_embeds
        else:
            hidden_states = inputs_embeds
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        
        # Process attention mask
        if attention_mask is not None:
            # Make causal mask for decoder
            attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
            attention_mask = 10000.0 * (1.0 - attention_mask)
        
        # Run through transformer layers
        encoder_outputs = self.encoder(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        hidden_states = encoder_outputs[0]
        hidden_states = self.final_layernorm(hidden_states)
        
        # Create outputs tuple
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (encoder_outputs[1],)
        if output_attentions:
            outputs = outputs + (encoder_outputs[2],)
        
        return outputs
    
    def load_weights(self, model_path, reduced_size=False):
        """Load model weights from the given path."""
        try:
            if os.path.isfile(model_path):
                # Load a single file
                params = stf.load_file(model_path)
                self.params = params
                print(f"Loaded model from {model_path}")
            elif os.path.isdir(model_path):
                # Load from sharded files
                file_pattern = "model-*.safetensors"
                model_files = [f for f in os.listdir(model_path) 
                             if f.startswith("model-") and f.endswith(".safetensors")]
                
                # Sort to ensure correct order
                model_files.sort()
                
                # If reduced_size, take only the first few files
                if reduced_size:
                    model_files = model_files[:5]
                    
                params = {}
                for model_file in model_files:
                    file_path = os.path.join(model_path, model_file)
                    file_params = stf.load_file(file_path)
                    params.update(file_params)
                    
                self.params = params
                print(f"Loaded model from {len(model_files)} files in {model_path}")
            else:
                raise ValueError(f"Path {model_path} is not a file or directory")
                
        except Exception as e:
            raise ValueError(f"Error loading model from {model_path}: {str(e)}")
        
    def save_weights(self, save_path, shard=True, num_shards=16):
        """Save model weights to the given path."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if not shard:
            # Save as a single file
            stf.save_file(self.params, save_path)
            return
            
        # Save as sharded files
        params_list = list(self.params.items())
        params_per_shard = len(params_list) // num_shards + 1
        
        for i in range(num_shards):
            start_idx = i * params_per_shard
            end_idx = min((i + 1) * params_per_shard, len(params_list))
            
            if start_idx >= len(params_list):
                break
                
            shard_params = dict(params_list[start_idx:end_idx])
            shard_path = f"{save_path}-{i+1:05d}-of-{num_shards:05d}.safetensors"
            stf.save_file(shard_params, shard_path)
        
        print(f"Saved model as {num_shards} shards with prefix {save_path}")
    
    @property
    def param_count(self):
        """Return the number of parameters in the model."""
        return sum(x.size for x in jax.tree_leaves(self.params))
