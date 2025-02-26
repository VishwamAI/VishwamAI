import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from einops import rearrange, repeat
import json
import os
import gc
from huggingface_hub import snapshot_download
import safetensors.flax as stf
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optimizer Creation (optimized for TPU v2-8)
def create_optimizer(learning_rate: float = 1e-4, weight_decay: float = 0.01, 
                     beta1: float = 0.9, beta2: float = 0.999, 
                     warmup_steps: int = 2000, num_train_steps: int = 100000):
    decay_scheduler = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )
    decay_scheduler = optax.join_schedules(
        schedules=[decay_scheduler, optax.linear_schedule(
            init_value=learning_rate,
            end_value=0,
            transition_steps=num_train_steps - warmup_steps
        )],
        boundaries=[warmup_steps]
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=decay_scheduler,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay
        )
    )
    return optimizer

# Model Configurations
@dataclass
class ModelArgs:
    dim: int = 768  # Adjusted to match error (12 * 64 = 768)
    n_layers: int = 32
    n_heads: int = 12  # Matches error-derived num_heads
    n_kv_heads: int = 8
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024  # Matches error-derived seq_len
    n_experts: int = 8
    expert_dim: int = 4096
    expert_pruning_threshold: float = 0.1
    min_active_experts: int = 4
    dynamic_expert_selection: bool = True
    expert_capacity_factor: float = 1.25
    window_size: int = 512
    global_tokens: int = 64
    attention_dropout: float = 0.1
    dropout_rate: float = 0.1
    expert_dropout: float = 0.1
    use_alibi: bool = True
    use_rope: bool = True
    num_alibi_heads: Optional[int] = None
    use_flash_attention: bool = True
    kv_cache_dtype: jnp.dtype = jnp.int8
    param_dtype: jnp.dtype = jnp.bfloat16
    vision_dim: int = 1024
    use_contrastive_loss: bool = True
    temperature: float = 0.07
    max_image_length: int = 256

@dataclass
class ModelConfig:
    @classmethod
    def map_config_params(cls, config_dict: Dict) -> Dict:
        mapped_dict = config_dict.copy()
        if 'attention_dropout' in mapped_dict:
            mapped_dict['attention_dropout_prob'] = mapped_dict.pop('attention_dropout')
        if 'dropout' in mapped_dict:
            mapped_dict['hidden_dropout_prob'] = mapped_dict.pop('dropout')
        if 'hidden_size' not in mapped_dict and 'dim' in mapped_dict:
            mapped_dict['hidden_size'] = mapped_dict.pop('dim')
        if 'num_attention_heads' not in mapped_dict and 'num_heads' in mapped_dict:
            mapped_dict['num_attention_heads'] = mapped_dict.pop('num_heads')
        if 'num_layers' not in mapped_dict and 'n_layers' in mapped_dict:
            mapped_dict['num_layers'] = mapped_dict.pop('n_layers')
        if 'intermediate_size' not in mapped_dict and 'intermediate_dim' in mapped_dict:
            mapped_dict['intermediate_size'] = mapped_dict.pop('intermediate_dim')
        mapped_dict.pop('attention_bias', None)
        return {k: v for k, v in mapped_dict.items() if k in cls.__dataclass_fields__}

    vocab_size: int = 32000
    hidden_size: int = 768  # Adjusted to match error
    num_layers: int = 32
    num_attention_heads: int = 12  # Matches error
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
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
    num_key_value_heads: int = 8
    dtype: str = "bfloat16"
    quantization: Optional[str] = None

    def __post_init__(self):
        if self.use_gqa:
            assert self.num_attention_heads % self.num_key_value_heads == 0, \
                "num_attention_heads must be divisible by num_key_value_heads for GQA"

# Model Components
class ParallelDense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    precision: Optional[tuple] = None
    kernel_init: callable = nn.initializers.normal(stddev=0.02)
    
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.dot(inputs, kernel, precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y

class RMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        return x * jnp.asarray(scale, self.dtype)

def precompute_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute frequencies for rotary positional encoding, optimized for TPU v2-8.
    
    Args:
        head_dim: Dimension of each attention head (e.g., 64)
        max_seq_len: Maximum sequence length (e.g., 1024)
        base: Base value for frequency computation
        dtype: Data type for TPU compatibility (bfloat16 recommended)
    """
    half_dim = head_dim // 2  # e.g., 32
    freqs = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=dtype) / half_dim))  # Shape: [32]
    t = jnp.arange(max_seq_len, dtype=dtype)  # Shape: [1024]
    freqs = jnp.outer(t, freqs)  # Shape: [1024, 32]
    sin = jnp.sin(freqs)  # Shape: [1024, 32]
    cos = jnp.cos(freqs)  # Shape: [1024, 32]
    # Ensure 4D shape persists for broadcasting
    sin = jnp.expand_dims(jnp.expand_dims(sin, 0), 0)  # Shape: [1, 1, 1024, 32]
    cos = jnp.expand_dims(jnp.expand_dims(cos, 0), 0)  # Shape: [1, 1, 1024, 32]
    return sin.astype(dtype), cos.astype(dtype)

def rotary_embedding(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray, use_local_repeat: bool = False) -> jnp.ndarray:
    """Apply rotary embeddings, optimized for TPU v2-8.
    
    Args:
        x: Input tensor [batch, heads, seq_len, head_dim]
        sin: Sine tensor [1, 1, seq_len, head_dim//2]
        cos: Cosine tensor [1, 1, seq_len, head_dim//2]
        use_local_repeat: Not used here (GQA-specific)
    """
    batch, heads, seq_len, head_dim = x.shape
    half_dim = head_dim // 2  # e.g., 32
    # Ensure sin and cos match seq_len and half_dim
    sin = sin[:, :, :seq_len, :half_dim]  # [1, 1, seq_len, 32]
    cos = cos[:, :, :seq_len, :half_dim]  # [1, 1, seq_len, 32]
    # Explicitly broadcast to match x1, x2 shape
    sin = jnp.broadcast_to(sin, (batch, heads, seq_len, half_dim))  # [1, 12, 1024, 32]
    cos = jnp.broadcast_to(cos, (batch, heads, seq_len, half_dim))  # [1, 12, 1024, 32]
    x1, x2 = jnp.split(x, 2, axis=-1)  # Each: [batch, heads, seq_len, 32]
    x_rot = x1 * cos - x2 * sin
    x_pass = x1 * sin + x2 * cos
    return jnp.concatenate([x_rot, x_pass], axis=-1)

class ParallelMLP(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        dim = self.config.dim
        hidden_dim = int(self.config.ffn_dim_multiplier * dim if self.config.ffn_dim_multiplier else dim * 4)
        x = RMSNorm(dtype=x.dtype)(x)
        gate = ParallelDense(hidden_dim, dtype=x.dtype)(x)
        up = ParallelDense(hidden_dim, dtype=x.dtype)(x)
        gate = nn.silu(gate)
        intermediate = gate * up
        output = ParallelDense(dim, dtype=x.dtype)(intermediate)
        return nn.Dropout(rate=self.config.dropout_rate)(output, deterministic=deterministic)

class MoELayer(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        batch_size, seq_len, dim = x.shape
        num_experts = self.config.n_experts
        expert_fn = ParallelMLP(self.config)
        router_weights = self.param('router_weights', nn.initializers.normal(stddev=0.02), (dim, num_experts))
        router_logits = jax.lax.stop_gradient(jnp.einsum('bsd,de->bse', x, router_weights))
        routing_probs = nn.softmax(router_logits, axis=-1)
        top2_weights, top2_indices = jax.lax.top_k(routing_probs, k=2)
        top2_weights = top2_weights / jnp.sum(top2_weights, axis=-1, keepdims=True)
        expert_inputs = []
        expert_indices = []
        for expert_idx in range(num_experts):
            expert_mask = (top2_indices == expert_idx).any(axis=-1)
            if jnp.any(expert_mask):
                mask_weights = jnp.where(top2_indices == expert_idx, top2_weights, 0).max(axis=-1)
                expert_inputs.append(x[expert_mask] * mask_weights[expert_mask, None])
                expert_indices.append(jnp.where(expert_mask)[0])
        expert_outputs = []
        if expert_inputs:
            batched_expert_input = jnp.concatenate(expert_inputs, axis=0)
            batched_expert_output = expert_fn(batched_expert_input, deterministic)
            offset = 0
            for expert_input in expert_inputs:
                expert_len = len(expert_input)
                expert_outputs.append((expert_indices[offset:offset + expert_len], batched_expert_output[offset:offset + expert_len]))
                offset += expert_len
        final_output = jnp.zeros_like(x)
        for indices, outputs in expert_outputs:
            final_output = final_output.at[indices].add(outputs)
        expert_usage = jnp.mean(routing_probs, axis=(0, 1))
        load_balancing_loss = -jnp.sum(expert_usage * jnp.log(expert_usage + 1e-6))
        return final_output, load_balancing_loss

def create_alibi_slopes(num_heads: int) -> jnp.ndarray:
    closest_power_of_2 = 2 ** jnp.floor(jnp.log2(num_heads))
    base = jnp.array([2 ** (-(2 ** -(jnp.log2(closest_power_of_2) - 3)))], dtype=jnp.float32)
    powers = jnp.arange(1, 1 + num_heads, dtype=jnp.float32)
    slopes = jnp.power(base, powers)
    return slopes

class MultiheadAttention(nn.Module):
    config: ModelArgs
    
    def create_sliding_window_mask(self, seq_len: int) -> jnp.ndarray:
        window_size = self.config.window_size
        global_tokens = self.config.global_tokens
        mask = jnp.full((seq_len, seq_len), -1e9)
        for i in range(seq_len):
            window_start = max(0, i - window_size // 2)
            window_end = min(seq_len, i + window_size // 2 + 1)
            mask = mask.at[i, window_start:window_end].set(0.0)
        mask = mask.at[:global_tokens, :].set(0.0)
        mask = mask.at[:, :global_tokens].set(0.0)
        return mask
    
    def setup(self):
        if self.config.use_alibi:
            num_alibi_heads = self.config.num_alibi_heads or self.config.n_heads
            self.alibi_slopes = create_alibi_slopes(num_alibi_heads)
    
    def compute_alibi_attention(self, qk: jnp.ndarray) -> jnp.ndarray:
        seq_len = qk.shape[-1]
        positions = jnp.arange(seq_len)
        distance = positions[:, None] - positions[None, :]
        distance = -jnp.abs(distance).astype(jnp.float32)
        distance = distance[None, :, :]
        alibi_bias = self.alibi_slopes[:, None, None] * distance
        return qk + alibi_bias

    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        batch_size, seq_len, dim = x.shape
        num_heads = self.config.n_heads
        num_kv_heads = self.config.n_kv_heads if hasattr(self.config, 'n_kv_heads') else num_heads
        head_dim = dim // num_heads  # e.g., 768 // 12 = 64
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads for GQA"
        
        token_complexity = jnp.sum(jnp.abs(x), axis=-1, keepdims=True)
        complexity_weights = nn.sigmoid(token_complexity)
        
        # Calculate dimensions for Q, K, V projections
        kv_dim = head_dim * num_kv_heads
        kv_proj_dim = 2 * kv_dim
        
        # Project inputs
        q = nn.remat(ParallelDense)(dim, use_bias=False, dtype=x.dtype)(x * complexity_weights)
        kv = nn.remat(ParallelDense)(kv_proj_dim, use_bias=False, dtype=x.dtype)(x)
        
        # Split KV into K and V with correct shapes
        k, v = jnp.split(kv, 2, axis=-1)
        
        # Reshape with correct head dimensions
        q = rearrange(q, 'b s (h d) -> b h s d', h=num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=num_kv_heads)
        
        if self.config.use_rope:
            compute_dtype = x.dtype
            sin, cos = precompute_freqs(head_dim, seq_len, dtype=compute_dtype)
            q = rotary_embedding(q, sin, cos, use_local_repeat=False)
            k = rotary_embedding(k, sin, cos, use_local_repeat=True)
        
        repeats = num_heads // num_kv_heads
        k = repeat(k, 'b h s d -> b (h r) s d', r=repeats)
        v = repeat(v, 'b h s d -> b (h r) s d', r=repeats)
        
        if mask is None:
            mask = self.create_sliding_window_mask(seq_len)
        else:
            window_mask = self.create_sliding_window_mask(seq_len)
            mask = jnp.minimum(mask, window_mask)
        
        scale = 1.0 / jnp.sqrt(head_dim)
        qk = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        if self.config.use_alibi:
            qk = self.compute_alibi_attention(qk)
        
        if self.config.use_flash_attention:
            qk = jnp.where(mask, qk, -1e9)
            attention = nn.softmax(qk, axis=-1)
        else:
            qk = jnp.where(mask, qk, -1e9)
            attention = nn.softmax(qk, axis=-1)
        
        if not deterministic:
            attention = nn.Dropout(rate=self.config.attention_dropout)(attention, deterministic=False)
        
        if not deterministic:
            v = jnp.asarray(v, self.config.kv_cache_dtype)
            
        output = jnp.einsum('bhij,bhjd->bhid', attention, v)
        output = rearrange(output, 'b h s d -> b s (h d)')
        
        output = nn.remat(ParallelDense)(dim, dtype=x.dtype)(output)
        output = nn.Dropout(rate=self.config.dropout_rate)(output, deterministic=deterministic)
        
        return output

class TransformerBlock(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        attn_norm = RMSNorm(dtype=x.dtype)(x)
        attn_output = MultiheadAttention(self.config)(attn_norm, mask, deterministic)
        x = x + attn_output
        
        ff_norm = RMSNorm(dtype=x.dtype)(x)
        if self.config.n_experts > 0:
            ff_output, load_balance_loss = MoELayer(self.config)(ff_norm, deterministic)
            self.sow('intermediates', 'load_balance_loss', load_balance_loss)
        else:
            ff_output = ParallelMLP(self.config)(ff_norm, deterministic)
        
        x = x + ff_output
        return x

class VishwamAIModel(nn.Module):
    config: ModelConfig

    def setup(self):
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.dtype(self.config.dtype)
        )
        self.encoder = [TransformerBlock(ModelArgs(
            dim=self.config.hidden_size,
            n_layers=1,
            n_heads=self.config.num_attention_heads,
            n_kv_heads=self.config.num_key_value_heads if self.config.use_gqa else self.config.num_attention_heads,
            vocab_size=self.config.vocab_size,
            use_rope=self.config.use_rope,
            dropout_rate=self.config.hidden_dropout_prob,
            attention_dropout=self.config.attention_dropout_prob,
            n_experts=4,
            expert_dim=4096,
            expert_capacity_factor=1.25,
            max_seq_len=self.config.max_position_embeddings,
            window_size=512,
            global_tokens=64,
            max_batch_size=32,
            use_gqa=self.config.use_gqa,
            use_flash_attention=self.config.use_flash_attention,
            use_alibi=self.config.use_alibi
        )) for _ in range(self.config.num_layers)]
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=jnp.dtype(self.config.dtype))
        self.lm_head = ParallelDense(features=self.config.vocab_size, use_bias=False, dtype=jnp.dtype(self.config.dtype))

    def _create_causal_mask(self, seq_len: int) -> jnp.ndarray:
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.where(mask, 0.0, -1e9)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None, 
                 deterministic: bool = True) -> Dict[str, jnp.ndarray]:
        hidden_states = self.embeddings(input_ids)
        
        if attention_mask is None:
            attention_mask = self._create_causal_mask(input_ids.shape[1])
        
        encoder_outputs = []
        for encoder_layer in self.encoder:
            hidden_states = encoder_layer(hidden_states, attention_mask, deterministic)
            encoder_outputs.append(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'encoder_outputs': encoder_outputs
        }

    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None):
        if not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors", "config.json"])
        if config is None:
            with open(os.path.join(model_path, "config.json"), 'r') as f:
                config_dict = json.load(f)
                config_dict = ModelConfig.map_config_params(config_dict)
            config = ModelConfig(**config_dict)
        
        model = cls(config)
        params = {}
        for shard_file in sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")]):
            shard_path = os.path.join(model_path, shard_file)
            params.update(stf.load_file(shard_path))
        return model, {'params': params}

    def load_weights(self, model_path: str, reduced_size: bool = False):
        if not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors"])
        if reduced_size:
            self.config.hidden_size //= 2
            self.config.num_attention_heads //= 2
            self.config.num_key_value_heads = max(1, self.config.num_key_value_heads // 2)
            self.config.intermediate_size //= 2
            self.config.num_layers //= 2
        
        params = {}
        dtype = getattr(jnp, self.config.dtype)
        for shard_file in sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")]):
            shard_path = os.path.join(model_path, shard_file)
            with stf.safe_open(shard_path, framework="numpy") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name).astype(np.float32)
                    if reduced_size and len(tensor.shape) >= 2:
                        new_shape = tuple(s // 2 if i < 2 else s for i, s in enumerate(tensor.shape))
                        tensor = tensor[tuple(slice(0, s) for s in new_shape)]
                    with jax.default_device(jax.devices("cpu")[0]):
                        params[name] = jnp.array(tensor, dtype=dtype)
                    del tensor
                    gc.collect()
        self.bind({'params': params})
        return self

# Training and Evaluation Utilities
@jax.jit
def train_step(state, input_ids, targets, rng):
    def loss_fn(params):
        outputs = state.apply_fn({'params': params}, input_ids, deterministic=False, rngs={'dropout': rng})
        logits = outputs['logits'][:, :-1]
        targets_shifted = targets[:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets_shifted).mean()
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jax.jit
def test_step(state, input_ids, targets):
    outputs = state.apply_fn({'params': state.params}, input_ids, deterministic=True)
    logits = outputs['logits'][:, :-1]
    targets_shifted = targets[:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets_shifted).mean()
    return loss

def split_dataset(text_file: str, tokenizer, batch_size: int, seq_len: int, train_ratio: float = 0.8):
    with open(text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_lines, test_lines = lines[:split_idx], lines[split_idx:]
    
    def process_lines(lines):
        tokens = []
        for line in lines:
            tokens.extend(tokenizer.encode(line.strip(), add_special_tokens=True))
        for i in range(0, len(tokens) - seq_len, seq_len):
            batch = tokens[i:i + seq_len + 1]
            if len(batch) == seq_len + 1:
                yield jnp.array(batch[:-1]), jnp.array(batch[1:])
    
    return list(process_lines(train_lines)), list(process_lines(test_lines))
