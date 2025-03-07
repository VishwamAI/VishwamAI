import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

@dataclass(frozen=True)
class ModelConfig:
    """Enhanced model configuration with MoE, MLA, and TPU optimizations."""
    vocab_size: int = 32000
    hidden_size: int = 2048
    num_layers: int = 27
    num_attention_heads: int = 16
    num_key_value_heads: int = 8  # For GQA
    head_dim: int = 128  # hidden_size // num_attention_heads
    q_lora_rank: int = 0  # LoRA rank for queries (0 disables LoRA)
    kv_lora_rank: int = 512  # LoRA rank for keys/values
    intermediate_size: int = 10944  # For dense MLP
    moe_intermediate_size: int = 1408  # For MoE experts
    num_routed_experts: int = 64  # Total MoE experts
    num_activated_experts: int = 6  # Top-k experts to activate
    num_shared_experts: int = 2  # Shared experts in MoE
    n_dense_layers: int = 1  # Number of layers with dense MLP before switching to MoE
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 4096 * 4
    rope_theta: float = 10000.0
    rope_factor: float = 40.0
    dtype: str = "bfloat16"
    use_cache: bool = True
    use_gqa: bool = True
    use_rope: bool = True

    def __post_init__(self):
        if self.use_gqa:
            assert self.num_attention_heads % self.num_key_value_heads == 0
        assert self.hidden_size == self.num_attention_heads * self.head_dim

class LayerNorm(nn.Module):
    """TPU-optimized Layer Normalization."""
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        inv_std = jax.lax.rsqrt(var + self.epsilon)
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],), self.dtype)
        x = (x - mean) * inv_std * scale
        return x.astype(self.dtype)

class Dense(nn.Module):
    """TPU-optimized Dense layer."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16
    kernel_init: callable = nn.initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features), self.dtype)
        y = jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())))
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,), self.dtype)
            y = y + bias
        return y

class MLA(nn.Module):
    """Multi-Head Linear Attention inspired by PyTorch MLA."""
    config: ModelConfig

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.n_heads = self.config.num_attention_heads
        self.n_kv_heads = self.config.num_key_value_heads
        self.head_dim = self.config.head_dim
        self.q_lora_rank = self.config.q_lora_rank
        self.kv_lora_rank = self.config.kv_lora_rank

        # Query projection with optional LoRA
        if self.q_lora_rank == 0:
            self.wq = Dense(self.n_heads * self.head_dim, use_bias=False, dtype=dtype)
        else:
            self.wq_a = Dense(self.q_lora_rank, use_bias=False, dtype=dtype)
            self.q_norm = LayerNorm(dtype=dtype)
            self.wq_b = Dense(self.n_heads * self.head_dim, use_bias=False, dtype=dtype)

        # Key/Value projection with LoRA
        self.wkv_a = Dense(self.kv_lora_rank + self.head_dim, use_bias=False, dtype=dtype)  # Head dim for RoPE
        self.kv_norm = LayerNorm(dtype=dtype)
        self.wkv_b = Dense(self.n_kv_heads * (self.head_dim * 2), use_bias=False, dtype=dtype)  # k and v
        self.wo = Dense(self.config.hidden_size, use_bias=False, dtype=dtype)
        self.dropout = nn.Dropout(self.config.attention_dropout_prob)
        self.softmax_scale = self.head_dim ** -0.5

    def precompute_freqs_cis(self, seq_len: int) -> jnp.ndarray:
        if not self.config.use_rope:
            return None
        dim = self.config.head_dim
        freqs = 1.0 / (self.config.rope_theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
        freqs = freqs * self.config.rope_factor
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, freqs)
        return jnp.cos(freqs) + 1j * jnp.sin(freqs)

    def apply_rotary_emb(self, x: jnp.ndarray, freqs_cis: jnp.ndarray) -> jnp.ndarray:
        if not self.config.use_rope:
            return x
        x_ = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = x_[..., 0] + 1j * x_[..., 1]
        x_rotated = x_complex * freqs_cis
        return jnp.stack([x_rotated.real, x_rotated.imag], axis=-1).reshape(*x.shape)

    def __call__(self, x: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None,
                 cache: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
                 position_ids: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, Optional[Tuple]]:
        bsz, seq_len, _ = x.shape
        dtype = getattr(jnp, self.config.dtype)

        # Query projection
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.reshape(bsz, seq_len, self.n_heads, self.head_dim)

        # Key/Value projection
        kv = self.wkv_a(x)
        kv_normed = self.kv_norm(kv[..., :self.kv_lora_rank])
        k_pe = kv[..., self.kv_lora_rank:]  # Rotary part
        kv = self.wkv_b(kv_normed)
        k_nope, v = jnp.split(kv.reshape(bsz, seq_len, self.n_kv_heads, -1), [self.head_dim], axis=-1)

        # Apply rotary embeddings
        if self.config.use_rope:
            if position_ids is None:
                position_ids = jnp.arange(seq_len)
            freqs_cis = self.precompute_freqs_cis(self.config.max_position_embeddings)[position_ids]
            q = self.apply_rotary_emb(q, freqs_cis)
            k_pe = self.apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis).squeeze(2)

        k = jnp.concatenate([k_nope, k_pe.expand(bsz, seq_len, self.n_kv_heads, self.head_dim)], axis=-1)

        # Cache handling
        if self.config.use_cache and cache is not None:
            past_k, past_v = cache
            k = jnp.concatenate([past_k, k], axis=1)
            v = jnp.concatenate([past_v, v], axis=1)
        new_cache = (k, v) if self.config.use_cache else None

        # GQA: repeat k and v
        if self.config.use_gqa:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = jnp.repeat(k, repeat_factor, axis=2)
            v = jnp.repeat(v, repeat_factor, axis=2)

        # Attention computation
        scores = jnp.einsum("bqhd,bkhd->bhqk", q, k) * self.softmax_scale
        if attention_mask is not None:
            scores += attention_mask
        scores = nn.softmax(scores, axis=-1).astype(dtype)
        scores = self.dropout(scores, deterministic=True)
        attn_output = jnp.einsum("bhqk,bkhd->bqhd", scores, v)

        # Output projection
        output = self.wo(attn_output.reshape(bsz, seq_len, -1))
        return output, new_cache

class MLP(nn.Module):
    """Gated MLP with SiLU activation."""
    config: ModelConfig
    inter_dim: int

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.w1 = Dense(self.inter_dim, dtype=dtype)
        self.w2 = Dense(self.config.hidden_size, dtype=dtype)
        self.w3 = Dense(self.inter_dim, dtype=dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class Expert(nn.Module):
    """Single expert network for MoE."""
    config: ModelConfig

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.w1 = Dense(self.config.moe_intermediate_size, dtype=dtype)
        self.w2 = Dense(self.config.hidden_size, dtype=dtype)
        self.w3 = Dense(self.config.moe_intermediate_size, dtype=dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    """Gating mechanism for MoE."""
    config: ModelConfig

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.weight = self.param('weight', nn.initializers.normal(stddev=0.02),
                                (self.config.num_routed_experts, self.config.hidden_size), dtype)
        self.bias = self.param('bias', nn.initializers.zeros, (self.config.num_routed_experts,), dtype)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        scores = jax.lax.dot_general(x, self.weight, (((x.ndim - 1,), (1,)), ((), ()))) + self.bias
        scores = nn.softmax(scores, axis=-1)
        weights, indices = jax.lax.top_k(scores, self.config.num_activated_experts)
        return weights.astype(x.dtype), indices

class MoE(nn.Module):
    """Mixture of Experts layer."""
    config: ModelConfig

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.gate = Gate(self.config)
        self.experts = [Expert(self.config) for _ in range(self.config.num_routed_experts)]
        self.shared_experts = MLP(self.config, self.config.num_shared_experts * self.config.moe_intermediate_size)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        bsz, seq_len, _ = x.shape
        x_flat = x.reshape(-1, self.config.hidden_size)
        
        # Gating
        weights, indices = self.gate(x_flat)
        
        # Expert computation
        y = jnp.zeros_like(x_flat)
        for expert_idx in range(self.config.num_routed_experts):
            mask = (indices == expert_idx).any(axis=-1)
            if mask.sum() > 0:
                expert_input = x_flat[mask]
                expert_output = self.experts[expert_idx](expert_input)
                expert_weights = weights[mask] * (indices[mask] == expert_idx).astype(x.dtype)
                y = y.at[mask].add(expert_output * expert_weights.sum(axis=-1, keepdims=True))
        
        # Shared experts
        z = self.shared_experts(x_flat)
        return (y + z).reshape(bsz, seq_len, self.config.hidden_size)

class EnhancedTransformerBlock(nn.Module):
    """Transformer block with MLA, MLP, or MoE."""
    config: ModelConfig
    layer_id: int

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.attn = MLA(self.config)
        self.ffn = (MLP(self.config, self.config.intermediate_size) if self.layer_id < self.config.n_dense_layers
                   else MoE(self.config))
        self.norm1 = LayerNorm(dtype=dtype)
        self.norm2 = LayerNorm(dtype=dtype)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def __call__(self, x: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None,
                 cache: Optional[Tuple] = None) -> Tuple[jnp.ndarray, Optional[Tuple]]:
        attn_output, new_cache = self.attn(self.norm1(x), attention_mask, cache)
        x = x + self.dropout(attn_output, deterministic=True)
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output, deterministic=True)
        return x, new_cache

class EnhancedVishwamAIModel(nn.Module):
    """Enhanced VishwamAI model with MoE, MLA, and MLP."""
    config: ModelConfig

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=dtype
        )
        self.layers = [EnhancedTransformerBlock(self.config, layer_id=i) for i in range(self.config.num_layers)]
        self.norm = LayerNorm(dtype=dtype)
        self.lm_head = Dense(self.config.vocab_size, use_bias=False, dtype=dtype)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None,
                 cache: Optional[List[Tuple]] = None, position_ids: Optional[jnp.ndarray] = None) -> Dict:
        x = self.embeddings(input_ids)
        bsz, seq_len = input_ids.shape

        if attention_mask is None and seq_len > 1:
            attention_mask = jnp.triu(jnp.full((seq_len, seq_len), -jnp.inf), k=1)
            attention_mask = attention_mask[None, None, :, :]

        if cache is None and self.config.use_cache:
            cache = [None] * self.config.num_layers

        new_cache = [] if self.config.use_cache else None
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache and i < len(cache) else None
            x, layer_new_cache = layer(x, attention_mask, layer_cache)
            if new_cache is not None:
                new_cache.append(layer_new_cache)

        x = self.norm(x)
        logits = self.lm_head(x)
        return {"logits": logits, "hidden_states": x, "cache": new_cache}

    @staticmethod
    def generate(model, params, input_ids: jnp.ndarray, max_length: int, temperature: float = 1.0,
                 top_k: int = 40) -> jnp.ndarray:
        bsz = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        cache = None

        @jax.jit
        def step(inputs, cache):
            outputs = model.apply({'params': params}, inputs, cache=cache)
            logits = outputs['logits'][:, -1, :] / temperature
            top_k_logits, top_k_indices = jax.lax.top_k(logits, top_k)
            probs = nn.softmax(top_k_logits, axis=-1)
            next_token = jax.random.choice(jax.random.PRNGKey(0), top_k_indices, shape=(bsz,), p=probs)
            return jnp.concatenate([inputs, next_token[:, None]], axis=1), outputs['cache']

        generated = input_ids
        for _ in range(max_length - seq_len):
            generated, cache = step(generated, cache)
        return generated

if __name__ == "__main__":
    config = ModelConfig()
    model = EnhancedVishwamAIModel(config)
    rng = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(rng, (2, 10), 0, config.vocab_size)
    
    params = model.init(rng, input_ids)
    outputs = model.apply({'params': params['params']}, input_ids)
    print("Logits shape:", outputs['logits'].shape)

    generated = EnhancedVishwamAIModel.generate(model, params['params'], input_ids, max_length=20)
    print("Generated shape:", generated.shape)