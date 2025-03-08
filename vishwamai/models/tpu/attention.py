# /home/kasinadhsarma/VishwamAI/vishwamai/models/tpu/attention.py
"""
Enhanced TPU-optimized attention mechanisms for VishwamAI using JAX, Haiku, and Optax:
- Dynamic sparse attention with learned sparsity (F dimension)
- Learned performer features with linear scaling
- Optimized Mixture of Experts (MoE) with hierarchical routing (E dimension)
- Cross-domain/multi-modal attention with channels (C dimension)
- Temporal convolution integration with temporal context (T dimension)
- Hierarchical MoE with multi-level routing
- FlashMLA for memory-efficient latent attention
- TPU-optimized attention with combined QKV
- Fractal attention with recursive levels (F dimension)
- Semantic group attention with role-aware processing (S dimension)
- Temporal cross attention with time and channels (T, C dimensions)
"""

import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from haiku import Module, transform, initializers
import haiku as hk
import optax
from typing import Optional, Tuple, Dict
import math

# Enable bfloat16 for TPU efficiency
from jax import config
config.update("jax_enable_x64", False)  # Use bfloat16 by default

class BaseAttention(Module):
    """Base class for TPU-optimized attention mechanisms."""
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1, name: str = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = hk.Linear(embed_dim, w_init=initializers.VarianceScaling(1/math.sqrt(2)), b_init=initializers.Zeros())
        self.k_proj = hk.Linear(embed_dim, w_init=initializers.VarianceScaling(1/math.sqrt(2)), b_init=initializers.Zeros())
        self.v_proj = hk.Linear(embed_dim, w_init=initializers.VarianceScaling(1/math.sqrt(2)), b_init=initializers.Zeros())
        self.o_proj = hk.Linear(embed_dim, w_init=initializers.VarianceScaling(1.0), b_init=initializers.Zeros())
        self.dropout = hk.Dropout(dropout_rate)

    def _reshape_for_multihead(self, x):
        batch_size, seq_len, _ = x.shape
        x = jnp.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return jnp.transpose(x, (0, 2, 1, 3))

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        raise NotImplementedError("Subclasses must implement __call__")

class DynamicSparseAttention(BaseAttention):
    """TPU-optimized attention with learned sparsity using Gumbel-Softmax."""
    def __init__(self, embed_dim: int, num_heads: int, k: int = 10, dropout_rate: float = 0.1, temperature: float = 0.5, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.k = k
        self.temperature = temperature
        self.sparsity_controller = hk.Linear(1, w_init=initializers.VarianceScaling(1.0), b_init=initializers.Zeros())

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        batch_size, seq_len, _ = x.shape
        rng = rng or random.PRNGKey(0)

        q = self._reshape_for_multihead(self.q_proj(x))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))

        attn_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        token_importance = self.sparsity_controller(x).squeeze(-1)

        if is_training:
            noise = -jnp.log(-jnp.log(random.uniform(rng, token_importance.shape) + 1e-10) + 1e-10)
            gumbel_logits = (token_importance.unsqueeze(1).unsqueeze(1) + noise) / self.temperature
            sparse_mask = jax.nn.gumbel_softmax(gumbel_logits, tau=self.temperature, hard=True)
            sparse_mask = jnp.repeat(sparse_mask, self.num_heads * seq_len, axis=1).reshape(batch_size, self.num_heads, seq_len, seq_len)
        else:
            top_indices = jnp.argsort(token_importance, axis=-1)[:, -min(self.k, seq_len):]
            sparse_mask = jnp.zeros((batch_size, 1, 1, seq_len))
            batch_indices = jnp.arange(batch_size)[:, jnp.newaxis]
            sparse_mask = sparse_mask.at[batch_indices, 0, 0, top_indices].set(1.0)
            sparse_mask = jnp.repeat(sparse_mask, self.num_heads * seq_len, axis=1).reshape(batch_size, self.num_heads, seq_len, seq_len)

        if mask is not None:
            attn_scores = jnp.where(mask == 0, -jnp.inf, attn_scores)
        attn_scores = attn_scores * sparse_mask

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs, rng, is_training)

        output = jnp.matmul(attn_probs, v)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
        return self.o_proj(output)

class LearnedPerformerAttention(BaseAttention):
    """TPU-optimized linear attention with learned feature maps."""
    def __init__(self, embed_dim: int, num_heads: int, kernel_dim: int = 256, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.kernel_dim = kernel_dim
        self.phi_proj = hk.Linear(kernel_dim, w_init=initializers.VarianceScaling(1.0), b_init=initializers.Zeros())
        self.psi_proj = hk.Linear(kernel_dim, w_init=initializers.VarianceScaling(1.0), b_init=initializers.Zeros())

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        batch_size, seq_len, _ = x.shape
        rng = rng or random.PRNGKey(0)

        q = self._reshape_for_multihead(self.q_proj(x))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))

        q_phi = jax.nn.elu(self.phi_proj(q)) + 1
        k_psi = jax.nn.elu(self.psi_proj(k)) + 1

        if mask is not None:
            k_psi = k_psi * mask.unsqueeze(-1)
            v = v * mask.unsqueeze(-1)

        kv = jnp.einsum('bhld,bhlm->bhdm', k_psi, v)
        qkv = jnp.einsum('bhld,bhdm->bhlm', q_phi, kv)
        normalizer = jnp.einsum('bhld,bhd->bhl', q_phi, jnp.sum(k_psi, axis=2)).unsqueeze(-1) + 1e-8

        output = qkv / normalizer
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
        return self.o_proj(output)

class OptimizedMoEAttention(BaseAttention):
    """TPU-optimized Mixture of Experts Attention with load balancing."""
    def __init__(self, embed_dim: int, num_heads: int, num_experts: int = 4, top_k: int = 2, dropout_rate: float = 0.1, gate_jitter: float = 0.01, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate_jitter = gate_jitter
        self.experts = [BaseExpertAttention(embed_dim, num_heads, dropout_rate) for _ in range(num_experts)]
        self.router = hk.Sequential([
            hk.Linear(embed_dim // 2), hk.LayerNorm(embed_dim // 2), jax.nn.relu, hk.Linear(num_experts)
        ])
        self.expert_priors = self.param('priors', lambda shape: jnp.ones(shape) / num_experts, (num_experts,))
        self.load_balancing_coeff = 0.01

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        batch_size, seq_len, _ = x.shape
        rng = rng or random.PRNGKey(0)

        router_logits = self.router(x)
        if is_training and self.gate_jitter > 0:
            router_logits += random.normal(rng, router_logits.shape) * self.gate_jitter

        routing_probs = jax.nn.softmax(router_logits, axis=-1)
        top_k_probs, top_k_indices = jax.lax.top_k(routing_probs, self.top_k)

        expert_mask = jnp.zeros((batch_size, seq_len, self.num_experts))
        for k in range(self.top_k):
            indices = top_k_indices[:, :, k]
            probs = top_k_probs[:, :, k]
            batch_idx = jnp.arange(batch_size)[:, jnp.newaxis]
            seq_idx = jnp.arange(seq_len)[jnp.newaxis, :]
            expert_mask = expert_mask.at[batch_idx, seq_idx, indices].add(probs)

        if is_training:
            expert_usage = jnp.mean(expert_mask, axis=(0, 1))
            load_balance_loss = jnp.mean((expert_usage - self.expert_priors) ** 2) * self.load_balancing_coeff
            self.aux_loss = load_balance_loss
        else:
            self.aux_loss = 0.0

        expert_outputs = jnp.zeros((batch_size, seq_len, self.embed_dim))
        for i, expert in enumerate(self.experts):
            mask_i = expert_mask[:, :, i].unsqueeze(-1)
            if jnp.sum(mask_i) > 0:
                expert_output = expert(x, context, mask, rng, is_training)
                expert_outputs += expert_output * mask_i

        return expert_outputs

class BaseExpertAttention(BaseAttention):
    """Expert attention module for MoE."""
    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        q = self._reshape_for_multihead(self.q_proj(x))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))

        attn_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = jnp.where(mask == 0, -jnp.inf, attn_scores)

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs, rng, is_training)

        output = jnp.matmul(attn_probs, v)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(x.shape)
        return self.o_proj(output)

class CrossDomainAttention(BaseAttention):
    """TPU-optimized cross-modal attention with channels (C dimension)."""
    def __init__(self, embed_dim: int, num_heads: int, num_domains: int = 2, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.num_domains = num_domains
        self.domain_q_projs = [hk.Linear(embed_dim) for _ in range(num_domains)]
        self.domain_adapter = hk.Sequential([hk.Linear(embed_dim), hk.LayerNorm(embed_dim), jax.nn.gelu, hk.Linear(embed_dim)])
        self.domain_gate = hk.Linear(num_domains)

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        q = self._reshape_for_multihead(self.domain_q_projs[domain_id](x))
        k = self._reshape_for_multihead(self.k_proj(context if context is not None else x))
        v = self._reshape_for_multihead(self.v_proj(context if context is not None else x))

        if context is not None:
            k_adapted = self.domain_adapter(k)
            v_adapted = self.domain_adapter(v)
            domain_gates = jax.nn.softmax(self.domain_gate(jnp.mean(x, axis=1)), axis=-1)
            k = k * (1 - domain_gates[:, 1][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]) + k_adapted * domain_gates[:, 1][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            v = v * (1 - domain_gates[:, 1][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]) + v_adapted * domain_gates[:, 1][:, jnp.newaxis, jnp.newaxis, jnp.newaxis]

        attn_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = jnp.where(mask == 0, -jnp.inf, attn_scores)

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs, rng, is_training)

        output = jnp.matmul(attn_probs, v)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(x.shape)
        return self.o_proj(output)

class TemporalConvAttention(BaseAttention):
    """TPU-optimized temporal convolution-enhanced attention with temporal context (T dimension)."""
    def __init__(self, embed_dim: int, num_heads: int, kernel_size: int = 3, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        padding = (kernel_size - 1) // 2
        self.temporal_conv = hk.Sequential([
            hk.Conv1D(embed_dim, kernel_size, padding=padding, feature_group_count=embed_dim),
            hk.Conv1D(embed_dim, 1),
            jax.nn.gelu
        ])
        self.fusion_layer = hk.Linear(embed_dim * 2)

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        x_conv = jnp.transpose(self.temporal_conv(jnp.transpose(x, (0, 2, 1))), (0, 2, 1))
        q = self._reshape_for_multihead(self.q_proj(x))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))

        attn_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = jnp.where(mask == 0, -jnp.inf, attn_scores)

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs, rng, is_training)

        attn_out = jnp.matmul(attn_probs, v)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(x.shape)
        attn_out = self.o_proj(attn_out)
        return self.fusion_layer(jnp.concatenate([attn_out, x_conv], axis=-1))

class HierarchicalMoEAttention(BaseAttention):
    """TPU-optimized hierarchical Mixture of Experts Attention with multi-level routing."""
    def __init__(self, embed_dim: int, num_heads: int, num_experts: int = 4, num_sub_experts: int = 2, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.num_experts = num_experts
        self.num_sub_experts = num_sub_experts
        attention_types = [DynamicSparseAttention, LearnedPerformerAttention, BaseExpertAttention, TemporalConvAttention]
        self.experts = [attention_types[i % len(attention_types)](embed_dim, num_heads, dropout_rate=dropout_rate) for i in range(num_experts)]
        self.sub_experts = [[hk.Sequential([hk.Linear(embed_dim), hk.LayerNorm(embed_dim), jax.nn.gelu, hk.Linear(embed_dim)]) for _ in range(num_sub_experts)] for _ in range(num_experts)]
        self.router_l1 = hk.Sequential([hk.Linear(embed_dim // 2), hk.LayerNorm(embed_dim // 2), jax.nn.relu, hk.Linear(num_experts)])
        self.router_l2 = [hk.Sequential([hk.Linear(embed_dim // 4), hk.LayerNorm(embed_dim // 4), jax.nn.relu, hk.Linear(num_sub_experts)]) for _ in range(num_experts)]

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        router_logits_l1 = self.router_l1(x)
        routing_probs_l1 = jax.nn.softmax(router_logits_l1, axis=-1)

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x, context, mask, rng, is_training)
            router_logits_l2 = self.router_l2[i](expert_output)
            routing_probs_l2 = jax.nn.softmax(router_logits_l2, axis=-1)

            sub_expert_output = jnp.zeros_like(expert_output)
            for j, sub_expert in enumerate(self.sub_experts[i]):
                sub_output = sub_expert(expert_output)
                sub_expert_weight = routing_probs_l2[:, :, j].unsqueeze(-1)
                sub_expert_output += sub_output * sub_expert_weight

            expert_outputs.append(sub_expert_output)

        final_output = jnp.zeros_like(x)
        for i, expert_output in enumerate(expert_outputs):
            expert_weight = routing_probs_l1[:, :, i].unsqueeze(-1)
            final_output += expert_output * expert_weight

        return final_output

class FlashMLAttention(BaseAttention):
    """TPU-optimized Multi-head Latent Attention with Flash Attention."""
    def __init__(self, embed_dim: int, num_heads: int, latent_dim: int = 64, dropout_rate: float = 0.1, block_size: int = 128, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.q_latent_proj = hk.Linear(latent_dim, w_init=initializers.VarianceScaling(1.0), b_init=initializers.Zeros())
        self.k_latent_proj = hk.Linear(latent_dim, w_init=initializers.VarianceScaling(1.0), b_init=initializers.Zeros())
        self.latent_mixer = hk.Sequential([hk.Linear(self.head_dim), jax.nn.gelu])
        self.block_size_adjuster = hk.get_parameter("block_size", (), init=initializers.Constant(math.log(block_size)))

    @jit
    def _compute_attention_blockwise(self, q, k, v, mask=None):
        batch_size, num_heads, seq_len, _ = q.shape
        output = jnp.zeros_like(v)

        q_latent = jax.nn.elu(self.q_latent_proj(q)) + 1
        k_latent = jax.nn.elu(self.k_latent_proj(k)) + 1

        effective_block_size = min(int(jnp.exp(self.block_size_adjuster)), seq_len)
        for i in range(0, seq_len, effective_block_size):
            end_idx = min(i + effective_block_size, seq_len)
            q_block = q_latent[:, :, i:end_idx]

            for j in range(0, seq_len, effective_block_size * 4):
                j_end = min(j + effective_block_size * 4, seq_len)
                k_block = k_latent[:, :, j:j_end]
                v_block = v[:, :, j:j_end]

                scores = jnp.matmul(q_block, jnp.transpose(k_block, (0, 1, 3, 2))) / jnp.sqrt(self.latent_dim)
                if mask is not None:
                    block_mask = mask[:, :, i:end_idx, j:j_end]
                    scores = jnp.where(block_mask == 0, -jnp.inf, scores)

                attn_probs = jax.nn.softmax(scores, axis=-1)
                output = output.at[:, :, i:end_idx].add(jnp.matmul(attn_probs, v_block))

        return self.latent_mixer(output)

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        q = self._reshape_for_multihead(self.q_proj(x))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))
        output = self._compute_attention_blockwise(q, k, v, mask)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(x.shape)
        return self.o_proj(output)

class TPUOptimizedAttention(BaseAttention):
    """TPU-optimized attention with combined QKV and gradient checkpointing."""
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.qkv_combined = hk.Linear(3 * embed_dim, w_init=initializers.VarianceScaling(1.0), b_init=initializers.Zeros())
        self.use_gradient_checkpointing = True

    @jit
    def _reshape_qkv(self, qkv):
        batch_size, seq_len, _ = qkv.shape
        qkv = jnp.reshape(qkv, (batch_size, seq_len, 3, self.num_heads, self.head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        return qkv[0], qkv[1], qkv[2]

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        input_states = x if context is None else context
        qkv = jax.checkpoint(self.qkv_combined)(input_states) if self.use_gradient_checkpointing and is_training else self.qkv_combined(input_states)
        q, k, v = self._reshape_qkv(qkv)

        attn_scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = jnp.where(mask.unsqueeze(1) == 0, -jnp.inf, attn_scores)

        attn_probs = jax.nn.softmax(attn_scores, axis=-1)
        attn_probs = self.dropout(attn_probs, rng, is_training)

        output = jnp.matmul(attn_probs, v)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(x.shape)
        return self.o_proj(output)

class FractalAttention(BaseAttention):
    """Fractal attention with recursive levels (F dimension) for hierarchical reasoning."""
    def __init__(self, embed_dim: int, num_heads: int, fractal_levels: int = 3, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.fractal_levels = fractal_levels
        self.level_attentions = [BaseExpertAttention(embed_dim, num_heads, dropout_rate) for _ in range(fractal_levels)]
        self.agg_layer = hk.Linear(embed_dim * fractal_levels)
        self.level_scales = hk.get_parameter("level_scales", (fractal_levels,), init=initializers.Ones())

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        batch_size, seq_len, _ = x.shape
        level_outputs = []

        current_input = x
        current_mask = mask
        for level in range(self.fractal_levels):
            level_output = self.level_attentions[level](current_input, context, current_mask, rng, is_training)
            level_output = level_output * self.level_scales[level]
            level_outputs.append(level_output)
            if level < self.fractal_levels - 1:
                current_input = level_output[:, ::2, :]
                if current_mask is not None:
                    current_mask = current_mask[:, :, ::2, ::2]

        combined = jnp.concatenate(level_outputs, axis=-1)
        output = self.agg_layer(combined)
        return output

class SemanticGroupAttention(BaseAttention):
    """Attention with semantic grouping (S dimension) for role-aware processing."""
    def __init__(self, embed_dim: int, num_heads: int, num_groups: int = 4, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.num_groups = num_groups
        self.group_assigner = hk.Linear(num_groups)
        self.group_attentions = [BaseExpertAttention(embed_dim, num_heads, dropout_rate) for _ in range(num_groups)]
        self.cross_group_attention = BaseExpertAttention(embed_dim, num_heads, dropout_rate)

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        group_logits = self.group_assigner(x)
        group_probs = jax.nn.softmax(group_logits, axis=-1)

        group_outputs = jnp.zeros_like(x)
        for s in range(self.num_groups):
            group_mask = group_probs[:, :, s].unsqueeze(-1)
            group_output = self.group_attentions[s](x, context, mask, rng, is_training)
            group_outputs += group_output * group_mask

        final_output = self.cross_group_attention(group_outputs, context, mask, rng, is_training)
        return final_output

class TemporalCrossAttention(BaseAttention):
    """Temporal and cross-modal attention with T (time) and C (channels) dimensions."""
    def __init__(self, embed_dim: int, num_heads: int, num_timesteps: int = 5, num_channels: int = 2, dropout_rate: float = 0.1, name: str = None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.num_timesteps = num_timesteps
        self.num_channels = num_channels
        self.time_projs = [hk.Linear(embed_dim) for _ in range(num_timesteps)]
        self.channel_projs = [hk.Linear(embed_dim) for _ in range(num_channels)]
        self.fusion_layer = hk.Linear(embed_dim * num_timesteps * num_channels)

    @jit
    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, rng=None, is_training=True):
        batch_size, seq_len, _ = x.shape
        if temporal_states is None:
            temporal_states = jnp.zeros((batch_size, self.num_timesteps, seq_len, self.embed_dim))

        time_outputs = []
        for t in range(self.num_timesteps):
            time_input = temporal_states[:, t, :, :]
            time_output = self.time_projs[t](time_input)
            time_outputs.append(time_output)

        channel_outputs = []
        for c in range(self.num_channels):
            channel_input = x if c == 0 else (context if c == 1 and context is not None else x)
            channel_output = self._reshape_for_multihead(self.channel_projs[c](channel_input))
            k = self._reshape_for_multihead(self.k_proj(channel_input))
            v = self._reshape_for_multihead(self.v_proj(channel_input))
            attn_scores = jnp.matmul(channel_output, jnp.transpose(k, (0, 1, 3, 2))) / jnp.sqrt(self.head_dim)
            if mask is not None:
                attn_scores = jnp.where(mask == 0, -jnp.inf, attn_scores)
            attn_probs = jax.nn.softmax(attn_scores, axis=-1)
            attn_probs = self.dropout(attn_probs, rng, is_training)
            channel_output = jnp.matmul(attn_probs, v)
            channel_output = jnp.transpose(channel_output, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
            channel_outputs.append(channel_output)

        all_outputs = []
        for t in range(self.num_timesteps):
            for c in range(self.num_channels):
                all_outputs.append(time_outputs[t] + channel_outputs[c])
        combined = jnp.concatenate(all_outputs, axis=-1)
        output = self.fusion_layer(combined)
        return output

# Example usage with transformation
def forward_attention(x, context=None, mask=None, temporal_states=None, domain_id=0, is_training=True):
    model = HierarchicalMoEAttention(512, 8)
    params = model.init(random.PRNGKey(42), x, context, mask, temporal_states, domain_id, is_training=is_training)
    return hk.transform(lambda x, c, m, t, d, r, i: model(x, c, m, t, d, r, i))(params, x, context, mask, temporal_states, domain_id, None, is_training)

if __name__ == "__main__":
    # Test with dummy data
    key = random.PRNGKey(0)
    x = random.normal(key, (2, 10, 512))
    params = forward_attention.init(key, x)
    output = forward_attention.apply(params, x)
    print("Output shape:", output.shape)