# /home/kasinadhsarma/VishwamAI/vishwamai/models/hybrid_attention.py
"""
Hybrid TPU/GPU-optimized attention mechanisms for VishwamAI:
- GPU: PyTorch with Triton (FlashMLAAttention, MultiModalAttention, TemporalAttention)
- TPU: JAX with Haiku/Sonnet (FlashMLAttentionTPU, MultiModalAttentionTPU, TemporalAttentionTPU, SonnetFlashAttentionTPU)
- Features: Dynamic sparse attention, cross-domain attention, temporal convolution, and more.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import triton
import triton.language as tl
import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import haiku as hk
import optax
import tensorflow as tf
import sonnet as snt
from tensorflow.experimental import dlpack
import math
from abc import ABC, abstractmethod

from .core import apply_rotary_embedding, create_causal_mask

# Enable bfloat16 for TPU efficiency
from jax import config
config.update("jax_enable_x64", False)

# Triton kernel for Xavier initialization (GPU)
@triton.jit
def xavier_init_kernel(output_ptr, n_elements, fan_in, fan_out, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gain = 1.0 / tl.sqrt(fan_in + fan_out)
    random_vals = tl.rand(offsets, seed=42) * 2.0 - 1.0
    values = random_vals * gain
    tl.store(output_ptr + offsets, values, mask=mask)

# Base Attention Classes (Abstract for GPU and TPU)
class BaseAttention(ABC):
    @abstractmethod
    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0, is_training=True):
        pass

# GPU Base Attention (PyTorch)
class BaseAttentionGPU(nn.Module, BaseAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.use_amp = use_amp
        self._reset_parameters()

    def _reset_parameters(self):
        def triton_xavier_init(tensor, fan_in, fan_out):
            n_elements = tensor.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            xavier_init_kernel[grid](tensor.data_ptr(), n_elements, fan_in, fan_out, BLOCK_SIZE=1024)
        triton_xavier_init(self.q_proj.weight, self.embed_dim, self.embed_dim)
        triton_xavier_init(self.k_proj.weight, self.embed_dim, self.embed_dim)
        triton_xavier_init(self.v_proj.weight, self.embed_dim, self.embed_dim)
        triton_xavier_init(self.o_proj.weight, self.embed_dim, self.embed_dim)
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.o_proj.bias, 0.0)

    def _reshape_for_multihead(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

# TPU Base Attention (Haiku)
class BaseAttentionTPU(hk.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1, name: str = None):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = hk.Linear(embed_dim, w_init=hk.initializers.VarianceScaling(1/math.sqrt(2)))
        self.k_proj = hk.Linear(embed_dim, w_init=hk.initializers.VarianceScaling(1/math.sqrt(2)))
        self.v_proj = hk.Linear(embed_dim, w_init=hk.initializers.VarianceScaling(1/math.sqrt(2)))
        self.o_proj = hk.Linear(embed_dim, w_init=hk.initializers.VarianceScaling(1.0))
        self.dropout_rate = dropout_rate

    def _reshape_for_multihead(self, x):
        batch_size, seq_len, _ = x.shape
        x = jnp.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        return jnp.transpose(x, (0, 2, 1, 3))

# GPU FlashMLAAttention
class FlashMLAAttentionGPU(BaseAttentionGPU):
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_amp=True, causal=False):
        super().__init__(embed_dim, num_heads, dropout, use_amp)
        self.causal = causal
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0, is_training=True):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            q = self._reshape_for_multihead(self.q_proj(x))
            k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
            v = self._reshape_for_multihead(self.v_proj(x if context is None else context))
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs) if is_training else attn_probs
            output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous()
            output = output.view(x.size(0), x.size(1), self.embed_dim)
            return self.o_proj(output)

# TPU FlashMLAttention
class FlashMLAttentionTPU(BaseAttentionTPU):
    def __init__(self, embed_dim, num_heads, block_size=128, dropout_rate=0.1, causal=False, sm_scale=0.5, name=None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.block_size = block_size
        self.causal = causal
        self.sm_scale = sm_scale

    @staticmethod
    def _attention_head(q_block, k_block, v_block, mask_block, causal, sm_scale):
        scores = jnp.matmul(q_block, k_block.transpose(-2, -1)) * sm_scale
        if causal:
            causal_mask = jnp.tril(jnp.ones((q_block.shape[-2], k_block.shape[-2]), dtype=jnp.bfloat16))
            scores = jnp.where(causal_mask[None, :, :], scores, -jnp.inf)
        if mask_block is not None:
            scores = jnp.where(mask_block == 0, -jnp.inf, scores)
        attn_probs = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(attn_probs, v_block)

    def _compute_attention_blockwise(self, q, k, v, mask=None):
        batch_size, num_heads, seq_len, head_dim = q.shape
        output = jnp.zeros_like(v, dtype=jnp.bfloat16)
        
        # Use centralized causal mask if needed
        causal_mask = create_causal_mask(seq_len) if self.causal else None

        @jit
        def process_block(i, carry):
            output, q, k, v, mask = carry
            i_end = jnp.minimum(i + self.block_size, seq_len)
            q_block = lax.dynamic_slice(q, (0, 0, i, 0), 
                                      (q.shape[0], q.shape[1], i_end - i, q.shape[3]))
            k_block = k if not self.causal else lax.dynamic_slice(
                k, (0, 0, 0, 0), 
                (k.shape[0], k.shape[1], i_end, k.shape[3])
            )
            v_block = v if not self.causal else lax.dynamic_slice(
                v, (0, 0, 0, 0), 
                (v.shape[0], v.shape[1], i_end, v.shape[3])
            )
            
            # Handle masks
            block_mask = None
            if mask is not None:
                block_mask = lax.dynamic_slice(
                    mask, (0, 0, i, 0), 
                    (mask.shape[0], mask.shape[1], i_end - i, i_end if self.causal else mask.shape[3])
                )
            if causal_mask is not None:
                block_causal = lax.dynamic_slice(
                    causal_mask, (i, 0), 
                    (i_end - i, i_end if self.causal else causal_mask.shape[1])
                )
                block_mask = block_causal if block_mask is None else (block_mask & block_causal)

            output_block = vmap(
                vmap(self._attention_head, in_axes=(0, 0, 0, 0, None, None)),
                in_axes=(0, 0, 0, 0, None, None)
            )(q_block, k_block, v_block, block_mask, self.causal, self.sm_scale)
            
            output = output.at[:, :, i:i_end, :].set(output_block)
            return output, q, k, v, mask

        output, *_ = lax.fori_loop(
            0, seq_len, 
            lambda i, carry: process_block(i, carry),
            (output, q, k, v, mask)
        )
        return output

    def __call__(self, x, context=None, mask=None, temporal_states=None, domain_id=0, is_training=True):
        """Forward pass for FlashMLAttentionTPU."""
        rng = hk.next_rng_key()
        # Project inputs to Q, K, V
        q = self._reshape_for_multihead(self.q_proj(x).astype(jnp.bfloat16))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context).astype(jnp.bfloat16))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context).astype(jnp.bfloat16))
        
        # Compute attention in blocks
        output = self._compute_attention_blockwise(q, k, v, mask)
        
        # Reshape and apply output projection
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(x.shape[0], x.shape[1], self.embed_dim)
        if is_training:
            output = hk.dropout(rng, self.dropout_rate, output)
            
        return self.o_proj(output.astype(jnp.float32))

# GPU MultiModalAttention
class MultiModalAttentionGPU(BaseAttentionGPU):
    def __init__(self, embed_dim, num_heads, num_domains=2, dropout=0.1, use_amp=True):
        super().__init__(embed_dim, num_heads, dropout, use_amp)
        self.num_domains = num_domains
        self.domain_projections = nn.ModuleList([nn.Linear(embed_dim, embed_dim) for _ in range(num_domains)])
        self.domain_mixing = nn.Parameter(torch.ones(num_domains, num_domains))
        self._reset_parameters_multi_modal()

    def _reset_parameters_multi_modal(self):
        self._reset_parameters()
        def triton_xavier_init(tensor, fan_in, fan_out):
            n_elements = tensor.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            xavier_init_kernel[grid](tensor.data_ptr(), n_elements, fan_in, fan_out, BLOCK_SIZE=1024)
        for proj in self.domain_projections:
            triton_xavier_init(proj.weight, self.embed_dim, self.embed_dim)
            nn.init.constant_(proj.bias, 0.0)
        triton_xavier_init(self.domain_mixing, self.num_domains, self.num_domains)

    def forward(self, x, domain_id=0, context=None, mask=None, temporal_states=None, is_training=True):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            batch_size, seq_len, _ = x.size()
            domain_features = [self._reshape_for_multihead(proj(x)) for proj in self.domain_projections]
            mixed_attention = 0
            for i in range(self.num_domains):
                q = domain_features[domain_id]
                k = v = domain_features[i]
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                attn_probs = F.softmax(scores, dim=-1)
                attn_probs = self.dropout(attn_probs) if is_training else attn_probs
                mixed_attention += self.domain_mixing[domain_id, i] * torch.matmul(attn_probs, v)
            mixed_attention = mixed_attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            return self.o_proj(mixed_attention)

# TPU MultiModalAttention
class MultiModalAttentionTPU(BaseAttentionTPU):
    def __init__(self, embed_dim, num_heads, num_domains=2, dropout_rate=0.1, name=None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.num_domains = num_domains
        self.domain_projections = [hk.Linear(embed_dim, w_init=hk.initializers.VarianceScaling(1/math.sqrt(2))) 
                                 for _ in range(num_domains)]
        self.domain_mixing = hk.get_parameter("domain_mixing", (num_domains, num_domains), 
                                            init=hk.initializers.VarianceScaling(1.0))

    @jit
    def __call__(self, x, domain_id=0, context=None, mask=None, temporal_states=None, rng=None, is_training=True):
        rng = rng or random.PRNGKey(0)
        batch_size, seq_len, _ = x.shape
        domain_features = [self._reshape_for_multihead(proj(x).astype(jnp.bfloat16)) for proj in self.domain_projections]
        mixed_attention = 0
        for i in range(self.num_domains):
            q = domain_features[domain_id]
            k = v = domain_features[i]
            scores = jnp.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = jnp.where(mask == 0, -jnp.inf, scores)
            attn_probs = jax.nn.softmax(scores, axis=-1)
            attn_probs = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn_probs) if is_training else attn_probs
            mixed_attention += self.domain_mixing[domain_id, i] * jnp.matmul(attn_probs, v)
        mixed_attention = jnp.transpose(mixed_attention, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
        return self.o_proj(mixed_attention.astype(jnp.float32))

# GPU TemporalAttention
class TemporalAttentionGPU(BaseAttentionGPU):
    def __init__(self, embed_dim, num_heads, max_temporal_length=512, dropout=0.1, use_amp=True):
        super().__init__(embed_dim, num_heads, dropout, use_amp)
        self.max_temporal_length = max_temporal_length
        self.temporal_embeddings = nn.Parameter(torch.randn(1, max_temporal_length, embed_dim))
        self.time_mixer = nn.Linear(embed_dim * 2, embed_dim)
        self._reset_parameters_temporal()

    def _reset_parameters_temporal(self):
        self._reset_parameters()
        def triton_xavier_init(tensor, fan_in, fan_out):
            n_elements = tensor.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            xavier_init_kernel[grid](tensor.data_ptr(), n_elements, fan_in, fan_out, BLOCK_SIZE=1024)
        triton_xavier_init(self.temporal_embeddings, self.max_temporal_length, self.embed_dim)
        triton_xavier_init(self.time_mixer.weight, self.embed_dim * 2, self.embed_dim)
        nn.init.constant_(self.time_mixer.bias, 0.0)

    def forward(self, x, temporal_positions=None, context=None, mask=None, is_training=True):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            batch_size, seq_len, _ = x.size()
            temporal_positions = torch.arange(seq_len, device=x.device) if temporal_positions is None else temporal_positions
            temporal_emb = self.temporal_embeddings[:, temporal_positions]
            x_temporal = self.time_mixer(torch.cat([x, temporal_emb], dim=-1))
            q = self._reshape_for_multihead(self.q_proj(x_temporal))
            k = self._reshape_for_multihead(self.k_proj(x_temporal if context is None else context))
            v = self._reshape_for_multihead(self.v_proj(x_temporal if context is None else context))
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn_probs = F.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs) if is_training else attn_probs
            output = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
            return self.o_proj(output)

# TPU TemporalAttention with Temporal Convolution
class TemporalAttentionTPU(BaseAttentionTPU):
    def __init__(self, embed_dim, num_heads, max_temporal_length=512, dropout_rate=0.1, name=None):
        super().__init__(embed_dim, num_heads, dropout_rate, name)
        self.max_temporal_length = max_temporal_length
        self.temporal_embeddings = hk.get_parameter("temporal_embeddings", (1, max_temporal_length, embed_dim),
                                                  init=hk.initializers.RandomNormal())
        self.time_mixer = hk.Linear(embed_dim * 2, w_init=hk.initializers.VarianceScaling(1/math.sqrt(2)))
        self.temporal_conv = hk.Conv1D(output_channels=embed_dim, kernel_shape=3, padding="CAUSAL")

    @jit
    def __call__(self, x, temporal_positions=None, context=None, mask=None, rng=None, is_training=True):
        rng = rng or random.PRNGKey(0)
        batch_size, seq_len, _ = x.shape
        temporal_positions = jnp.arange(seq_len) if temporal_positions is None else temporal_positions
        
        # Apply rotary embeddings to temporal positions
        freqs = jnp.exp(-temporal_positions[:, None] / 10000 ** (2 * jnp.arange(self.embed_dim // 2) / self.embed_dim))
        freqs_cis = jnp.exp(1j * freqs).astype(jnp.complex64)
        temporal_emb = apply_rotary_embedding(self.temporal_embeddings, freqs_cis)
        
        x_temporal = self.time_mixer(jnp.concatenate([x, temporal_emb], axis=-1))
        x_temporal = self.temporal_conv(x_temporal.transpose(0, 2, 1)).transpose(0, 2, 1)  # Temporal convolution
        q = self._reshape_for_multihead(self.q_proj(x_temporal).astype(jnp.bfloat16))
        k = self._reshape_for_multihead(self.k_proj(x_temporal if context is None else context).astype(jnp.bfloat16))
        v = self._reshape_for_multihead(self.v_proj(x_temporal if context is None else context).astype(jnp.bfloat16))
        scores = jnp.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = jnp.where(mask == 0, -jnp.inf, scores)
        attn_probs = jax.nn.softmax(scores, axis=-1)
        attn_probs = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn_probs) if is_training else attn_probs
        output = jnp.matmul(attn_probs, v)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(batch_size, seq_len, self.embed_dim)
        return self.o_proj(output.astype(jnp.float32))

# Sonnet Flash Attention TPU
class SonnetFlashAttentionTPU(snt.Module):
    def __init__(self, embed_dim, num_heads, block_size=128, dropout_rate=0.1, causal=False, sm_scale=0.5, name="sonnet_flash_tpu"):
        super().__init__(name=name)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.block_size = block_size
        self.dropout_rate = dropout_rate
        self.causal = causal
        self.sm_scale = sm_scale
        self.q_proj = snt.Linear(embed_dim)
        self.k_proj = snt.Linear(embed_dim)
        self.v_proj = snt.Linear(embed_dim)
        self.o_proj = snt.Linear(embed_dim)
        self.dropout = snt.Dropout(dropout_rate)

    def _reshape_for_multihead(self, x):
        batch_size, seq_len, _ = x.shape
        return tf.reshape(x, [batch_size, seq_len, self.num_heads, self.head_dim])

    @tf.function(jit_compile=True)
    def forward(self, x, context=None, mask=None, temporal_states=None, domain_id=0, is_training=True):
        x_tf = tf.experimental.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))
        context_tf = tf.experimental.dlpack.from_dlpack(jax.dlpack.to_dlpack(context)) if context is not None else x_tf
        mask_tf = tf.experimental.dlpack.from_dlpack(jax.dlpack.to_dlpack(mask)) if mask is not None else None
        q = tf.transpose(self._reshape_for_multihead(self.q_proj(x_tf)), [0, 2, 1, 3])
        k = tf.transpose(self._reshape_for_multihead(self.k_proj(context_tf)), [0, 2, 1, 3])
        v = tf.transpose(self._reshape_for_multihead(self.v_proj(context_tf)), [0, 2, 1, 3])
        batch_size, num_heads, seq_len, head_dim = q.shape
        output = tf.zeros_like(v, dtype=tf.bfloat16)
        for i in range(0, seq_len, self.block_size):
            i_end = tf.minimum(i + self.block_size, seq_len)
            q_block = q[:, :, i:i_end, :]
            k_block = k[:, :, :i_end, :] if self.causal else k
            v_block = v[:, :, :i_end, :] if self.causal else v
            scores = tf.einsum('bhqd,bhkd->bhqk', q_block, k_block) * self.sm_scale
            if self.causal:
                causal_mask = tf.linalg.band_part(tf.ones((i_end - i, i_end - i), dtype=tf.bfloat16), -1, 0)
                scores = tf.where(causal_mask[None, None, :, :], scores, -1e9)
            if mask_tf is not None:
                scores = tf.where(mask_tf[:, :, i:i_end, :i_end] == 0, -1e9, scores)
            attn_probs = tf.nn.softmax(scores, axis=-1)
            output = tf.tensor_scatter_nd_add(output, [[range(batch_size)] * (i_end - i), [range(i, i_end)] * batch_size],
                                            tf.einsum('bhqk,bhkd->bhqd', attn_probs, v_block))
        output = tf.transpose(output, [0, 2, 1, 3]).reshape([batch_size, seq_len, self.embed_dim])
        output = self.dropout(output, training=is_training)
        return jax.dlpack.from_dlpack(tf.experimental.dlpack.to_dlpack(self.o_proj(output)))

# Haiku Transformation Wrappers
def forward_flash_mla_tpu(x, context=None, mask=None, temporal_states=None, domain_id=0, is_training=True):
    model = FlashMLAttentionTPU(embed_dim=512, num_heads=8, block_size=128, causal=True)
    return hk.transform(lambda x, c, m, t, d, r, i: model(x, c, m, t, d, i))(x, context, mask, temporal_states, domain_id, None, is_training)

def forward_multimodal_tpu(x, domain_id=0, context=None, mask=None, temporal_states=None, is_training=True):
    model = MultiModalAttentionTPU(embed_dim=512, num_heads=8, num_domains=2)
    return hk.transform(lambda x, d, c, m, t, r, i: model(x, d, c, m, t, r, i))(x, domain_id, context, mask, temporal_states, None, is_training)

def forward_temporal_tpu(x, temporal_positions=None, context=None, mask=None, is_training=True):
    model = TemporalAttentionTPU(embed_dim=512, num_heads=8, max_temporal_length=512)
    return hk.transform(lambda x, t, c, m, r, i: model(x, t, c, m, r, i))(x, temporal_positions, context, mask, None, is_training)

# Hybrid Framework
class HybridAttention:
    def __init__(self, device_type="gpu", embed_dim=512, num_heads=8, **kwargs):
        self.device_type = device_type.lower()
        if self.device_type == "gpu":
            self.flash_mla = FlashMLAAttentionGPU(embed_dim, num_heads, **kwargs).cuda()
            self.multimodal = MultiModalAttentionGPU(embed_dim, num_heads, **kwargs).cuda()
            self.temporal = TemporalAttentionGPU(embed_dim, num_heads, **kwargs).cuda()
        elif self.device_type == "tpu":
            self.flash_mla = forward_flash_mla_tpu
            self.multimodal = forward_multimodal_tpu
            self.temporal = forward_temporal_tpu
            self.flash_mla_params = self.flash_mla.init(random.PRNGKey(0), jnp.ones((2, 64, embed_dim)))
            self.multimodal_params = self.multimodal.init(random.PRNGKey(0), jnp.ones((2, 64, embed_dim)), domain_id=0)
            self.temporal_params = self.temporal.init(random.PRNGKey(0), jnp.ones((2, 64, embed_dim)))

    def forward(self, x, attention_type="flash_mla", **kwargs):
        if self.device_type == "gpu":
            x = torch.tensor(x, device="cuda") if not torch.is_tensor(x) else x
            if attention_type == "flash_mla":
                return self.flash_mla(x, **kwargs)
            elif attention_type == "multimodal":
                return self.multimodal(x, **kwargs)
            elif attention_type == "temporal":
                return self.temporal(x, **kwargs)
        elif self.device_type == "tpu":
            x = jnp.array(x) if not isinstance(x, jnp.ndarray) else x
            if attention_type == "flash_mla":
                return self.flash_mla.apply(self.flash_mla_params, random.PRNGKey(0), x, **kwargs)
            elif attention_type == "multimodal":
                return self.multimodal.apply(self.multimodal_params, random.PRNGKey(0), x, **kwargs)
            elif attention_type == "temporal":
                return self.temporal.apply(self.temporal_params, random.PRNGKey(0), x, **kwargs)
        raise ValueError(f"Unsupported attention_type: {attention_type}")

# Example Usage
if __name__ == "__main__":
    # GPU Test
    gpu_model = HybridAttention(device_type="gpu")
    x_gpu = torch.randn(2, 64, 512).cuda()
    print("GPU FlashMLA:", gpu_model.forward(x_gpu, "flash_mla").shape)
    print("GPU MultiModal:", gpu_model.forward(x_gpu, "multimodal", domain_id=0).shape)
    print("GPU Temporal:", gpu_model.forward(x_gpu, "temporal").shape)

    # TPU Test
    tpu_model = HybridAttention(device_type="tpu")
    x_tpu = jnp.ones((2, 64, 512))
    print("TPU FlashMLA:", tpu_model.forward(x_tpu, "flash_mla").shape)
    print("TPU MultiModal:", tpu_model.forward(x_tpu, "multimodal", domain_id=0).shape)
    print("TPU Temporal:", tpu_model.forward(x_tpu, "temporal").shape)