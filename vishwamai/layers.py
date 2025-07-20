"""
Neural network layers and components for VishwamAI.

Implements optimized layers including feed-forward networks, normalization,
embeddings, and mixture-of-experts components.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Callable
import chex
import math

from .kernels import get_optimal_kernels


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    More stable and efficient alternative to LayerNorm,
    used in models like LLaMA and PaLM.
    """
    
    dim: int
    eps: float = 1e-6
    
    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.dim,))
    
    def __call__(self, x: chex.Array) -> chex.Array:
        """Apply RMS normalization."""
        
        # Compute RMS
        rms = jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + self.eps)
        
        # Normalize and scale
        normalized = x / rms
        return self.weight * normalized


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).
    
    Encodes positional information by rotating query and key embeddings
    in a way that preserves relative positions.
    """
    
    dim: int
    max_seq_len: int = 8192
    base: float = 10000.0
    
    def setup(self):
        # Precompute inverse frequencies
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        self.inv_freq = self.variable('constants', 'inv_freq', lambda: inv_freq)
    
    def __call__(self, seq_len: int) -> tuple[chex.Array, chex.Array]:
        """Generate cosine and sine embeddings."""
        
        t = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.outer(t, self.inv_freq.value)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        
        return jnp.cos(emb), jnp.sin(emb)


class FeedForward(nn.Module):
    """Feed-forward network with optional MoE and MoR support.
    
    Implements standard FFN, Mixture of Experts (MoE), and 
    Mixture of Recursions (MoR) variants for improved parameter efficiency.
    """
    
    dim: int
    hidden_dim: int
    dropout: float = 0.1
    activation: str = "gelu"
    use_moe: bool = False
    expert_count: int = 8
    expert_capacity: int = 4
    use_bias: bool = False
    
    # Mixture of Recursions parameters
    use_recursion: bool = False
    max_recursion_depth: int = 3
    recursion_capacity: int = 2
    
    def setup(self):
        kernels = get_optimal_kernels()
        
        if self.use_moe:
            # Mixture of Experts setup
            self.experts = [
                self._create_expert() for _ in range(self.expert_count)
            ]
            self.router = nn.Dense(
                features=self.expert_count,
                use_bias=self.use_bias,
                name="router"
            )
        
        if self.use_recursion:
            # Mixture of Recursions setup
            self.recursion_router = nn.Dense(
                features=self.max_recursion_depth,
                use_bias=self.use_bias,
                name="recursion_router"
            )
        else:
            # Standard FFN
            self.up_proj = nn.Dense(
                features=self.hidden_dim,
                use_bias=self.use_bias,
                name="up_proj"
            )
            self.down_proj = nn.Dense(
                features=self.dim,
                use_bias=self.use_bias,
                name="down_proj"
            )
        
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        
        # Use optimized activation if available
        if hasattr(kernels.kernels, 'fast_gelu') and kernels.kernels['fast_gelu']:
            self._activation_fn = kernels.kernels['fast_gelu']
        else:
            self._activation_fn = self._get_activation_fn()
    
    def _create_expert(self) -> nn.Module:
        """Create a single expert for MoE."""
        
        class Expert(nn.Module):
            dim: int = self.dim
            hidden_dim: int = self.hidden_dim
            use_bias: bool = self.use_bias
            
            def setup(self):
                self.up_proj = nn.Dense(
                    features=self.hidden_dim,
                    use_bias=self.use_bias
                )
                self.down_proj = nn.Dense(
                    features=self.dim,
                    use_bias=self.use_bias
                )
            
            def __call__(self, x):
                h = self.up_proj(x)
                h = jax.nn.gelu(h)  # Use standard GELU for experts
                return self.down_proj(h)
        
        return Expert()
    
    def _get_activation_fn(self) -> Callable:
        """Get activation function."""
        if self.activation == "gelu":
            return jax.nn.gelu
        elif self.activation == "relu":
            return jax.nn.relu
        elif self.activation == "swish":
            return jax.nn.swish
        elif self.activation == "silu":
            return jax.nn.silu
        else:
            return jax.nn.gelu
    
    def __call__(self, x: chex.Array, training: bool = True) -> chex.Array:
        """Forward pass through feed-forward network."""
        
        if self.use_recursion and self.use_moe:
            return self._recursive_moe_forward(x, training)
        elif self.use_moe:
            return self._moe_forward(x, training)
        else:
            return self._standard_forward(x, training)
    
    def _standard_forward(self, x: chex.Array, training: bool) -> chex.Array:
        """Standard feed-forward computation."""
        
        # Check if we can use fused MLP kernel
        kernels = get_optimal_kernels()
        if kernels.kernels.get('fused_mlp') is not None:
            # Use optimized fused kernel
            up_weights = self.up_proj.variables['params']['kernel']
            down_weights = self.down_proj.variables['params']['kernel']
            up_bias = self.up_proj.variables['params'].get('bias')
            down_bias = self.down_proj.variables['params'].get('bias')
            
            output = kernels.kernels['fused_mlp'](
                x, up_weights, down_weights, up_bias, down_bias, self.activation
            )
        else:
            # Standard computation
            h = self.up_proj(x)
            h = self._activation_fn(h)
            output = self.down_proj(h)
        
        # Apply dropout
        if training:
            output = self.dropout_layer(output, deterministic=not training)
        
        return output
    
    def _moe_forward(self, x: chex.Array, training: bool) -> chex.Array:
        """Mixture of Experts forward pass."""
        
        batch_size, seq_len, dim = x.shape
        
        # Router logits
        router_logits = self.router(x)  # [batch, seq, expert_count]
        
        # Compute routing probabilities
        routing_weights = jax.nn.softmax(router_logits, axis=-1)
        
        # Select top-k experts per token
        topk_weights, topk_indices = jax.lax.top_k(routing_weights, self.expert_capacity)
        
        # Normalize weights
        topk_weights = topk_weights / jnp.sum(topk_weights, axis=-1, keepdims=True)
        
        # Compute expert outputs
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(x)  # [batch, seq, dim]
            expert_outputs.append(expert_out)
        
        expert_outputs = jnp.stack(expert_outputs, axis=-2)  # [batch, seq, expert_count, dim]
        
        # Combine expert outputs using routing weights
        output = jnp.zeros_like(x)
        for k in range(self.expert_capacity):
            expert_idx = topk_indices[:, :, k]  # [batch, seq]
            weight = topk_weights[:, :, k:k+1]  # [batch, seq, 1]
            
            # Gather expert outputs
            expert_out = jnp.take_along_axis(
                expert_outputs,
                expert_idx[:, :, None, None],
                axis=-2
            ).squeeze(-2)  # [batch, seq, dim]
            
            output = output + weight * expert_out
        
        # Apply dropout
        if training:
            output = self.dropout_layer(output, deterministic=not training)
        
        return output
    
    def _recursive_moe_forward(self, x: chex.Array, training: bool) -> chex.Array:
        """Mixture of Recursions forward pass with selective computation."""
        
        batch_size, seq_len, dim = x.shape
        
        recursion_logits = self.recursion_router(x)  # [batch, seq, max_depth]
        recursion_weights = jax.nn.softmax(recursion_logits, axis=-1)
        
        # Select top-k recursion depths per token (ensure k <= available depths)
        k = min(self.recursion_capacity, self.max_recursion_depth)
        topk_depths, topk_depth_indices = jax.lax.top_k(recursion_weights, k)
        topk_depths = topk_depths / jnp.sum(topk_depths, axis=-1, keepdims=True)
        
        output = jnp.zeros_like(x)
        kv_cache = {}
        
        for depth in range(self.max_recursion_depth):
            # Create mask for tokens using this depth
            depth_mask = (topk_depth_indices == depth).any(axis=-1, keepdims=True)  # [batch, seq, 1]
            active_tokens = jnp.where(depth_mask, x, 0)
            
            if depth == 0:
                depth_output = self._moe_forward(active_tokens, training)
                kv_cache[depth] = depth_output
            else:
                prev_kv = kv_cache.get(depth-1, jnp.zeros_like(active_tokens))
                depth_output = self._moe_forward(active_tokens + prev_kv, training)
                kv_cache[depth] = depth_output
            
            # Weight the output by recursion weights for this depth
            depth_weight_mask = (topk_depth_indices == depth)  # [batch, seq, k]
            if depth_weight_mask.any():
                depth_weights = jnp.where(depth_weight_mask, topk_depths, 0).sum(axis=-1, keepdims=True)  # [batch, seq, 1]
                output = output + depth_weights * depth_output
        
        return output


class GLU(nn.Module):
    """Gated Linear Unit (GLU) activation.
    
    Splits input into two parts, applies sigmoid to one part
    and uses it to gate the other part.
    """
    
    dim: int
    
    def setup(self):
        self.proj = nn.Dense(features=2 * self.dim, use_bias=False)
    
    def __call__(self, x: chex.Array) -> chex.Array:
        """Apply GLU activation."""
        
        x_proj = self.proj(x)
        x1, x2 = jnp.split(x_proj, 2, axis=-1)
        return x1 * jax.nn.sigmoid(x2)


class SwiGLU(nn.Module):
    """SwiGLU activation function.
    
    Combines Swish activation with GLU gating mechanism.
    Used in models like PaLM and LLaMA.
    """
    
    dim: int
    hidden_dim: Optional[int] = None
    
    def setup(self):
        if self.hidden_dim is None:
            self.hidden_dim = self.dim * 4
        
        self.gate_proj = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.up_proj = nn.Dense(features=self.hidden_dim, use_bias=False)
        self.down_proj = nn.Dense(features=self.dim, use_bias=False)
    
    def __call__(self, x: chex.Array) -> chex.Array:
        """Apply SwiGLU activation."""
        
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU: swish(gate) * up
        activated = jax.nn.swish(gate) * up
        
        return self.down_proj(activated)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.
    
    Standard transformer positional encoding using sine and cosine functions.
    """
    
    dim: int
    max_seq_len: int = 8192
    
    def setup(self):
        # Precompute positional encodings
        position = jnp.arange(self.max_seq_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.dim, 2) * (-math.log(10000.0) / self.dim)
        )
        
        pe = jnp.zeros((self.max_seq_len, self.dim))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        
        self.pe = self.variable('constants', 'pe', lambda: pe)
    
    def __call__(self, x: chex.Array) -> chex.Array:
        """Add positional encoding to input."""
        
        seq_len = x.shape[1]
        pos_emb = self.pe.value[:seq_len]
        
        return x + pos_emb[None, :, :]


class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings."""
    
    max_seq_len: int
    dim: int
    
    def setup(self):
        self.pos_embedding = nn.Embed(
            num_embeddings=self.max_seq_len,
            features=self.dim
        )
    
    def __call__(self, x: chex.Array) -> chex.Array:
        """Add learned positional encoding."""
        
        seq_len = x.shape[1]
        positions = jnp.arange(seq_len)
        pos_emb = self.pos_embedding(positions)
        
        return x + pos_emb[None, :, :]


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional weight tying."""
    
    vocab_size: int
    dim: int
    scale_by_sqrt_dim: bool = True
    
    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.dim
        )
        
        if self.scale_by_sqrt_dim:
            self.scale = math.sqrt(self.dim)
        else:
            self.scale = 1.0
    
    def __call__(self, input_ids: chex.Array) -> chex.Array:
        """Convert token IDs to embeddings."""
        
        embeddings = self.embedding(input_ids)
        return embeddings * self.scale
    
    def decode(self, embeddings: chex.Array) -> chex.Array:
        """Convert embeddings back to token logits (weight tying)."""
        
        # Use embedding weights as output projection
        vocab_weights = self.embedding.variables['params']['embedding']
        logits = jnp.dot(embeddings, vocab_weights.T)
        
        return logits
