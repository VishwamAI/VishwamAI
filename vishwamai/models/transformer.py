# /home/kasinadhsarma/VishwamAI/vishwamai/models/transformer.py
"""
VishwamAI Transformer with unified GPU/TPU support and optimizations.
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Optional, Dict, Any

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import flax.linen as flax_nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Import core layers and attention mechanisms
from vishwamai.models.kernel_layers import TokenEmbedding, PositionalEncoding, FeedForward, OptimizedLayerNorm, DeviceAgnosticModule
from vishwamai.models.attention import DynamicSparseAttention, LearnedPerformerAttention, OptimizedMoEAttention, FlashMLAttention, TPUOptimizedAttention

# Import device-specific implementations
from vishwamai.models.gpu.transformer import DualPipeTransformer as GPUTransformer
from vishwamai.models.tpu.transformer import VishwamAITransformer as TPUTransformer

def get_device_type():
    """Determine the available device type."""
    if torch.cuda.is_available():
        return "gpu"
    elif HAS_JAX and len(jax.devices("tpu")) > 0:
        return "tpu"
    return "cpu"

class DeviceAgnosticModule:
    """Base class for device-agnostic modules"""
    def __init__(self):
        self.device_type = get_device_type()
        self.gpu_module = None
        self.tpu_module = None
    
    def to_device(self, x):
        """Convert input to appropriate device format"""
        if self.device_type == "tpu" and HAS_JAX:
            if isinstance(x, torch.Tensor):
                return jnp.array(x.cpu().numpy())
        elif self.device_type == "gpu":
            if not isinstance(x, torch.Tensor):
                return torch.tensor(x)
        return x

class HybridThoughtAwareAttention(DeviceAgnosticModule, nn.Module):
    """
    Hybrid Thought-Aware Attention combining DynamicSparseAttention and LearnedPerformerAttention
    to prioritize tokens relevant to intermediate thoughts with learned sparsity and efficiency.
    """
    def __init__(self, embed_dim, num_heads, k=10, kernel_dim=256, dropout=0.1):
        """
        Initialize Hybrid Thought-Aware Attention.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            k (int): Number of top-k tokens for sparsity.
            kernel_dim (int): Dimension of feature maps for Performer.
            dropout (float): Dropout rate.
        """
        super(HybridThoughtAwareAttention, self).__init__()
        self.sparse_attn = DynamicSparseAttention(embed_dim, num_heads, k=k)
        self.performer_attn = LearnedPerformerAttention(embed_dim, num_heads, kernel_dim=kernel_dim)
        self.gate = nn.Linear(embed_dim, 2)  # Weights for sparse vs. performer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Forward pass of Hybrid Thought-Aware Attention.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: Attention output (batch_size, seq_len, embed_dim).
        """
        sparse_output = self.sparse_attn(x)
        performer_output = self.performer_attn(x)
        
        # Gate weights based on input context
        avg_hidden = x.mean(dim=1)  # (batch_size, embed_dim)
        gate_weights = F.softmax(self.gate(avg_hidden), dim=-1)  # (batch_size, 2)
        sparse_weight, performer_weight = gate_weights[:, 0].unsqueeze(-1).unsqueeze(-1), gate_weights[:, 1].unsqueeze(-1).unsqueeze(-1)

        # Combine outputs
        output = sparse_weight * sparse_output + performer_weight * performer_output
        return self.dropout(output)


class AdaptiveReasoningGate(DeviceAgnosticModule, nn.Module):
    """
    Adaptive Reasoning Gate (ARG) to dynamically adjust attention and feed-forward computations
    based on reasoning complexity, enhanced with MoE load balancing insights.
    """
    def __init__(self, embed_dim, gate_dim=128):
        super(AdaptiveReasoningGate, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, gate_dim),
            nn.GELU(),
            nn.Linear(gate_dim, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Compute gate weights for attention and feed-forward.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, embed_dim).

        Returns:
            tuple: (attn_weight, ffn_weight) - Weights for attention and feed-forward.
        """
        avg_hidden = x.mean(dim=1)  # (batch_size, embed_dim)
        gate_weights = self.gate(avg_hidden)  # (batch_size, 2)
        attn_weight, ffn_weight = gate_weights[:, 0], gate_weights[:, 1]
        return attn_weight.unsqueeze(-1).unsqueeze(-1), ffn_weight.unsqueeze(-1).unsqueeze(-1)


class VishwamAITransformerLayer(DeviceAgnosticModule):
    """Single transformer layer with hardware-specific optimizations."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int,
                 attention_kwargs: Dict[str, Any], layer_idx: int, num_layers: int,
                 dropout_rate: float = 0.1, force_device: str = None):
        super().__init__()
        if force_device:
            self.device_type = force_device
            
        self.layer_idx = layer_idx
        self.num_layers = num_layers
        
        # Select attention implementation based on device
        if self.device_type == "tpu" and HAS_JAX:
            self.attention = TPUOptimizedAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout_rate
            )
        elif self.device_type == "gpu":
            self.attention = FlashMLAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout_rate,
                **attention_kwargs.get('flash_kwargs', {})
            )
        else:
            self.attention = DynamicSparseAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout_rate
            )
            
        # Device-agnostic components
        self.norm1 = OptimizedLayerNorm(embed_dim, force_device=self.device_type)
        self.norm2 = OptimizedLayerNorm(embed_dim, force_device=self.device_type)
        self.ff = FeedForward(embed_dim, ff_dim, dropout_rate, force_device=self.device_type)
        
    def __call__(self, x, mask=None, context=None, layer_states=None, training=False):
        # Reasoning Depth Scaling factor
        depth_factor = 1.0 + (self.layer_idx / self.num_layers) * 0.5
        
        # Attention block with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, context=context, mask=mask)
        x = x * depth_factor + residual
        
        # Feed-forward block with residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x * depth_factor + residual
        
        return x

class VishwamAITransformer(DeviceAgnosticModule):
    """
    Device-agnostic transformer with optimizations for different hardware.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ff_dim: int,
        max_seq_len: int = 512,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        dropout_rate: float = 0.1,
        force_device: str = None
    ):
        super().__init__()
        if force_device:
            self.device_type = force_device
            
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Device-agnostic embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size, embed_dim, force_device=self.device_type
        )
        self.pos_embedding = PositionalEncoding(
            embed_dim, max_seq_len, dropout_rate, force_device=self.device_type
        )
        
        # Initialize attention kwargs
        self.attention_kwargs = attention_kwargs or {
            "num_experts": 4,
            "flash_kwargs": {
                "latent_dim": 64,
                "block_size": 128
            }
        }
        
        # Create transformer layers
        self.layers = [
            VishwamAITransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                attention_kwargs=self.attention_kwargs,
                layer_idx=i,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                force_device=self.device_type
            )
            for i in range(num_layers)
        ]
        
        # Final normalization and output projection
        self.norm = OptimizedLayerNorm(embed_dim, force_device=self.device_type)
        
        if self.device_type == "tpu" and HAS_JAX:
            self.output_projection = flax_nn.Dense(vocab_size)
        else:
            import torch.nn as nn
            self.output_projection = nn.Linear(embed_dim, vocab_size)
            
    def __call__(self, x, mask=None, context=None, training=False):
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        
        layer_states = []
        for layer in self.layers:
            layer_output = layer(x, mask, context, layer_states, training)
            layer_states.append(layer_output)
            x = layer_output
            
        x = self.norm(x)
        return self.output_projection(x)
        
    def get_hidden_state(self, x, mask=None, context=None, training=False):
        """Get the final hidden state without output projection."""
        x = self.token_embedding(x)
        x = self.pos_embedding(x)
        
        layer_states = []
        for layer in self.layers:
            layer_output = layer(x, mask, context, layer_states, training)
            layer_states.append(layer_output)
            x = layer_output
            
        return self.norm(x)


# Example usage
if __name__ == "__main__":
    # Mock tokenizer for testing
    class MockTokenizer:
        def __init__(self, vocab_size=50000):
            self.vocab_size = vocab_size
            self.special_tokens = {
                "<think>": vocab_size-4, "</think>": vocab_size-3,
                "<answer>": vocab_size-2, "</answer>": vocab_size-1
            }
            self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
            self.inverse_vocab.update({i: f"token_{i}" for i in range(vocab_size-4)})

        def encode(self, text, return_tensors="pt"):
            tokens = [self.special_tokens.get(text, i) for i in range(5)]  # Simplified
            if return_tensors == "pt":
                return torch.tensor([tokens], dtype=torch.long)
            return tokens

        def decode(self, token_ids, skip_special_tokens=False):
            text = ""
            for token in token_ids:
                if token.item() in self.inverse_vocab:
                    if not skip_special_tokens or token.item() < self.vocab_size-4:
                        text += self.inverse_vocab[token.item()] + " "
            return text.strip()

    # Initialize the transformer
    vocab_size = 50000
    embed_dim = 512
    num_layers = 12
    num_heads = 8
    ff_dim = 2048
    max_seq_len = 512
    attention_kwargs = {"num_experts": 4, "flash_kwargs": {"k": 10, "kernel_dim": 256}}

    transformer = VishwamAITransformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        max_seq_len=max_seq_len,
        attention_kwargs=attention_kwargs
    )

    # Test with mock input
    tokenizer = MockTokenizer(vocab_size)
    input_text = "Test input"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(transformer.device)
    logits = transformer(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {logits.shape}")
    
    # Test with context (e.g., cross-domain input)
    context_ids = tokenizer.encode("Context input", return_tensors="pt").to(transformer.device)
    logits_with_context = transformer(input_ids, context=context_ids)
    print(f"Output logits with context shape: {logits_with_context.shape}")
