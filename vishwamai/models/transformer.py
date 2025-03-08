# /home/kasinadhsarma/VishwamAI/vishwamai/models/transformer.py
"""
Device-agnostic VishwamAI Transformer with support for GPU and TPU execution.
Combines PyTorch (GPU) and JAX (TPU) implementations with dynamic routing.
"""

import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import flax.linen as flax_nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Import core layers and attention mechanisms
from vishwamai.models.kernel_layers import TokenEmbedding, PositionalEncoding, FeedForward
from vishwamai.models.attention import DynamicSparseAttention, LearnedPerformerAttention, OptimizedMoEAttention

# Import device-specific implementations
from vishwamai.models.gpu.transformer import VishwamAITransformer as GPUTransformer
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


class VishwamAITransformerLayer(DeviceAgnosticModule, nn.Module):
    """
    Single layer of the VishwamAI Transformer with OptimizedMoEAttention, HybridThoughtAwareAttention,
    Adaptive Reasoning Gate, Hybrid MoE-Dense layers, and Reasoning Depth Scaling.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, attention_kwargs, layer_idx, num_layers, dropout=0.1):
        """
        Initialize a VishwamAI Transformer layer.

        Args:
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward hidden dimension.
            attention_kwargs (dict): Keyword arguments for MoE attention.
            layer_idx (int): Index of the layer (for depth scaling).
            num_layers (int): Total number of layers (for depth scaling).
            dropout (float): Dropout rate.
        """
        super(VishwamAITransformerLayer, self).__init__()
        self.layer_idx = layer_idx
        self.num_layers = num_layers

        # Hybrid Attention: OptimizedMoEAttention with HybridThoughtAwareAttention as a fallback
        self.moe_attention = OptimizedMoEAttention(embed_dim, num_heads, **attention_kwargs)
        self.taa_attention = HybridThoughtAwareAttention(embed_dim, num_heads, **attention_kwargs.get('taa_kwargs', {}))
        self.attn_gate = nn.Linear(embed_dim, 2)  # Gate for MoE vs. TAA

        # Feed-forward network
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)

        # Adaptive Reasoning Gate
        self.arg = AdaptiveReasoningGate(embed_dim)

        # Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, context=None):
        """
        Forward pass of the VishwamAI Transformer layer.

        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, embed_dim).
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len, seq_len).
            context (torch.Tensor, optional): Context tensor for cross-domain attention.

        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, embed_dim).
        """
        # Reasoning Depth Scaling: Increase computation for deeper layers
        depth_factor = 1.0 + (self.layer_idx / self.num_layers) * 0.5  # Scale from 1.0 to 1.5

        # Adaptive Reasoning Gate weights
        attn_weight, ffn_weight = self.arg(x)

        # Hybrid Attention with gating
        moe_output = self.moe_attention(x, context) if context else self.moe_attention(x)
        taa_output = self.taa_attention(x, mask)
        attn_gate_weights = F.softmax(self.attn_gate(x.mean(dim=1)), dim=-1)  # (batch_size, 2)
        moe_weight, taa_weight = attn_gate_weights[:, 0].unsqueeze(-1).unsqueeze(-1), attn_gate_weights[:, 1].unsqueeze(-1).unsqueeze(-1)
        attn_output = (moe_weight * moe_output + taa_weight * taa_output) * depth_factor
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward block with ARG and depth scaling
        ffn_output = self.ffn(x) * ffn_weight * depth_factor
        x = self.norm2(x + self.dropout(ffn_output))

        return self.norm3(x)


class VishwamAITransformer(DeviceAgnosticModule, nn.Module):
    """
    Unique VishwamAI Transformer with advanced reasoning capabilities.
    """
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, max_seq_len=512,
                 attention_kwargs=None, dropout=0.1, force_device=None):
        """
        Initialize the VishwamAI Transformer.

        Args:
            vocab_size (int): Vocabulary size.
            embed_dim (int): Embedding dimension.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward hidden dimension.
            max_seq_len (int): Maximum sequence length.
            attention_kwargs (dict): Keyword arguments for MoE attention (e.g., num_experts, taa_kwargs).
            dropout (float): Dropout rate.
        """
        super(VishwamAITransformer, self).__init__()
        DeviceAgnosticModule.__init__(self)
        
        if force_device:
            self.device_type = force_device
            
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Initialize device-specific implementations
        if self.device_type == "gpu":
            self.model = GPUTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                max_seq_len=max_seq_len,
                attention_kwargs=attention_kwargs,
                dropout=dropout
            )
        elif self.device_type == "tpu" and HAS_JAX:
            self.model = TPUTransformer(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                max_seq_len=max_seq_len,
                attention_kwargs=attention_kwargs,
                dropout_rate=dropout
            )
        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

        # Embedding layers
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout)
        attention_kwargs = attention_kwargs or {"num_experts": 4, "taa_kwargs": {"k": 10, "kernel_dim": 256}}

        # Stack of VishwamAI transformer layers
        self.layers = nn.ModuleList([
            VishwamAITransformerLayer(embed_dim, num_heads, ff_dim, attention_kwargs, idx, num_layers, dropout)
            for idx in range(num_layers)
        ])

        # Output projection
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x, mask=None, context=None):
        # Convert inputs to appropriate device format
        x = self.to_device(x)
        if mask is not None:
            mask = self.to_device(mask)
        if context is not None:
            context = self.to_device(context)
            
        # Route to appropriate implementation
        if self.device_type == "gpu":
            return self.model(x, mask, context)
        elif self.device_type == "tpu":
            return self.model.apply({'params': self.model.params}, x, mask, context)
        """
        Forward pass of the VishwamAI Transformer.

        Args:
            x (torch.Tensor): Input token IDs (batch_size, seq_len).
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len, seq_len).
            context (torch.Tensor, optional): Context tensor for cross-domain attention.

        Returns:
            torch.Tensor: Logits (batch_size, seq_len, vocab_size).
        """
        # Embed tokens and add positional encodings
        x = self.token_embedding(x)
        x = self.positional_encoding(x)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, mask, context)

        # Final normalization and projection
        x = self.norm(x)
        logits = self.output_projection(x)
        return logits

    def get_hidden_state(self, x, mask=None, context=None):
        # Convert inputs to appropriate device format
        x = self.to_device(x)
        if mask is not None:
            mask = self.to_device(mask)
        if context is not None:
            context = self.to_device(context)
            
        # Route to appropriate implementation
        if self.device_type == "gpu":
            return self.model.get_hidden_state(x, mask, context)
        elif self.device_type == "tpu":
            return self.model.apply({'params': self.model.params}, x, mask, context, method=self.model.get_hidden_state)
        """
        Get hidden state from the transformer without final projection.

        Args:
            x (torch.Tensor): Input token IDs (batch_size, seq_len).
            mask (torch.Tensor, optional): Attention mask.
            context (torch.Tensor, optional): Context tensor for cross-domain attention.

        Returns:
            torch.Tensor: Hidden state (batch_size, seq_len, embed_dim).
        """
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        
        for layer in self.layers:
            x = layer(x, mask, context)
            
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
    attention_kwargs = {"num_experts": 4, "taa_kwargs": {"k": 10, "kernel_dim": 256}}

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
