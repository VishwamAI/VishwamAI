import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
# from .fp8_cast_bf16 import fp8_cast # Removed import as fp8_cast_bf16 is not defined in provided code

@dataclass
class VishwamaiConfig:
    # Model architecture
    vocab_size: int = 256000  # Increased for better coverage
    hidden_size: int = 12288  # Scaled for 641B parameters
    num_hidden_layers: int = 96
    num_attention_heads: int = 96
    num_key_value_heads: int = 12  # GQA for memory efficiency
    intermediate_size: int = 49152
    max_position_embeddings: int = 32768  # Extended context

    # Memory configuration
    memory_layers: int = 8
    memory_size: int = 16384
    memory_heads: int = 16
    memory_momentum: float = 0.9
    memory_decay: float = 0.1
    use_memory: bool = True

    # Training hyperparameters
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000
    num_reasoning_steps: int = 16
    max_reasoning_depth: int = 8

    # Tokenization
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Optimization
    use_fp8_training: bool = True
    use_quantization: bool = True
    quantization_bits: int = 8
    load_balancing: str = "auxiliary_free"  # Options: auxiliary_free, standard

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # RMSNorm implementation
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute rotary position embeddings.

        Args:
            positions: Tensor of shape (batch_size, seq_length) containing position indices

        Returns:
            Tensor of shape (batch_size, seq_length, dim) containing position embeddings
        """
        device = positions.device
        half_dim = self.dim // 2

        # Create position embeddings for half the dimension
        emb = torch.arange(half_dim, device=device).float()
        emb = self.theta ** (2 * (emb // 2) / half_dim)

        # Compute sin and cos
        freqs = positions.unsqueeze(-1) * emb.unsqueeze(0)
        sin = torch.sin(freqs)
        cos = torch.cos(freqs)

        # Duplicate for complex rotation
        sin = torch.cat([sin, sin], dim=-1)
        cos = torch.cat([cos, cos], dim=-1)

        return sin, cos

class VishwamaiAttention(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size,
                               self.head_dim * config.num_key_value_heads,
                               bias=False)
        self.v_proj = nn.Linear(config.hidden_size,
                               self.head_dim * config.num_key_value_heads,
                               bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, hidden_size = hidden_states.shape
        head_dim = hidden_size // self.num_attention_heads

        # Project inputs to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_length, self.num_attention_heads, head_dim)
        k = k.view(batch_size, seq_length, self.num_key_value_heads, head_dim)
        v = v.view(batch_size, seq_length, self.num_key_value_heads, head_dim)

        # Apply rotary position embeddings
        position_ids = torch.arange(seq_length, device=hidden_states.device)
        sin_pos, cos_pos = self.rotary_emb(position_ids)

        def rotate_half(x):
            x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
            return torch.cat([-x2, x1], dim=-1)

        # Apply rotary embeddings
        cos_pos_q = cos_pos.view(1, seq_length, 1, head_dim)
        sin_pos_q = sin_pos.view(1, seq_length, 1, head_dim)
        cos_pos_k = cos_pos.view(1, seq_length, 1, head_dim)
        sin_pos_k = sin_pos.view(1, seq_length, 1, head_dim)

        # Apply rotary embeddings
        q_rot = rotate_half(q)
        k_rot = rotate_half(k)

        q_embed = (q * cos_pos_q + q_rot * sin_pos_q)
        k_embed = (k * cos_pos_k + k_rot * sin_pos_k)
        v_embed = v  # Values don't need rotary embeddings

        # Handle Grouped-Query Attention
        k_embed = k_embed.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=2)
        v_embed = v_embed.repeat_interleave(self.num_attention_heads // self.num_key_value_heads, dim=2)

        # Compute scaled dot-product attention
        scale = torch.sqrt(torch.tensor(head_dim, dtype=torch.float32, device=hidden_states.device))
        scores = torch.matmul(q_embed, k_embed.transpose(2, 3)) / scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_bias = attention_mask[:, None, None, :].expand(batch_size, self.num_attention_heads, seq_length, seq_length)
            mask = (1.0 - attention_bias) * torch.finfo(scores.dtype).min
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_embed)

        # Reshape and project output
        attn_output = attn_output.contiguous().view(batch_size, seq_length, hidden_size) # Corrected reshape
        output = self.o_proj(attn_output)

        return output

class VishwamaiMLP(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        # SwiGLU activation
        return self.down(self.act(self.gate(x)) * self.up(x))

class AdvancedMemoryModule(nn.Module):
    """Enhanced memory module with multi-head attention and hierarchical structure"""
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.config = config

        # Multi-head memory attention
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.memory_heads,
            batch_first=True
        )

        # Hierarchical memory layers
        self.memory_hierarchy = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(config.memory_layers)
        ])

        # Surprise and relevance networks
        self.surprise_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)
        )

        self.relevance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 1)
        )

        # Memory compression
        self.compressor = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.decompressor = nn.Linear(config.hidden_size // 2, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, prev_memory: Optional[torch.Tensor] = None):
        batch_size, seq_length, hidden_size = hidden_states.shape
        device = hidden_states.device

        # Initialize memory if needed
        if prev_memory is None:
            prev_memory = torch.zeros(
                batch_size,
                self.config.memory_size,
                self.config.hidden_size,  # Full hidden size for memory
                device=device
            )

        # Ensure memory has correct shape
        if prev_memory.shape != (batch_size, self.config.memory_size, self.config.hidden_size):
            raise ValueError(f"Memory shape {prev_memory.shape} does not match expected shape {(batch_size, self.config.memory_size, self.config.hidden_size)}")
        # Ensure memory is on the correct device
        if prev_memory.device != hidden_states.device:
            prev_memory = prev_memory.to(hidden_states.device)

        # Calculate surprise and relevance scores
        surprise = torch.sigmoid(self.surprise_net(hidden_states))
        relevance = torch.sigmoid(self.relevance_net(hidden_states))

        # Update memory through hierarchy
        memory = prev_memory
        for layer in self.memory_hierarchy:
            # Decompress memory for processing
            decompressed_memory = self.decompressor(memory)
            # Ensure decompressed memory is on the correct device
            decompressed_memory = decompressed_memory.to(hidden_states.device)

            # Combine with input through attention
            attended_memory, _ = self.memory_attention(
                query=hidden_states,
                key=decompressed_memory,
                value=decompressed_memory
            )

            # Update memory based on surprise and relevance
            memory_input = torch.cat([attended_memory, hidden_states], dim=-1)
            memory_update = layer(memory_input)

        # Apply gated update
        update_gate = surprise * relevance
        memory = (
            (1 - self.config.memory_decay) * memory +
            self.config.memory_momentum * self.compressor(memory_update) * update_gate
        )
        # Ensure memory is on the correct device
        memory = memory.to(hidden_states.device)

        # Decompress final memory state for output
        return self.decompressor(memory)

class VishwamaiBlock(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.attention = VishwamaiAttention(config)
        self.mlp = VishwamaiMLP(config)
        self.memory = AdvancedMemoryModule(config)

        # Normalization layers
        self.input_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.memory_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)

        # Reasoning gate for memory integration
        self.reasoning_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, x, attention_mask=None, memory_state=None):
        # Pre-norm attention
        h = self.input_norm(x)
        h = self.attention(h, attention_mask)
        if h is not None:
            x = x + h

        # Memory component with normalization
        m = self.memory_norm(x)
        memory_output = self.memory(m, memory_state)

        # Memory integration with reasoning gate
        combined = torch.cat([x, memory_output], dim=-1)
        gate = torch.sigmoid(self.reasoning_gate(combined))
        x = x + gate * memory_output

        # Feed-forward network with normalization
        h = self.post_attention_norm(x)
        h = self.mlp(h)
        x = x + h

        return x, memory_output

class VishwamaiModel(nn.Module):
    def __init__(self, config: VishwamaiConfig, device: str = 'cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.to(device)

        # Standard components
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size).to(device)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size).to(device)

        # Transformer blocks with memory
        self.blocks = nn.ModuleList([VishwamaiBlock(config).to(device) for _ in range(config.num_hidden_layers)])

        # Final components
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps).to(device)
        self.head = nn.Linear(config.hidden_size, config.vocab_size).to(device)

        # Multi-token prediction head
        self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size).to(device)

    def forward(self, input_ids, attention_mask=None, memory_state=None):
        batch_size, seq_length = input_ids.shape

        # Move input_ids to correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        # Create position IDs and embeddings
        position_ids = torch.arange(seq_length, device=self.device).expand(batch_size, -1)
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)

        # Combine embeddings
        hidden_states = inputs_embeds + position_embeds

        # Initialize or validate memory state
        if memory_state is None:
            memory_state = [None] * len(self.blocks)
        elif isinstance(memory_state, list) and len(memory_state) != len(self.blocks):
            raise ValueError(f"Memory state length {len(memory_state)} does not match number of blocks {len(self.blocks)}")

        # Process through transformer blocks with memory
        new_memory_state = []
        for i, block in enumerate(self.blocks):
            # Handle device mismatch
            block_memory_state = memory_state[i]
            if block_memory_state is not None and block_memory_state.device != hidden_states.device:
                block_memory_state = block_memory_state.to(hidden_states.device)

            hidden_states, block_memory = block(hidden_states, attention_mask, block_memory_state)
            new_memory_state.append(block_memory)

        # Final processing
        hidden_states = self.ln_f(hidden_states)

        # Main language modeling head
        logits = self.head(hidden_states)

        # Multi-token prediction (predict next token and token after)
        mtp_logits = self.mtp_head(hidden_states)

        return {
            'logits': logits,
            'mtp_logits': mtp_logits,
            'memory_state': new_memory_state,
            'hidden_states': hidden_states
        }
