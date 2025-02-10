import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Literal
import torch.distributed as dist
import torch.nn.functional as F
import math

@dataclass
class VishwamaiConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096  # Increased for Titan-like scale
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA from O1/MiniMax
    intermediate_size: int = 16384
    max_position_embeddings: int = 8192  # Increased context length
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000  # RoPE base
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    dtype: Literal["bf16", "fp8"] = "bf16"
    max_batch_size: int = 8
    max_seq_len: int = 8192
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.0
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 4096
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.0

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs_cis = self.precompute_freqs_cis(dim, theta)

    def precompute_freqs_cis(self, dim: int, theta: float) -> torch.Tensor:
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(self.max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def forward(self, positions):
        return self.freqs_cis[positions]

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

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

    def forward(self, x, attention_mask=None):
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(bsz, seqlen, self.num_attention_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.num_key_value_heads, self.head_dim)

        q = apply_rotary_emb(q, self.rotary_emb(torch.arange(seqlen, device=x.device)))
        k = apply_rotary_emb(k, self.rotary_emb(torch.arange(seqlen, device=x.device)))

        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
        attn_output = attn_output.reshape(bsz, seqlen, -1)
        return self.o_proj(attn_output)

class MoELayer(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.n_experts = config.n_routed_experts
        self.n_active = config.n_activated_experts
        self.hidden_dim = config.intermediate_size
        self.route_scale = config.route_scale
        
        # Expert network components
        self.gate = nn.Linear(config.hidden_size, self.n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, config.hidden_size)
            ) for _ in range(self.n_experts)
        ])

    def forward(self, x):
        # Gate computation
        route_scores = self.gate(x)
        routing_weights = F.softmax(route_scores, dim=-1)
        
        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, k=self.n_active, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        final_output = torch.zeros_like(x)
        for i in range(self.n_active):
            expert_idx = top_k_indices[:, :, i]
            expert_weight = top_k_weights[:, :, i].unsqueeze(-1)
            for j, expert in enumerate(self.experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = expert(expert_input)
                    final_output[mask] += expert_output * expert_weight[mask]
        
        return final_output * self.route_scale

class VishwamaiMLP(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down(self.act(self.gate(x)) * self.up(x))

class VishwamaiBlock(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.attention = VishwamaiAttention(config)
        self.moe = MoELayer(config)
        self.input_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, x, attention_mask=None):
        h = self.input_norm(x)
        h = self.attention(h, attention_mask)
        x = x + h
        
        h = self.post_attention_norm(x)
        h = self.moe(h)
        x = x + h
        
        return x

class VishwamaiModel(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.blocks = nn.ModuleList([VishwamaiBlock(config) for _ in range(config.num_hidden_layers)])
        
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self._device = "cpu"
        
    @property
    def device(self):
        return next(self.parameters()).device
        
    def to(self, device):
        return super().to(device)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        position_ids = torch.arange(seq_length, device=input_ids.device).expand(batch_size, -1)
        
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        hidden_states = inputs_embeds + position_embeds
        
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
            
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)
        
        return logits

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    print(model(input_ids).size())