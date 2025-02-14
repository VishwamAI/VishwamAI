import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from kernel import act_quant, weight_dequant, fp8_gemm
from cache_augmentation import CacheConfig, DifferentiableCacheAugmentation
from neural_memory import ReasoningMemoryTransformer
from tree_of_thoughts import TreeConfig, TreeOfThoughts

# Global settings
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    use_cache_augmentation: bool = True
    cache_hidden_size: int = 256
    cache_num_heads: int = 4
    cache_dropout: float = 0.1
    cache_max_length: int = 1024
    use_neural_memory: bool = True
    memory_size: int = 512
    num_memory_layers: int = 3
    memory_dropout: float = 0.1
    use_tree_of_thoughts: bool = True
    tot_max_branches: int = 4
    tot_max_depth: int = 3
    tot_beam_width: int = 2
    tot_temperature: float = 0.8
    tot_min_score_diff: float = 0.1

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """GPU-optimized linear operation with FP8/BF16 support."""
    if weight.element_size() > 1:
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y

class Linear(nn.Module):
    """GPU Linear layer with FP8/BF16 support."""
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or self.dtype))
        
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
            
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)

class ColumnParallelLinear(nn.Module):
    """GPU-optimized column parallel linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype = None):
        super().__init__()
        assert out_features % world_size == 0
        self.in_features = in_features
        self.out_features = out_features // world_size
        self.weight = nn.Parameter(torch.empty(self.out_features, in_features, dtype=dtype or Linear.dtype))
        
        if self.weight.element_size() == 1:
            scale_out_features = (self.out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
            
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    """GPU-optimized row parallel linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype = None):
        super().__init__()
        assert in_features % world_size == 0
        self.in_features = in_features // world_size
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features, dtype=dtype or Linear.dtype))
        
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (self.in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
            
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = linear(x, self.weight, None)
        if world_size > 1:
            dist.all_reduce(output)
        if self.bias is not None:
            output = output + self.bias
        return output

class RMSNorm(nn.Module):
    """RMS normalization layer."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class ParallelEmbedding(nn.Module):
    """Parallel embedding layer."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        assert vocab_size % world_size == 0
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(vocab_size // world_size, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.embedding(x, self.weight)
        if world_size > 1:
            dist.all_reduce(output)
        return output

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    freqs = torch.arange(0, dim, 2, dtype=torch.float32, device="cuda")
    freqs = args.rope_theta ** (-freqs / dim)
    t = torch.arange(seqlen, dtype=torch.float32, device="cuda")
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
    x_rotated = x_complex * freqs_cis
    return torch.view_as_real(x_rotated).flatten(3).type_as(x)

class MLA(nn.Module):
    """Multi-head Linear Attention."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        self.q_proj = ColumnParallelLinear(args.dim, args.dim)
        self.k_proj = ColumnParallelLinear(args.dim, args.dim)
        self.v_proj = ColumnParallelLinear(args.dim, args.dim)
        self.o_proj = RowParallelLinear(args.dim, args.dim)
        
        self.cache = None
        if args.use_cache_augmentation:
            self.cache = DifferentiableCacheAugmentation(
                CacheConfig(
                    hidden_size=args.cache_hidden_size,
                    num_heads=args.cache_num_heads,
                    dropout=args.cache_dropout,
                    max_length=args.cache_max_length
                )
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bs, seqlen, _ = x.size()
        
        # QKV projections
        q = self.q_proj(x).view(bs, seqlen, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bs, seqlen, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(bs, seqlen, self.n_heads, self.head_dim)
        
        # Compute attention
        q = q * (self.head_dim ** -0.5)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        if mask is not None:
            scores = scores + mask.unsqueeze(1).unsqueeze(2)
            
        scores = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        
        # Apply cache if enabled
        if self.cache is not None:
            v, _ = self.cache(v, mask)
            
        # Compute output
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.o_proj(output)

class Block(nn.Module):
    """Transformer block with attention and MLP."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.mlp = Linear(args.dim, args.inter_dim)
        self.norm1 = RMSNorm(args.dim)
        self.norm2 = RMSNorm(args.dim)
        
        # Optional components
        if args.use_neural_memory:
            self.memory = ReasoningMemoryTransformer(
                hidden_size=args.dim,
                memory_size=args.memory_size,
                num_layers=args.num_memory_layers,
                dropout=args.memory_dropout
            )
            
        if args.use_tree_of_thoughts:
            self.tot = TreeOfThoughts(
                hidden_size=args.dim,
                config=TreeConfig(
                    max_branches=args.tot_max_branches,
                    max_depth=args.tot_max_depth,
                    beam_width=args.tot_beam_width,
                    temperature=args.tot_temperature,
                    min_score_diff=args.tot_min_score_diff
                )
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attn(self.norm1(x), mask)
        
        if hasattr(self, 'memory'):
            h = self.memory(h)
            
        if hasattr(self, 'tot'):
            h = self.tot(h)
            
        return h + self.mlp(self.norm2(h))

class Transformer(nn.Module):
    """Main transformer model."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.blocks = nn.ModuleList([Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim)
        self.out_proj = ColumnParallelLinear(args.dim, args.vocab_size)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.embed(tokens)
        
        for block in self.blocks:
            h = block(h, mask)
            
        h = self.norm(h)
        output = self.out_proj(h)
        
        if world_size > 1:
            output_list = [torch.empty_like(output) for _ in range(world_size)]
            dist.all_gather(output_list, output)
            output = torch.cat(output_list, dim=-1)
            
        return output

if __name__ == "__main__":
    # Test model
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    
    args = ModelArgs()
    model = Transformer(args)
    
    # Generate random input
    bs, seqlen = 2, 1024
    x = torch.randint(0, args.vocab_size, (bs, seqlen))
    
    # Test forward pass
    with torch.inference_mode():
        out = model(x)
    print(f"Output shape: {out.shape}")
