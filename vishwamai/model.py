import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

# Local imports
from cpu_kernels import act_quant, weight_dequant, fp8_gemm
from triton_cpu_kernels import TritonLinearCPU, triton_layer_norm
from cache_augmentation import CacheConfig, DifferentiableCacheAugmentation
from neural_memory import ReasoningMemoryTransformer
from tree_of_thoughts import TreeConfig, TreeOfThoughts

# Global settings
world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

# Model configuration
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
    """Basic linear operation with CPU/GPU implementation."""
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

# Base linear layer
class Linear(nn.Module):
    """Base linear layer with CPU/GPU implementations."""
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
            if self.weight.element_size() == 1:
                scale_out_features = (out_features + block_size - 1) // block_size
                scale_in_features = (in_features + block_size - 1) // block_size
                self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
            else:
                self.register_parameter("scale", None)
        else:
            # Use CPU optimized implementation
            self.cpu_linear = TritonLinearCPU(in_features, out_features, bias, dtype)
            self.weight = self.cpu_linear.weight
            
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'cpu':
            return self.cpu_linear(x)
        else:
            return linear(x, self.weight, self.bias)

# Parallel linear layers
class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype = None):
        super().__init__()
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.in_features = in_features
        self.out_features = out_features // world_size
        
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.empty(self.out_features, in_features, dtype=dtype or Linear.dtype))
            if self.weight.element_size() == 1:
                scale_out_features = (self.out_features + block_size - 1) // block_size
                scale_in_features = (in_features + block_size - 1) // block_size
                self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
            else:
                self.register_parameter("scale", None)
        else:
            # Use CPU optimized implementation
            self.cpu_linear = TritonLinearCPU(in_features, self.out_features, bias, dtype)
            self.weight = self.cpu_linear.weight
            
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'cpu':
            return self.cpu_linear(x)
        else:
            return linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype = None):
        super().__init__()
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.in_features = in_features // world_size
        self.out_features = out_features
        
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.empty(out_features, self.in_features, dtype=dtype or Linear.dtype))
            if self.weight.element_size() == 1:
                scale_out_features = (out_features + block_size - 1) // block_size
                scale_in_features = (self.in_features + block_size - 1) // block_size
                self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
            else:
                self.register_parameter("scale", None)
        else:
            # Use CPU optimized implementation
            self.cpu_linear = TritonLinearCPU(self.in_features, out_features, bias, dtype)
            self.weight = self.cpu_linear.weight
            
        if bias:
            # Bias is applied after the parallel reduction
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == 'cpu':
            output = self.cpu_linear(x)
        else:
            output = linear(x, self.weight, None)  # Don't apply bias yet
            
        if world_size > 1:
            dist.all_reduce(output)
            
        if self.bias is not None:
            output = output + self.bias
            
        return output

# Common layers
class RMSNorm(nn.Module):
    """RMS normalization with CPU/GPU implementations."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        if x.device.type == 'cpu':
            return triton_layer_norm(x, self.weight, None, self.eps)
        else:
            return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class ParallelEmbedding(nn.Module):
    """Parallel embedding layer."""
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y

def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """Precompute frequencies for rotary embeddings."""
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to input tensors."""
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

class MLA(nn.Module):
    """Multi-head Linear Attention with cache augmentation."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        
        # Query, Key, Value projections
        self.q_proj = ColumnParallelLinear(args.dim, args.n_heads * args.qk_nope_head_dim)
        self.k_proj = ColumnParallelLinear(args.dim, args.n_heads * args.qk_nope_head_dim)
        self.v_proj = ColumnParallelLinear(args.dim, args.n_heads * args.v_head_dim)
        
        # Output projection
        self.o_proj = RowParallelLinear(args.n_heads * args.v_head_dim, args.dim)
        
        # Attention scaling
        self.scale = args.qk_nope_head_dim ** -0.5
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale
        
        # Rotary embeddings
        self.rope_theta = args.rope_theta
        self.max_seq_len = args.max_seq_len
        
        # Cache setup
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_heads,
                args.qk_nope_head_dim
            ), persistent=False)
            self.register_buffer("v_cache", torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_heads,
                args.v_head_dim
            ), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        
        q = xq.view(bsz, seqlen, self.n_local_heads, -1)
        k = xk.view(bsz, seqlen, self.n_local_heads, -1)
        v = xv.view(bsz, seqlen, self.n_local_heads, -1)
        
        # Apply rotary embeddings
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        
        # Update cache
        if attn_impl == "naive":
            self.k_cache[:bsz, start_pos:start_pos+seqlen] = k
            self.v_cache[:bsz, start_pos:start_pos+seqlen] = v
            
            # Get cached keys and values
            k = self.k_cache[:bsz, :start_pos+seqlen]
            v = self.v_cache[:bsz, :start_pos+seqlen]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores + mask.unsqueeze(1)
        
        # Attention weights
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        
        # Compute attention output
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        return self.o_proj(out)

# Model components
class MLP(nn.Module):
    """MLP layer."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = ColumnParallelLinear(dim, inter_dim)
        self.w2 = RowParallelLinear(inter_dim, dim)
        self.w3 = ColumnParallelLinear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Expert(nn.Module):
    """Expert layer."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = Linear(dim, inter_dim)
        self.w2 = Linear(inter_dim, dim)
        self.w3 = Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Gate(nn.Module):
    """Gate layer."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.topk = args.n_activated_experts
        self.n_groups = args.n_expert_groups
        self.topk_groups = args.n_limited_groups
        self.score_func = args.score_func
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices

class MoE(nn.Module):
    """Mixture of experts layer."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)

class Block(nn.Module):
    """Transformer block."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.args = args
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

        # Initialize optional components
        if args.use_cache_augmentation:
            self.cache = DifferentiableCacheAugmentation(
                CacheConfig(
                    hidden_size=args.cache_hidden_size,
                    num_heads=args.cache_num_heads,
                    dropout=args.cache_dropout,
                    max_cache_length=args.cache_max_length
                )
            )
            
        if args.use_neural_memory:
            self.memory = ReasoningMemoryTransformer(
                hidden_size=args.dim,
                memory_size=args.memory_size,
                num_memory_layers=args.num_memory_layers,
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
        h = self.attn_norm(x)
        
        # Apply cache augmentation if enabled
        if self.args.use_cache_augmentation:
            h, _ = self.cache(h, mask)
            
        # Apply neural memory if enabled
        if self.args.use_neural_memory:
            h = self.memory(h, mask)
            
        # Apply tree of thoughts if enabled
        if self.args.use_tree_of_thoughts:
            h = self.tot(h, mask)
        
        # FFN processing
        h = h + self.ffn(self.ffn_norm(h))
        
        return h

class Transformer(nn.Module):
    """Main transformer model."""
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        bsz, seqlen = tokens.size()
        h = self.embed(tokens)
        
        # Compute attention mask
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
            
        # Get rotary embeddings
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        
        # Process through layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            
        # Final normalization and head
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
