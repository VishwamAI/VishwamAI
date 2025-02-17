import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

from .model import ModelArgs, world_size, rank, Linear, precompute_freqs_cis
from .parallel import ColumnParallelLinear, RMSNorm, ParallelEmbedding
from .MLA import MLA
from .MLP import MLP
from .MoE import MoE

class Block(nn.Module):
    """Transformer block combining attention and feed-forward layers."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    """Transformer model with positional embeddings, multiple layers, and output projection."""
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.gradient_checkpointing = args.gradient_checkpointing
        
        # Add ALiBi slopes if enabled
        if args.use_alibi:
            self.alibi_slopes = self._get_alibi_slopes()

    def _get_alibi_slopes(self):
        """Generate ALiBi attention slopes."""
        slopes = torch.arange(1 - self.n_heads, 1, 2, dtype=torch.float32)
        return -torch.abs(slopes)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        # Enable gradient checkpointing during training
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, tokens, start_pos, use_reentrant=False
            )
        return self._forward_impl(tokens, start_pos)

    def _forward_impl(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits
