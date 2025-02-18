import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional

from .config import ModelArgs
from .base_layers import Linear
from .utils import precompute_freqs_cis
from .parallel import ColumnParallelLinear, RMSNorm, ParallelEmbedding
from .MLA import MLA
from .MLP import MLP
from .MoE import MoE

# Default values for distributed training
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0

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
    def __init__(self, args: ModelArgs, device: Optional[torch.device] = None):
        super().__init__()  # Call super first to ensure proper initialization
        
        # Validate args type
        if not isinstance(args, ModelArgs):
            raise TypeError("args must be an instance of ModelArgs")
            
        # Initialize distributed settings
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Set dtype
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" and hasattr(torch, 'float8_e4m3fn') else torch.bfloat16
        
        # Store configuration as instance variables
        self.device = device
        self.max_seq_len = args.max_seq_len
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.vocab_size = args.vocab_size
        self.gradient_checkpointing = args.gradient_checkpointing
        
        # Initialize embeddings
        dtype = torch.get_default_dtype()
        self.embed = ParallelEmbedding(
            vocab_size=self.vocab_size,
            dim=self.dim,
            device=device,
            dtype=dtype
        )
        
        # Initialize transformer blocks
        self.layers = nn.ModuleList([
            Block(layer_id, args) for layer_id in range(args.n_layers)
        ])
        
        # Initialize normalization and output projection
        self.norm = RMSNorm(self.dim)
        self.head = ColumnParallelLinear(
            in_features=self.dim,
            out_features=self.vocab_size,
            bias=True,
            device=device,
            dtype=dtype
        )
        
        # Compute positional embeddings
        try:
            print(f"Computing frequencies with dim={self.dim}, max_seq_len={self.max_seq_len}")
            freqs_cis = precompute_freqs_cis(dim=self.dim, end=self.max_seq_len)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
            print("Successfully computed frequencies")
        except Exception as e:
            print(f"Error computing frequencies: dim={self.dim}, max_seq_len={self.max_seq_len}")
            raise e
        
        # Initialize alibi slopes if enabled
        if args.use_alibi:
            slopes = torch.arange(1 - self.n_heads, 1, 2, dtype=torch.float32)
            self.register_buffer("alibi_slopes", -torch.abs(slopes), persistent=False)
        
        # Move to device if specified
        if device is not None:
            self.to(device)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
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
