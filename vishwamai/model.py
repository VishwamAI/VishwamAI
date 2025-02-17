import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Literal

from .config import ModelArgs
from .base_layers import Linear
from .parallel import ColumnParallelLinear, RowParallelLinear, RMSNorm, ParallelEmbedding
from .utils import precompute_freqs_cis
from .shared_constants import world_size, rank

# Global config variables
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

class Block(nn.Module):
    """Transformer block combining attention and feed-forward layers."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        # Import here to avoid circular imports
        from .MLA import MLA
        from .MLP import MLP
        from .MoE import MoE
        
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    """Main transformer model."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        
        self.tok_embeddings = ParallelEmbedding(
            args.vocab_size, args.dim
        )
        
        self.layers = torch.nn.ModuleList()
        for i in range(args.n_layers):
            self.layers.append(Block(i, args))
        
        self.norm = RMSNorm(args.dim)
        self.output = RowParallelLinear(
            args.dim, args.vocab_size, bias=False
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Set parallel processing variables
        global world_size, rank
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Compute attention patterns
        freqs_cis = precompute_freqs_cis(
            self.args.dim // self.args.n_heads, seqlen + start_pos, 
            self.args.rope_theta, device=tokens.device
        )
        
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            
        # Forward through layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            
        h = self.norm(h)
        output = self.output(h)
        
        return output

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(args)
    print(model(x).size())
