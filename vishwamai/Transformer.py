import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, Tuple
import warnings
    
from .config import ModelArgs
from .base_layers import Linear
from .utils import precompute_freqs_cis
from .parallel import ColumnParallelLinear, RMSNorm, ParallelEmbedding
from .MLA import MLA
from .MLP import MLP

class Block(nn.Module):
    """Transformer block combining attention and feed-forward layers."""
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim)
        
        # Initialize normalization layers
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Process input
        x_dtype = next(self.parameters()).dtype
        x = x.to(dtype=x_dtype)
        
        # Attention path
        normed = self.attn_norm(x)
        attn_out = self.attn(normed, start_pos, freqs_cis, mask)
        x = x + attn_out.to(x_dtype)
        
        # FFN path
        normed = self.ffn_norm(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out.to(x_dtype)
        
        return x

class Transformer(nn.Module):
    """Transformer model with integrated FP8/BF16 support."""
    
    def __init__(self, args: ModelArgs, device: Optional[torch.device] = None):
        super().__init__()
        
        if not isinstance(args, ModelArgs):
            raise TypeError("args must be an instance of ModelArgs")
        
        # Store configuration
        self.args = args
        self.device = device
        
        # Initialize dtype based on args and hardware support
        self.dtype = self._get_optimal_dtype()
        print(f"Using dtype: {self.dtype}")  # Debug info
        Linear.dtype = self.dtype  # Set global dtype for Linear layers
        
        # Initialize embeddings with proper dtype
        self.embed = ParallelEmbedding(
            vocab_size=self.args.vocab_size,
            dim=self.args.dim,
            device=self.device,
            dtype=self.dtype
        )
        
        # Initialize transformer blocks
        self.layers = nn.ModuleList([
            Block(layer_id, args) for layer_id in range(args.n_layers)
        ])
        
        # Initialize output layers
        self.norm = RMSNorm(args.dim)
        self.head = ColumnParallelLinear(
            in_features=self.args.dim,
            out_features=self.args.vocab_size,
            bias=True,
            device=self.device,
            dtype=self.dtype
        )
        
        # Initialize positional embeddings
        self._init_positional_embeddings()
        
        # Move to device if specified and set dtype
        if device is not None:
            self.to(device)
        self.to(self.dtype)
            
    def _get_optimal_dtype(self) -> torch.dtype:
        """Determine optimal dtype based on hardware and configuration."""
        # Default to bf16 per ModelArgs
        if hasattr(self.args, 'dtype') and self.args.dtype == "bf16":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            warnings.warn("BF16 requested but not supported, falling back to FP32")
            return torch.float32
            
        # Automatic selection based on hardware support
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif torch.cuda.is_available():
            return torch.float16
        return torch.float32
        
    def _init_positional_embeddings(self):
        """Initialize positional embeddings."""
        try:
            print(f"Computing frequencies with dim={self.args.dim}, max_seq_len={self.args.max_seq_len}")
            # Call with positional arguments only
            freqs_cis = precompute_freqs_cis(self.args.dim, self.args.max_seq_len)
            # Convert to model's dtype and device after creation
            freqs_cis = freqs_cis.to(device=self.device, dtype=self.dtype)
            self.register_buffer("freqs_cis", freqs_cis, persistent=False)
            print("Successfully computed frequencies")
            
            if self.args.use_alibi:
                slopes = torch.arange(1 - self.args.n_heads, 1, 2, dtype=self.dtype)
                self.register_buffer("alibi_slopes", -torch.abs(slopes), persistent=False)
                
        except Exception as e:
            print(f"Error computing frequencies: {str(e)}")
            raise

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, tokens, start_pos, use_reentrant=False
            )
        return self._forward_impl(tokens, start_pos)

    def _forward_impl(self, tokens: torch.Tensor, start_pos: int = 0):
        seqlen = tokens.size(1)
        tokens = tokens.to(dtype=torch.long)  # Input tokens should be integers
        h = self.embed(tokens)
        h = h.to(dtype=self.dtype)  # Ensure embeddings are in correct dtype
        
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        
        # Process through layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
            h = h.to(dtype=self.dtype)  # Maintain dtype consistency
        
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        
        # Handle distributed case
        if self.world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        
        return logits
