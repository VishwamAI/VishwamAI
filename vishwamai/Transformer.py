import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, Tuple
import warnings

try:
    import transformer_engine as te
    TRANSFORMER_ENGINE_AVAILABLE = True
    TransformerEngine = te
except ImportError:
    TRANSFORMER_ENGINE_AVAILABLE = False
    TransformerEngine = None
from .config import ModelArgs
from .base_layers import Linear
from .utils import precompute_freqs_cis
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
        
        # Initialize normalization layers
        self.attn_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)
        
        # Apply TransformerEngine conversion if available and enabled
        if (TRANSFORMER_ENGINE_AVAILABLE and 
            getattr(args, 'use_transformer_engine', True) and 
            hasattr(TransformerEngine, 'convert_module')):
            self.attn_norm = TransformerEngine.convert_module(self.attn_norm)
            self.ffn_norm = TransformerEngine.convert_module(self.ffn_norm)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Ensure consistent dtype throughout block
        x = x.to(dtype=next(self.parameters()).dtype)
        attn_out = self.attn(self.attn_norm(x), start_pos, freqs_cis, mask)
        x = x + attn_out.to(x.dtype)
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + ffn_out.to(x.dtype)
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
        self.use_te = (TRANSFORMER_ENGINE_AVAILABLE and 
                      getattr(args, 'use_transformer_engine', True) and 
                      hasattr(TransformerEngine, 'convert_module'))
        
        # Set up distributed training
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Initialize dtype based on args and hardware support
        self.dtype = self._get_optimal_dtype()
        Linear.dtype = self.dtype  # Set global dtype for Linear layers
        
        # Initialize embeddings with proper dtype
        self.embed = self._create_embeddings()
        
        # Initialize transformer blocks
        self.layers = nn.ModuleList([
            Block(layer_id, args) for layer_id in range(args.n_layers)
        ])
        
        # Initialize output layers
        self.norm = RMSNorm(args.dim)
        if self.use_te:
            self.norm = TransformerEngine.convert_module(self.norm)
        
        self.head = self._create_output_head()
        
        # Initialize positional embeddings
        self._init_positional_embeddings()
        
        # Move to device if specified
        if device is not None:
            self.to(device)
            
    def _get_optimal_dtype(self) -> torch.dtype:
        """Determine optimal dtype based on hardware and configuration."""
        if hasattr(self.args, 'dtype'):
            dtype_str = self.args.dtype
            if dtype_str == "bf16" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            elif dtype_str == "fp8" and self.use_te and hasattr(torch, 'float8_e4m3fn'):
                return torch.float8_e4m3fn
            # If requested dtype is not supported, warn and fallback
            warnings.warn(f"Requested dtype {dtype_str} not supported, falling back to automatic selection")
            
        # Automatic selection based on hardware support
        if self.use_te and hasattr(torch, 'float8_e4m3fn'):
            return torch.float8_e4m3fn
        elif torch.cuda.is_bf16_supported():
            return torch.bfloat16
        elif torch.cuda.is_available():
            return torch.float16
        return torch.float32
            
    def _create_embeddings(self) -> nn.Module:
        """Create embedding layer with proper configuration."""
        embed = ParallelEmbedding(
            vocab_size=self.args.vocab_size,
            dim=self.args.dim,
            device=self.device,
            dtype=self.dtype
        )
        if self.use_te:
            embed = TransformerEngine.convert_module(embed)
        return embed
        
    def _create_output_head(self) -> nn.Module:
        """Create output projection with proper configuration."""
        head = ColumnParallelLinear(
            in_features=self.args.dim,
            out_features=self.args.vocab_size,
            bias=True,
            device=self.device,
            dtype=self.dtype
        )
        if self.use_te:
            head = TransformerEngine.convert_module(head)
        return head
        
    def _init_positional_embeddings(self):
        """Initialize positional embeddings."""
        try:
            print(f"Computing frequencies with dim={self.args.dim}, max_seq_len={self.args.max_seq_len}")
            freqs_cis = precompute_freqs_cis(
                dim=self.args.dim,
                end=self.args.max_seq_len,
                dtype=self.dtype
            )
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
        h = h.to(self.dtype)  # Ensure embeddings are in correct dtype
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)
        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)[:, -1]
        logits = self.head(h)
        if self.world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits
