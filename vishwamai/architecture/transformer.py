"""
Transformer Architecture Implementation
====================================
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from .attention import MultiHeadAttention
from .mlp import MLP

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim=config.dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        self.mlp = MLP(
            dim=config.dim,
            hidden_dim=4 * config.dim,
            dropout=config.dropout
        )
        self.ln1 = nn.LayerNorm(config.dim)
        self.ln2 = nn.LayerNorm(config.dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attention(self.ln1(x), mask=mask)
        out = h + self.mlp(self.ln2(h))
        return out

class VishwamaiModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.tok_embeddings = nn.Embedding(
            config.vocab_size, 
            config.dim
        )
        self.pos_embeddings = nn.Parameter(torch.zeros(1, config.max_seq_length, config.dim))
        
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.depth)
        ])
        self.ln_f = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = tokens.shape
        
        # Get embeddings
        h = self.tok_embeddings(tokens)
        h = h + self.pos_embeddings[:, :seq_length, :]
        h = self.drop(h)
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h, mask)
            
        h = self.ln_f(h)
        logits = self.head(h)
        
        return logits

    def generate(
        self,
        tokens: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = False
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            for _ in range(max_length):
                # Get predictions
                logits = self(tokens)
                next_token_logits = logits[:, -1, :] / temperature
                
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                tokens = torch.cat([tokens, next_token], dim=-1)
                
                # Check for end of text token
                if next_token.item() == self.config.eos_token_id:
                    break
                    
        return tokens
