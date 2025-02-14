import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryConfig:
    hidden_size: int = 8192
    memory_size: int = 2048
    num_memory_layers: int = 3
    num_attention_heads: int = 32
    intermediate_size: int = 16384
    dropout: float = 0.1

class ReasoningMemoryTransformer(nn.Module):
    """Neural memory implementation using transformer architecture for structured reasoning."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        super().__init__()
        self.config = config or MemoryConfig()
        
        # Memory embeddings
        self.memory_embeddings = nn.Parameter(
            torch.randn(self.config.memory_size, self.config.hidden_size)
        )
        
        # Multi-head attention layers
        self.memory_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.dropout,
                batch_first=True
            )
            for _ in range(self.config.num_memory_layers)
        ])
        
        # Layer norms and feedforward networks
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.config.hidden_size)
            for _ in range(self.config.num_memory_layers * 2)
        ])
        
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.intermediate_size),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.intermediate_size, self.config.hidden_size),
                nn.Dropout(self.config.dropout)
            )
            for _ in range(self.config.num_memory_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(
            self.config.hidden_size, self.config.hidden_size, bias=False
        )
        
        self.dropout = nn.Dropout(self.config.dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process hidden states through memory-augmented layers."""
        batch_size = hidden_states.size(0)
        
        # Expand memory embeddings for batch
        memory = self.memory_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Process through memory layers
        for i in range(self.config.num_memory_layers):
            # Self attention
            normed_memory = self.layer_norms[i*2](memory)
            attn_output, _ = self.memory_layers[i](
                query=normed_memory,
                key=normed_memory,
                value=normed_memory
            )
            memory = memory + self.dropout(attn_output)
            
            # Feedforward
            normed_memory = self.layer_norms[i*2 + 1](memory)
            ff_output = self.ffns[i](normed_memory)
            memory = memory + ff_output
            
        # Cross attention with input hidden states
        output, _ = self.memory_layers[-1](
            query=hidden_states,
            key=memory,
            value=memory
        )
        
        # Project outputs
        output = self.output_projection(output)
        
        return output
    
    def save_pretrained(self, save_path: str):
        """Save memory components."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, f"{save_path}/neural_memory.pt")
        
    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load memory components."""
        checkpoint = torch.load(f"{load_path}/neural_memory.pt")
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
