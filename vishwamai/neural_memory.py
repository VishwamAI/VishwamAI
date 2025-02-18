import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .config import ModelArgs

class NeuralMemory(nn.Module):
    """Neural memory module for long-term information storage."""
    
    def __init__(
        self,
        args: ModelArgs,
        memory_size: int = 512,
        num_memory_heads: int = 4
    ):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.memory_size = memory_size
        self.num_memory_heads = num_memory_heads
        
        # Initialize memory components
        self.memory = nn.Parameter(torch.randn(memory_size, self.dim))
        self.query_net = nn.Linear(self.dim, num_memory_heads * self.dim)
        self.key_net = nn.Linear(self.dim, num_memory_heads * self.dim)
        self.value_net = nn.Linear(self.dim, num_memory_heads * self.dim)
        self.output_net = nn.Linear(num_memory_heads * self.dim, self.dim)
        
        # Layer normalization for memory access
        self.norm = nn.LayerNorm(self.dim)
        
        # Initialize memory buffer
        self.register_buffer('usage_tracker', torch.zeros(memory_size))
        
        # Temperature parameter for attention
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Optional expert layers for memory augmentation
        if hasattr(args, 'n_routed_experts') and args.n_routed_experts > 0:
            self.memory_experts = nn.ModuleList([
                nn.Linear(self.dim, self.dim)
                for _ in range(args.n_routed_experts)
            ])
        else:
            self.memory_experts = None

    def access_memory(self, query: torch.Tensor) -> torch.Tensor:
        """Access memory using attention mechanism."""
        batch_size = query.size(0)
        
        # Project query, keys, and values
        q = self.query_net(query).view(batch_size, -1, self.num_memory_heads, self.dim)
        k = self.key_net(self.memory).view(-1, self.num_memory_heads, self.dim)
        v = self.value_net(self.memory).view(-1, self.num_memory_heads, self.dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        scores = scores * self.temperature
        
        # Apply attention
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # Update usage statistics
        with torch.no_grad():
            self.usage_tracker += attn.sum(dim=(0,1))
        
        # Process with experts if available
        if self.memory_experts is not None:
            expert_outputs = []
            for expert in self.memory_experts:
                expert_outputs.append(expert(out))
            out = torch.stack(expert_outputs).mean(0)
        
        # Project to output dimension
        return self.output_net(out.view(batch_size, -1))

    def update_memory(self, new_memories: torch.Tensor):
        """Update memory contents."""
        # Find least used memory locations
        _, indices = torch.topk(self.usage_tracker, 
                              k=min(new_memories.size(0), self.memory_size),
                              largest=False)
        
        # Update memory at those locations
        self.memory.data[indices] = self.norm(new_memories[:len(indices)])
        
        # Reset usage for updated locations
        self.usage_tracker[indices] = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through memory module."""
        # Access memory
        memory_output = self.access_memory(x)
        
        # Update memory if training
        if self.training:
            self.update_memory(x)
        
        # Combine with input using residual connection
        return x + memory_output

    def reset_memory(self):
        """Reset memory contents and usage statistics."""
        nn.init.normal_(self.memory)
        self.usage_tracker.zero_()

    @property
    def device(self) -> torch.device:
        """Get the device memory is stored on."""
        return self.memory.device
