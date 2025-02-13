import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class MemoryGatingUnit(nn.Module):
    """
    Gating unit that controls information flow into and out of the memory module.
    """
    def __init__(self, hidden_size: int, memory_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        
        # Write gate
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_size + memory_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, memory_size),
            nn.Sigmoid()
        )
        
        # Read gate
        self.read_gate = nn.Sequential(
            nn.Linear(hidden_size + memory_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, memory_size),
            nn.Sigmoid()
        )
        
        # Forget gate
        self.forget_gate = nn.Sequential(
            nn.Linear(memory_size, memory_size),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Combine input and memory for gate computation
        combined = torch.cat([x, memory], dim=-1)
        
        # Compute gates
        write = self.write_gate(combined)
        read = self.read_gate(combined)
        forget = self.forget_gate(memory)
        
        # Update memory
        new_memory = forget * memory + write * x
        
        # Read from memory
        output = read * new_memory
        
        return output, new_memory

class NeuralMemoryModule(nn.Module):
    """
    Neural long-term memory module that can learn to store and retrieve information over long sequences.
    """
    def __init__(self, 
                 hidden_size: int,
                 memory_size: int,
                 num_memory_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_memory_layers = num_memory_layers
        
        # Input projection
        self.input_proj = nn.Linear(hidden_size, memory_size)
        self.input_norm = nn.LayerNorm(memory_size)
        
        # Memory layers
        self.memory_layers = nn.ModuleList([
            MemoryGatingUnit(hidden_size, memory_size)
            for _ in range(num_memory_layers)
        ])
        
        # Memory attention
        self.memory_attention = nn.MultiheadAttention(
            memory_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(memory_size, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize memory states
        self.register_buffer("memory_states", 
                           torch.zeros(num_memory_layers, 1, 1, memory_size))
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process hidden states through the memory module.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            Processed hidden states with long-term memory information
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project input to memory space
        memory = self.input_proj(hidden_states)
        memory = self.input_norm(memory)
        
        # Expand memory states if needed
        if self.memory_states.size(1) != batch_size:
            self.memory_states = self.memory_states.expand(-1, batch_size, -1, -1)
        
        # Process through memory layers
        layer_outputs = []
        current_memory = memory
        
        for i, layer in enumerate(self.memory_layers):
            # Process through gating unit
            output, new_memory = layer(current_memory, self.memory_states[i])
            
            # Update memory state
            if self.training:
                self.memory_states[i] = new_memory.detach()
            
            # Apply memory attention
            attended_memory, _ = self.memory_attention(
                output,
                output,
                output,
                key_padding_mask=attention_mask,
                need_weights=False
            )
            
            current_memory = self.dropout(attended_memory) + output
            layer_outputs.append(current_memory)
        
        # Combine layer outputs
        combined_memory = torch.stack(layer_outputs).mean(0)
        
        # Project back to hidden space
        output = self.output_proj(combined_memory)
        output = self.output_norm(output)
        
        return output
    
    def reset_memory(self):
        """Reset memory states"""
        self.memory_states.zero_()

class ReasoningMemoryTransformer(nn.Module):
    """
    Wrapper that combines the neural memory module with standard transformer processing.
    """
    def __init__(self,
                 hidden_size: int,
                 memory_size: int,
                 num_memory_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        
        self.memory = NeuralMemoryModule(
            hidden_size=hidden_size,
            memory_size=memory_size,
            num_memory_layers=num_memory_layers,
            dropout=dropout
        )
        
        # Memory integration layers
        self.pre_memory_norm = nn.LayerNorm(hidden_size)
        self.post_memory_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process hidden states through memory augmented reasoning.
        
        Args:
            hidden_states: Input tensor from transformer
            attention_mask: Optional attention mask
            
        Returns:
            Memory-augmented hidden states
        """
        # Pre-memory normalization
        normed_states = self.pre_memory_norm(hidden_states)
        
        # Process through memory module
        memory_states = self.memory(normed_states, attention_mask)
        
        # Combine with input and normalize
        output = hidden_states + memory_states
        output = self.post_memory_norm(output)
        
        return output
    
    def reset_memory(self):
        """Reset all memory states"""
        self.memory.reset_memory()
