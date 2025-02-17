import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

class NeuralMemory(nn.Module):
    def __init__(self, memory_size: int, hidden_dim: int = 768):
        super().__init__()
        
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        
        # Memory components
        self.memory_keys = nn.Parameter(torch.randn(memory_size, hidden_dim))
        self.memory_values = nn.Parameter(torch.randn(memory_size, hidden_dim))
        
        # Memory update components
        self.key_transform = nn.Linear(hidden_dim, hidden_dim)
        self.value_transform = nn.Linear(hidden_dim, hidden_dim)
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # Memory control gates
        self.write_gate = nn.Linear(hidden_dim * 2, 1)
        self.read_gate = nn.Linear(hidden_dim * 2, 1)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize memory parameters"""
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.02)
        
        # Initialize transformation matrices
        for module in [self.key_transform, self.value_transform, self.query_transform]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
            
        # Initialize gates
        for gate in [self.write_gate, self.read_gate]:
            nn.init.normal_(gate.weight, std=0.02)
            nn.init.zeros_(gate.bias)
    
    def forward(
        self,
        inputs: torch.Tensor,
        update_memory: bool = True
    ) -> torch.Tensor:
        """Process inputs through neural memory"""
        
        batch_size = inputs.size(0)
        
        # Transform inputs
        queries = self.query_transform(inputs)
        keys = self.key_transform(inputs)
        values = self.value_transform(inputs)
        
        # Memory attention
        memory_scores = torch.matmul(queries, self.memory_keys.T)
        memory_scores = memory_scores / math.sqrt(self.hidden_dim)
        memory_attention = F.softmax(memory_scores, dim=-1)
        
        # Read from memory
        read_values = torch.matmul(memory_attention, self.memory_values)
        
        # Apply read gate
        read_gate_input = torch.cat([inputs, read_values], dim=-1)
        read_gate = torch.sigmoid(self.read_gate(read_gate_input))
        gated_read = read_values * read_gate
        
        # Update memory if required
        if update_memory and self.training:
            self._update_memory(keys, values, memory_attention)
        
        # Combine with input
        output = self.layer_norm(inputs + gated_read)
        
        return output
    
    def _update_memory(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_weights: torch.Tensor
    ):
        """Update memory with new information"""
        
        # Calculate update strength
        update_gate = torch.sigmoid(self.write_gate(
            torch.cat([keys, values], dim=-1)
        ))
        
        # Calculate memory updates
        key_update = torch.matmul(attention_weights.transpose(1, 0), keys)
        value_update = torch.matmul(attention_weights.transpose(1, 0), values)
        
        # Apply updates
        key_update = key_update * update_gate
        value_update = value_update * update_gate
        
        self.memory_keys.data = self.memory_keys * (1 - update_gate) + key_update
        self.memory_values.data = self.memory_values * (1 - update_gate) + value_update
    
    def reset_memory(self):
        """Reset memory to initial state"""
        nn.init.normal_(self.memory_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.memory_values, mean=0.0, std=0.02)
    
    def get_memory_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return current memory state"""
        return self.memory_keys.detach(), self.memory_values.detach()
    
    def set_memory_state(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ):
        """Set memory state to provided values"""
        assert keys.size() == self.memory_keys.size()
        assert values.size() == self.memory_values.size()
        
        self.memory_keys.data.copy_(keys)
        self.memory_values.data.copy_(values)
    
    def state_dict(self, *args, **kwargs):
        """Save memory state"""
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['memory_size'] = self.memory_size
        state_dict['hidden_dim'] = self.hidden_dim
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load memory state"""
        memory_size = state_dict.pop('memory_size')
        hidden_dim = state_dict.pop('hidden_dim')
        
        assert memory_size == self.memory_size
        assert hidden_dim == self.hidden_dim
        
        super().load_state_dict(state_dict)
