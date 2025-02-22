"""
Feed-forward neural network implementation for Vishwamai model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Feed-forward neural network with GELU activation
    """
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Linear transformations
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward transformation to hidden states
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Project to intermediate size
        intermediate = self.dense_h_to_4h(hidden_states)
        
        # Apply GELU activation
        intermediate = F.gelu(intermediate)
        
        # Project back to hidden size and apply dropout
        output = self.dense_4h_to_h(intermediate)
        output = self.dropout(output)
        
        return output
