"""
Neural Memory implementation for enhanced context retention.

This module provides a neural memory system that can be used to store and
retrieve information over long sequences, enabling better handling of
long-range dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math

from vishwamai.models.base_layers import Linear
from vishwamai.utils.config import ModelConfig

class MemoryBlock(nn.Module):
    """Individual memory block with key-value storage."""
    
    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        num_heads: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_heads = num_heads
        
        # Memory state
        self.register_buffer(
            'memory_keys',
            torch.zeros(num_heads, memory_size, hidden_size)
        )
        self.register_buffer(
            'memory_values',
            torch.zeros(num_heads, memory_size, hidden_size)
        )
        self.register_buffer(
            'memory_age',
            torch.zeros(num_heads, memory_size)
        )
        
        # Projections
        self.query_proj = Linear(hidden_size, hidden_size)
        self.key_proj = Linear(hidden_size, hidden_size)
        self.value_proj = Linear(hidden_size, hidden_size)
        self.output_proj = Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process input through memory block.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, memory access statistics)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project queries, keys and values
        queries = self.query_proj(hidden_states)
        keys = self.key_proj(hidden_states)
        values = self.value_proj(hidden_states)
        
        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, -1)
        keys = keys.view(batch_size, seq_len, self.num_heads, -1)
        values = values.view(batch_size, seq_len, self.num_heads, -1)
        
        # Compute attention scores with memory
        memory_scores = torch.einsum(
            'bshd,hmd->bshm',
            queries,
            self.memory_keys
        )
        
        # Scale scores
        memory_scores = memory_scores / math.sqrt(self.hidden_size)
        
        if attention_mask is not None:
            memory_scores = memory_scores.masked_fill(
                attention_mask.unsqueeze(2).unsqueeze(3),
                float('-inf')
            )
            
        # Get attention weights
        memory_probs = F.softmax(memory_scores, dim=-1)
        memory_probs = self.dropout(memory_probs)
        
        # Read from memory
        memory_output = torch.einsum(
            'bshm,hmd->bshd',
            memory_probs,
            self.memory_values
        )
        
        # Update memory
        self._update_memory(keys, values, memory_probs)
        
        # Combine and project output
        output = memory_output.view(batch_size, seq_len, -1)
        output = self.output_proj(output)
        
        # Compute access statistics
        stats = {
            'access_counts': torch.sum(memory_probs > 0.01, dim=(0, 1)),
            'max_scores': torch.max(memory_scores, dim=-1)[0].mean(),
            'memory_usage': torch.mean((self.memory_age > 0).float())
        }
        
        return output, stats
        
    def _update_memory(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        access_weights: torch.Tensor
    ):
        """Update memory contents based on access patterns."""
        # Increment age of all memories
        self.memory_age += 1
        
        # Compute update importance for each memory slot
        importance = access_weights.sum(dim=(0, 1))  # [num_heads, memory_size]
        
        # Find least recently used memories
        _, update_indices = torch.topk(
            self.memory_age + (1 - importance),
            k=keys.size(1),
            dim=-1,
            largest=True
        )
        
        # Update selected memories
        for head in range(self.num_heads):
            for idx in update_indices[head]:
                self.memory_keys[head, idx] = keys[0, -1, head]
                self.memory_values[head, idx] = values[0, -1, head]
                self.memory_age[head, idx] = 0

class NeuralMemory(nn.Module):
    """
    Neural memory system with multiple memory blocks.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        num_blocks: int = 4
    ):
        super().__init__()
        self.config = config
        self.num_blocks = num_blocks
        
        # Create memory blocks
        self.blocks = nn.ModuleList([
            MemoryBlock(
                hidden_size=config.hidden_size,
                memory_size=config.memory_size,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Output projection
        self.output_proj = Linear(
            config.hidden_size * num_blocks,
            config.hidden_size
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process input through all memory blocks.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Dict containing:
                output: Processed tensor
                stats: Memory access statistics
        """
        block_outputs = []
        block_stats = []
        
        for block in self.blocks:
            output, stats = block(hidden_states, attention_mask)
            block_outputs.append(output)
            block_stats.append(stats)
            
        # Combine outputs
        combined = torch.cat(block_outputs, dim=-1)
        output = self.output_proj(combined)
        
        # Aggregate statistics
        avg_stats = {
            key: torch.mean(torch.stack([
                stats[key] for stats in block_stats
            ]))
            for key in block_stats[0].keys()
        }
        
        return {
            'output': output,
            'stats': avg_stats
        }
