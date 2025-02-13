import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class CacheConfig:
    hidden_size: int = 256  # Size of cache embeddings
    num_heads: int = 4      # Number of attention heads for cache processing
    dropout: float = 0.1    # Dropout rate
    max_cache_length: int = 1024  # Maximum number of cached items
    use_gated_connections: bool = True  # Whether to use gated connections

class DifferentiableCacheAugmentation(nn.Module):
    """
    Differentiable cache augmentation module that processes and enriches the transformer's key-value cache.
    This allows for asynchronous reasoning and better long-term memory utilization.
    """
    def __init__(self, config: CacheConfig):
        super().__init__()
        self.config = config
        
        # Cache embedding layers
        self.cache_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.cache_layernorm = nn.LayerNorm(config.hidden_size)
        
        # Multi-head attention for cache processing
        self.cache_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Gate mechanism for selective updating
        if config.use_gated_connections:
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Sigmoid()
            )
        
        # Cache update network
        self.update_net = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size * 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        
        # Importance scoring network
        self.importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.register_buffer("cache_memory", torch.zeros(1, config.max_cache_length, config.hidden_size))
        self.register_buffer("cache_mask", torch.ones(1, config.max_cache_length, dtype=torch.bool))
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process hidden states through cache augmentation.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, sequence_length, hidden_size)
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (augmented_states, cache_states)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Project input states
        cache_states = self.cache_proj(hidden_states)
        cache_states = self.cache_layernorm(cache_states)
        
        # Process with self-attention
        cache_states, _ = self.cache_attention(
            cache_states, 
            self.cache_memory,
            self.cache_memory,
            key_padding_mask=self.cache_mask,
            need_weights=False
        )
        
        # Update gate
        if self.config.use_gated_connections:
            gate_values = self.gate(torch.cat([hidden_states, cache_states], dim=-1))
            cache_states = gate_values * cache_states
        
        # Generate importance scores for cache updating
        importance_scores = self.importance_net(cache_states)
        
        # Update cache memory based on importance
        if self.training:
            # During training, update cache with most important items
            _, top_indices = importance_scores.squeeze(-1).topk(
                min(seq_len, self.config.max_cache_length),
                dim=1
            )
            new_cache = cache_states.gather(
                1, 
                top_indices.unsqueeze(-1).expand(-1, -1, hidden_size)
            )
            self.cache_memory = new_cache.detach()
            self.cache_mask = torch.zeros_like(self.cache_mask)
            self.cache_mask[:, :new_cache.size(1)] = True
            
        # Generate augmented states
        augmented_states = self.update_net(torch.cat([hidden_states, cache_states], dim=-1))
        
        return augmented_states, cache_states
    
    def reset_cache(self):
        """Reset the cache memory and mask"""
        self.cache_memory.zero_()
        self.cache_mask.fill_(True)
