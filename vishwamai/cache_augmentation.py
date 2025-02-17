import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math
import numpy as np
from collections import OrderedDict

class CacheAugmentation(nn.Module):
    def __init__(
        self,
        cache_size: int,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.cache_size = cache_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Cache storage
        self.cache_keys = nn.Parameter(torch.randn(cache_size, hidden_dim))
        self.cache_values = nn.Parameter(torch.randn(cache_size, hidden_dim))
        self.cache_age = nn.Parameter(torch.zeros(cache_size))
        
        # Cache attention components
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        self.value_net = nn.Linear(hidden_dim, hidden_dim)
        self.output_net = nn.Linear(hidden_dim, hidden_dim)
        
        # Cache update components
        self.update_gate = nn.Linear(hidden_dim * 2, 1)
        self.relevance_score = nn.Linear(hidden_dim, 1)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize cache parameters"""
        # Initialize cache storage
        nn.init.normal_(self.cache_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.cache_values, mean=0.0, std=0.02)
        
        # Initialize attention components
        for module in [self.query_net, self.key_net, self.value_net, self.output_net]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
            
        # Initialize update components
        nn.init.normal_(self.update_gate.weight, std=0.02)
        nn.init.zeros_(self.update_gate.bias)
        nn.init.normal_(self.relevance_score.weight, std=0.02)
        nn.init.zeros_(self.relevance_score.bias)
    
    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Process inputs through cache"""
        batch_size = inputs.size(0)
        
        # Transform inputs for attention
        queries = self._split_heads(self.query_net(inputs))
        cache_k = self._split_heads(self.cache_keys.unsqueeze(0).expand(batch_size, -1, -1))
        cache_v = self._split_heads(self.cache_values.unsqueeze(0).expand(batch_size, -1, -1))
        
        # Calculate attention scores
        attention_scores = torch.matmul(queries, cache_k.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply age penalty to attention scores
        age_penalty = -0.1 * self.cache_age.unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores + age_penalty
        
        # Get attention weights and weighted sum
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, cache_v)
        context = self._combine_heads(context)
        
        # Transform and combine with input
        output = self.layer_norm(inputs + self.output_net(context))
        
        # Update cache age
        with torch.no_grad():
            self.cache_age.data += 1
            
            # Reset age for accessed entries
            access_mask = (attention_weights.mean(dim=(0, 1)) > 0.01).float()
            self.cache_age.data *= (1 - access_mask)
        
        if return_attention:
            return output, attention_weights.detach()
        return output
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Split tensor into attention heads"""
        batch_size = tensor.size(0)
        tensor = tensor.view(batch_size, -1, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)
    
    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Combine attention heads"""
        tensor = tensor.transpose(1, 2)
        batch_size = tensor.size(0)
        return tensor.reshape(batch_size, -1, self.hidden_dim)
    
    def update_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        relevance_threshold: float = 0.5
    ):
        """Update cache entries"""
        with torch.no_grad():
            # Calculate relevance scores for new entries
            relevance = torch.sigmoid(self.relevance_score(values)).squeeze(-1)
            
            # Find cache entries to update (oldest first)
            _, update_indices = torch.topk(
                self.cache_age,
                k=min(keys.size(0), self.cache_size),
                largest=True
            )
            
            # Update selected cache entries
            for idx, key, value, rel in zip(
                update_indices,
                keys,
                values,
                relevance
            ):
                if rel > relevance_threshold:
                    self.cache_keys.data[idx] = key
                    self.cache_values.data[idx] = value
                    self.cache_age.data[idx] = 0
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> Optional[torch.Tensor]:
        """Retrieve most relevant cached information"""
        with torch.no_grad():
            # Calculate similarity scores
            query_emb = self.query_net(query)
            similarity = F.cosine_similarity(
                query_emb.unsqueeze(1),
                self.cache_keys.unsqueeze(0),
                dim=-1
            )
            
            # Get top-k relevant entries
            values, indices = torch.topk(similarity, k=min(top_k, self.cache_size))
            
            if torch.max(values) < 0.5:  # Relevance threshold
                return None
                
            retrieved = self.cache_values[indices]
            
            # Weight by similarity
            weights = F.softmax(values, dim=-1).unsqueeze(-1)
            weighted_sum = torch.sum(retrieved * weights, dim=1)
            
            return weighted_sum
    
    def reset_cache(self):
        """Reset cache to initial state"""
        nn.init.normal_(self.cache_keys, mean=0.0, std=0.02)
        nn.init.normal_(self.cache_values, mean=0.0, std=0.02)
        nn.init.zeros_(self.cache_age)
    
    def get_cache_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        with torch.no_grad():
            avg_age = torch.mean(self.cache_age).item()
            max_age = torch.max(self.cache_age).item()
            unused = torch.sum(self.cache_age >= 100).item()  # Long unused entries
            
            return {
                'average_age': avg_age,
                'max_age': max_age,
                'unused_entries': unused,
                'total_entries': self.cache_size
            }
    
    def state_dict(self, *args, **kwargs):
        """Save cache state"""
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({
            'cache_size': self.cache_size,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads
        })
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load cache state"""
        cache_size = state_dict.pop('cache_size')
        hidden_dim = state_dict.pop('hidden_dim')
        num_heads = state_dict.pop('num_heads')
        
        assert cache_size == self.cache_size
        assert hidden_dim == self.hidden_dim
        assert num_heads == self.num_heads
        
        super().load_state_dict(state_dict)
