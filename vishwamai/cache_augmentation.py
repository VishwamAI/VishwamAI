import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math
import numpy as np
from collections import OrderedDict

class CacheAugmentation(nn.Module):
    """
    Neural cache augmentation module implementing hierarchical caching with hot and cold storage.
    
    This module provides a learnable cache mechanism that stores and retrieves information
    with a two-tier hierarchy (hot and cold cache) for efficient memory usage.
    """
    
    def __init__(
        self,
        cache_size: int,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize the cache augmentation module.
        
        Args:
            cache_size: Total size of cache
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            dropout: Dropout probability
        
        Raises:
            ValueError: If cache_size or hidden_dim are invalid
        """
        super().__init__()
        
        if cache_size < 4:
            raise ValueError("Cache size must be at least 4")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
            
        self.cache_size = cache_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Initialize cache sizes
        self._init_cache_structure()
        
        # Initialize network components
        self._init_networks(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_cache_structure(self):
        """Initialize hierarchical cache structure."""
        # Hierarchical cache storage
        self.hot_cache_size = max(4, self.cache_size // 4)
        self.cold_cache_size = self.cache_size - self.hot_cache_size
        
        # Hot cache for frequently accessed items
        self.hot_cache_keys = nn.Parameter(torch.randn(self.hot_cache_size, self.hidden_dim))
        self.hot_cache_values = nn.Parameter(torch.randn(self.hot_cache_size, self.hidden_dim))
        self.hot_cache_age = nn.Parameter(torch.zeros(self.hot_cache_size))
        self.hot_access_count = nn.Parameter(torch.zeros(self.hot_cache_size))
        
        # Cold cache for less frequently accessed items
        self.cold_cache_keys = nn.Parameter(torch.randn(self.cold_cache_size, self.hidden_dim))
        self.cold_cache_values = nn.Parameter(torch.randn(self.cold_cache_size, self.hidden_dim))
        self.cold_cache_age = nn.Parameter(torch.zeros(self.cold_cache_size))
        self.cold_access_count = nn.Parameter(torch.zeros(self.cold_cache_size))
        
    def _init_networks(self, dropout: float):
        """Initialize neural network components."""
        # Cache compression
        self.compression_ratio = 0.5
        compressed_dim = max(1, int(self.hidden_dim * self.compression_ratio))
        self.compressor = nn.Linear(self.hidden_dim, compressed_dim)
        self.decompressor = nn.Linear(compressed_dim, self.hidden_dim)
        
        # Cache attention components
        self.query_net = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_net = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_net = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_net = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Cache update components
        self.update_gate = nn.Linear(self.hidden_dim * 2, 1)
        self.relevance_score = nn.Linear(self.hidden_dim, 1)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def _init_parameters(self):
        """Initialize model parameters with appropriate initialization schemes."""
        # Initialize cache storage with small random values
        for cache in [self.hot_cache_keys, self.hot_cache_values,
                     self.cold_cache_keys, self.cold_cache_values]:
            nn.init.normal_(cache, mean=0.0, std=0.02)
        
        # Initialize compression layers with Xavier initialization
        nn.init.xavier_uniform_(self.compressor.weight)
        nn.init.zeros_(self.compressor.bias)
        nn.init.xavier_uniform_(self.decompressor.weight)
        nn.init.zeros_(self.decompressor.bias)
        
        # Initialize attention components
        for module in [self.query_net, self.key_net, self.value_net, self.output_net]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
            
        # Initialize update components
        for module in [self.update_gate, self.relevance_score]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        inputs: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Process inputs through cache augmentation.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, hidden_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output tensor, optional attention info dictionary)
            
        Raises:
            RuntimeError: If input tensor has incorrect shape
        """
        if inputs.dim() != 3 or inputs.size(-1) != self.hidden_dim:
            raise RuntimeError(
                f"Expected input shape (batch_size, seq_len, {self.hidden_dim}), "
                f"got {inputs.shape}"
            )
            
        batch_size = inputs.size(0)
        
        try:
            # Process hot and cold cache separately
            hot_output, hot_attention = self._process_cache_level(
                inputs,
                self.hot_cache_keys,
                self.hot_cache_values,
                self.hot_cache_age,
                self.hot_access_count,
                is_hot=True
            )
            
            cold_output, cold_attention = self._process_cache_level(
                inputs,
                self.cold_cache_keys,
                self.cold_cache_values,
                self.cold_cache_age,
                self.cold_access_count,
                is_hot=False
            )
            
            # Combine outputs with weighted sum based on attention scores
            combined_output = self.layer_norm(hot_output + cold_output)
            
            # Update cache states during training
            if self.training:
                self._update_cache_states(hot_attention, cold_attention)
                self._manage_cache_promotion()
            
            # Return combined output and attention info if requested
            if return_attention:
                attention_info = {
                    'hot_attention': hot_attention,
                    'cold_attention': cold_attention
                }
                return combined_output, attention_info
                
            return combined_output, None
            
        except Exception as e:
            raise RuntimeError(f"Error in cache forward pass: {str(e)}") from e

    def _process_cache_level(
        self,
        inputs: torch.Tensor,
        cache_keys: torch.Tensor,
        cache_values: torch.Tensor,
        cache_age: torch.Tensor,
        access_count: torch.Tensor,
        is_hot: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a cache level (hot or cold).
        
        Args:
            inputs: Input tensor
            cache_keys: Cache keys tensor
            cache_values: Cache values tensor
            cache_age: Cache age tracking tensor
            access_count: Access count tracking tensor
            is_hot: Whether this is the hot cache
            
        Returns:
            Tuple of (output tensor, attention weights tensor)
        """
        batch_size = inputs.size(0)
        
        # Transform inputs for attention
        queries = self._split_heads(self.query_net(inputs))
        keys = self._split_heads(self.key_net(cache_keys).unsqueeze(0).expand(batch_size, -1, -1))
        
        # Handle value compression for cold cache
        if not is_hot:
            compressed_values = self._compress_values(cache_values)
            values = self._split_heads(
                self.decompressor(compressed_values).unsqueeze(0).expand(batch_size, -1, -1)
            )
        else:
            values = self._split_heads(
                self.value_net(cache_values).unsqueeze(0).expand(batch_size, -1, -1)
            )
        
        # Calculate attention scores with age and access count adjustments
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply age penalty and access count bonus
        age_penalty = -0.1 * cache_age.unsqueeze(0).unsqueeze(0)
        access_bonus = 0.05 * access_count.unsqueeze(0).unsqueeze(0)
        attention_scores = attention_scores + age_penalty + access_bonus
        
        # Apply attention
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute weighted sum
        attention_output = torch.matmul(attention_weights, values)
        output = self._combine_heads(attention_output)
        output = self.output_net(output)
        
        return output, attention_weights
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Split tensor into attention heads."""
        batch_size = tensor.size(0)
        tensor = tensor.view(batch_size, -1, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)
    
    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Combine attention heads back into full tensor."""
        tensor = tensor.transpose(1, 2)
        batch_size = tensor.size(0)
        return tensor.reshape(batch_size, -1, self.hidden_dim)
    
    def _compress_values(self, values: torch.Tensor) -> torch.Tensor:
        """Compress cache values for storage efficiency."""
        return self.compressor(values)

    def _update_cache_states(
        self,
        hot_attention: torch.Tensor,
        cold_attention: torch.Tensor
    ):
        """Update cache ages and access counts."""
        with torch.no_grad():
            # Update hot cache statistics
            self.hot_cache_age.data += 1
            hot_access = (hot_attention.mean(dim=(0, 1)) > 0.01).float()
            self.hot_cache_age.data *= (1 - hot_access)
            self.hot_access_count.data += hot_access
            
            # Update cold cache statistics
            self.cold_cache_age.data += 1
            cold_access = (cold_attention.mean(dim=(0, 1)) > 0.01).float()
            self.cold_cache_age.data *= (1 - cold_access)
            self.cold_access_count.data += cold_access

    def _manage_cache_promotion(self):
        """Manage promotion/demotion between hot and cold cache."""
        with torch.no_grad():
            try:
                # Calculate cache entry scores
                cold_scores = self.cold_access_count / (self.cold_cache_age + 1)
                promote_mask = cold_scores > torch.quantile(cold_scores, 0.7)
                
                if promote_mask.any():
                    # Find hot cache entries to demote
                    hot_scores = self.hot_access_count / (self.hot_cache_age + 1)
                    _, demote_indices = torch.topk(
                        hot_scores,
                        k=min(promote_mask.sum().item(), self.hot_cache_size),
                        largest=False
                    )
                    
                    # Swap entries
                    self._swap_cache_entries(
                        promote_mask,
                        demote_indices,
                        self.cold_cache_keys,
                        self.cold_cache_values,
                        self.hot_cache_keys,
                        self.hot_cache_values
                    )
                    
                    # Reset counters for swapped entries
                    self.hot_cache_age.data[demote_indices] = 0
                    self.hot_access_count.data[demote_indices] = 0
                    cold_indices = torch.where(promote_mask)[0]
                    self.cold_cache_age.data[cold_indices] = 0
                    self.cold_access_count.data[cold_indices] = 0
                    
            except Exception as e:
                print(f"Warning: Cache promotion failed: {str(e)}")

    def _swap_cache_entries(
        self,
        promote_mask: torch.Tensor,
        demote_indices: torch.Tensor,
        src_keys: torch.Tensor,
        src_values: torch.Tensor,
        dst_keys: torch.Tensor,
        dst_values: torch.Tensor
    ):
        """Swap entries between hot and cold cache."""
        promote_indices = torch.where(promote_mask)[0]
        
        # Save hot cache entries being demoted
        temp_keys = dst_keys[demote_indices].clone()
        temp_values = dst_values[demote_indices].clone()
        
        # Promote cold cache entries
        dst_keys.data[demote_indices] = src_keys[promote_indices]
        dst_values.data[demote_indices] = src_values[promote_indices]
        
        # Demote hot cache entries
        src_keys.data[promote_indices] = temp_keys
        src_values.data[promote_indices] = temp_values

    def update_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        relevance_threshold: float = 0.5,
        target_cache: str = 'hot'
    ):
        """
        Update cache entries with new key-value pairs.
        
        Args:
            keys: New keys to add
            values: New values to add
            relevance_threshold: Minimum relevance score for updates
            target_cache: Which cache to update ('hot' or 'cold')
        """
        if target_cache not in ['hot', 'cold']:
            raise ValueError("target_cache must be 'hot' or 'cold'")
            
        with torch.no_grad():
            try:
                # Calculate relevance scores for new entries
                relevance = torch.sigmoid(self.relevance_score(values)).squeeze(-1)
                
                # Select target cache components
                if target_cache == 'hot':
                    cache_keys = self.hot_cache_keys
                    cache_values = self.hot_cache_values
                    cache_age = self.hot_cache_age
                    access_count = self.hot_access_count
                    max_entries = self.hot_cache_size
                else:
                    cache_keys = self.cold_cache_keys
                    cache_values = self.cold_cache_values
                    cache_age = self.cold_cache_age
                    access_count = self.cold_access_count
                    max_entries = self.cold_cache_size
                    
                    # Compress values for cold cache
                    values = self._compress_values(values)
                
                # Find entries to update based on age and access patterns
                update_score = cache_age / (access_count + 1)
                _, update_indices = torch.topk(
                    update_score,
                    k=min(keys.size(0), max_entries),
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
                        cache_keys.data[idx] = key
                        cache_values.data[idx] = value
                        cache_age.data[idx] = 0
                        
            except Exception as e:
                print(f"Warning: Cache update failed: {str(e)}")
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5
    ) -> Optional[torch.Tensor]:
        """
        Retrieve most relevant cached information.
        
        Args:
            query: Query tensor
            top_k: Number of top entries to retrieve
            
        Returns:
            Optional retrieved and weighted tensor
        """
        if not 1 <= top_k <= self.cache_size:
            raise ValueError(f"top_k must be between 1 and {self.cache_size}")
            
        with torch.no_grad():
            try:
                # Calculate similarity scores
                query_emb = self.query_net(query)
                
                # Combine hot and cold cache
                all_keys = torch.cat([self.hot_cache_keys, self.cold_cache_keys])
                all_values = torch.cat([
                    self.hot_cache_values,
                    self.decompressor(self._compress_values(self.cold_cache_values))
                ])
                
                similarity = F.cosine_similarity(
                    query_emb.unsqueeze(1),
                    all_keys.unsqueeze(0),
                    dim=-1
                )
                
                # Get top-k relevant entries
                values, indices = torch.topk(
                    similarity,
                    k=min(top_k, self.cache_size)
                )
                
                if torch.max(values) < 0.5:  # Relevance threshold
                    return None
                    
                retrieved = all_values[indices]
                
                # Weight by similarity
                weights = F.softmax(values, dim=-1).unsqueeze(-1)
                weighted_sum = torch.sum(retrieved * weights, dim=1)
                
                return weighted_sum
                
            except Exception as e:
                print(f"Warning: Cache retrieval failed: {str(e)}")
                return None
    
    def reset_cache(self):
        """Reset cache to initial state."""
        with torch.no_grad():
            self._init_parameters()
            self.hot_cache_age.data.zero_()
            self.hot_access_count.data.zero_()
            self.cold_cache_age.data.zero_()
            self.cold_access_count.data.zero_()
    
    def get_cache_stats(self) -> Dict[str, float]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics
        """
        with torch.no_grad():
            hot_stats = {
                'hot_avg_age': torch.mean(self.hot_cache_age).item(),
                'hot_max_age': torch.max(self.hot_cache_age).item(),
                'hot_unused': torch.sum(self.hot_cache_age >= 100).item(),
                'hot_total': self.hot_cache_size
            }
            
            cold_stats = {
                'cold_avg_age': torch.mean(self.cold_cache_age).item(),
                'cold_max_age': torch.max(self.cold_cache_age).item(),
                'cold_unused': torch.sum(self.cold_cache_age >= 100).item(),
                'cold_total': self.cold_cache_size
            }
            
            memory_stats = {
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'compressed_ratio': self.compression_ratio
            }
            
            return {**hot_stats, **cold_stats, **memory_stats}
    
    def state_dict(self, *args, **kwargs) -> Dict:
        """
        Save cache state.
        
        Returns:
            Dictionary containing the module state
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({
            'cache_size': self.cache_size,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'hot_cache_size': self.hot_cache_size,
            'cold_cache_size': self.cold_cache_size,
            'compression_ratio': self.compression_ratio
        })
        return state_dict
    
    def load_state_dict(self, state_dict: Dict):
        """
        Load cache state.
        
        Args:
            state_dict: Dictionary containing module state
            
        Raises:
            ValueError: If state dict contains incompatible parameters
        """
        # Verify compatibility
        for key in ['cache_size', 'hidden_dim', 'num_heads']:
            if state_dict[key] != getattr(self, key):
                raise ValueError(
                    f"Incompatible {key}: expected {getattr(self, key)}, "
                    f"got {state_dict[key]}"
                )
        
        # Remove config items from state dict
        for key in ['cache_size', 'hidden_dim', 'num_heads',
                   'hot_cache_size', 'cold_cache_size', 'compression_ratio']:
            state_dict.pop(key, None)
            
        super().load_state_dict(state_dict)
