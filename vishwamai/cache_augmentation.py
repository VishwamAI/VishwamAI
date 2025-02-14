import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

@dataclass
class CacheConfig:
    """Configuration for differentiable cache module."""
    hidden_size: int = 8192
    num_heads: int = 8
    max_cache_length: int = 65536
    dropout: float = 0.1
    retrieval_factor: float = 1.0  # Controls cache retrieval strength
    update_freq: int = 100  # Update frequency for cache entries

class DifferentiableCacheAugmentation(nn.Module):
    """Differentiable cache augmentation for enhanced memory retrieval."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        super().__init__()
        self.config = config or CacheConfig()
        
        # Cache storage
        self.cache_keys = nn.Parameter(
            torch.randn(self.config.max_cache_length, self.config.hidden_size)
        )
        self.cache_values = nn.Parameter(
            torch.randn(self.config.max_cache_length, self.config.hidden_size)
        )
        
        # Cache attention mechanism
        self.cache_attention = nn.MultiheadAttention(
            embed_dim=self.config.hidden_size,
            num_heads=self.config.num_heads,
            dropout=self.config.dropout,
            batch_first=True
        )
        
        # Cache update networks
        self.key_update_net = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
        self.value_update_net = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # Cache statistics
        self.access_counts = torch.zeros(self.config.max_cache_length)
        self.update_step = 0
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process hidden states through cache retrieval and update."""
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Move cache to correct device
        self._ensure_cache_on_device(device)
        
        # Retrieve from cache
        enhanced_states = self._retrieve_from_cache(hidden_states)
        
        # Update cache if needed
        if self.training and self.update_step % self.config.update_freq == 0:
            self._update_cache(hidden_states)
            
        self.update_step += 1
        return enhanced_states
    
    def _ensure_cache_on_device(self, device: torch.device):
        """Ensure cache tensors are on the correct device."""
        if self.cache_keys.device != device:
            self.cache_keys.data = self.cache_keys.data.to(device)
            self.cache_values.data = self.cache_values.data.to(device)
            self.access_counts = self.access_counts.to(device)
    
    def _retrieve_from_cache(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Retrieve relevant information from cache using attention."""
        # Compute attention scores
        attn_output, attn_weights = self.cache_attention(
            query=hidden_states,
            key=self.cache_keys.unsqueeze(0).expand(hidden_states.size(0), -1, -1),
            value=self.cache_values.unsqueeze(0).expand(hidden_states.size(0), -1, -1)
        )
        
        # Update access statistics
        if self.training:
            self.access_counts += attn_weights.sum(dim=(0, 1)).detach()
        
        # Combine with input using learnable factor
        enhanced = hidden_states + self.config.retrieval_factor * attn_output
        return enhanced
    
    def _update_cache(self, hidden_states: torch.Tensor):
        """Update cache entries based on current inputs."""
        with torch.no_grad():
            # Find least accessed entries
            _, update_indices = torch.topk(
                self.access_counts, 
                k=min(hidden_states.size(0), self.config.max_cache_length),
                largest=False
            )
            
            # Compute updates
            new_keys = self.key_update_net(
                torch.cat([
                    hidden_states,
                    self.cache_keys[update_indices].repeat(hidden_states.size(0), 1)
                ], dim=-1)
            )
            
            new_values = self.value_update_net(
                torch.cat([
                    hidden_states,
                    self.cache_values[update_indices].repeat(hidden_states.size(0), 1)
                ], dim=-1)
            )
            
            # Update cache
            self.cache_keys.data[update_indices] = new_keys.detach().mean(dim=0)
            self.cache_values.data[update_indices] = new_values.detach().mean(dim=0)
            self.access_counts[update_indices] = 0
    
    def save_pretrained(self, save_path: str):
        """Save cache components."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
            'access_counts': self.access_counts,
            'update_step': self.update_step
        }, f"{save_path}/cache_augmentation.pt")
    
    @classmethod
    def from_pretrained(cls, load_path: str):
        """Load cache components."""
        checkpoint = torch.load(f"{load_path}/cache_augmentation.pt")
        model = cls(config=checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.access_counts = checkpoint['access_counts']
        model.update_step = checkpoint['update_step']
        return model

    def reset_statistics(self):
        """Reset cache statistics."""
        self.access_counts.zero_()
        self.update_step = 0
