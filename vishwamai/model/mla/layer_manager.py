"""Layer manager for Multi-Level Attention."""

from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class MLALayerManager(nn.Module):
    """Manager for controlling MLA layer computation."""
    
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_attention_levels: int,
        reduction_factor: float = 0.5,
        min_layers_per_level: int = 1,
        adaptive_computation: bool = True,
        computation_threshold: float = 0.1,
        layer_temperature: float = 1.0,
        layer_dropout: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize MLA layer manager.
        
        Args:
            num_layers: Total number of layers
            hidden_size: Size of hidden dimension
            num_attention_levels: Number of attention levels
            reduction_factor: Factor for reducing layers at each level
            min_layers_per_level: Minimum layers per attention level
            adaptive_computation: Whether to use adaptive computation
            computation_threshold: Threshold for adaptive computation
            layer_temperature: Temperature for layer importance scores
            layer_dropout: Layer dropout probability
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_attention_levels = num_attention_levels
        self.reduction_factor = reduction_factor
        self.min_layers_per_level = min_layers_per_level
        self.adaptive_computation = adaptive_computation
        self.computation_threshold = computation_threshold
        self.layer_temperature = layer_temperature
        
        # Compute layer distribution across levels
        self.layer_counts = self._compute_layer_distribution()
        
        # Layer importance prediction
        if adaptive_computation:
            self.importance_proj = nn.Linear(hidden_size, 1, **factory_kwargs)
            self.layer_dropout = nn.Dropout(layer_dropout)
            
            # Initialize importance projection
            nn.init.normal_(self.importance_proj.weight, std=0.02)
            nn.init.zeros_(self.importance_proj.bias)
            
    def _compute_layer_distribution(self) -> List[int]:
        """Compute number of layers for each attention level.
        
        Returns:
            List of layer counts for each level
        """
        remaining_layers = self.num_layers
        layer_counts = []
        
        for level in range(self.num_attention_levels):
            # Compute target layer count for this level
            if level == self.num_attention_levels - 1:
                # Use all remaining layers for final level
                layer_count = remaining_layers
            else:
                # Reduce layer count by factor
                layer_count = max(
                    self.min_layers_per_level,
                    int(remaining_layers * self.reduction_factor)
                )
                
            layer_counts.append(layer_count)
            remaining_layers -= layer_count
            
            if remaining_layers < self.min_layers_per_level:
                break
                
        return layer_counts
        
    def get_layer_config(self) -> Dict[str, List[int]]:
        """Get layer configuration for MLA block.
        
        Returns:
            Dictionary containing layer configuration
        """
        return {
            "layer_counts": self.layer_counts,
            "cumulative_layers": [sum(self.layer_counts[:i+1]) for i in range(len(self.layer_counts))]
        }
        
    def compute_importance_scores(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute importance scores for adaptive computation.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Importance scores of shape [batch_size, seq_length]
        """
        if not self.adaptive_computation:
            return None
            
        # Project hidden states to scalar scores
        scores = self.importance_proj(hidden_states).squeeze(-1)  # [B, L]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
            
        # Apply temperature scaling and softmax
        scores = F.softmax(scores / self.layer_temperature, dim=-1)
        
        # Apply dropout during training
        if self.training:
            scores = self.layer_dropout(scores)
            
        return scores
        
    def should_skip_layer(
        self,
        importance_scores: torch.Tensor,
        layer_idx: int,
        attention_level: int
    ) -> bool:
        """Determine whether to skip layer computation.
        
        Args:
            importance_scores: Token importance scores
            layer_idx: Current layer index
            attention_level: Current attention level
            
        Returns:
            Whether to skip layer computation
        """
        if not self.adaptive_computation:
            return False
            
        # Get cumulative layer count up to current level
        prev_layers = sum(self.layer_counts[:attention_level])
        
        # Get relative layer position within level
        level_position = (layer_idx - prev_layers) / self.layer_counts[attention_level]
        
        # Compute mean importance score
        mean_importance = importance_scores.mean().item()
        
        # Skip if importance score is below threshold and not first/last layer of level
        return (
            mean_importance < self.computation_threshold and
            level_position > 0 and
            level_position < 1
        )
        
    def get_level_scale(self, attention_level: int) -> float:
        """Get scaling factor for attention level.
        
        Args:
            attention_level: Attention level index
            
        Returns:
            Scaling factor for level
        """
        # Scale attention by level depth
        level_depth = self.layer_counts[attention_level]
        return 1.0 / math.sqrt(level_depth)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute layer management outputs.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask tensor
            
        Returns:
            Tuple containing:
            - Input tensor (unchanged)
            - Dictionary of auxiliary outputs including importance scores
        """
        # Compute importance scores if using adaptive computation
        importance_scores = None
        if self.adaptive_computation:
            importance_scores = self.compute_importance_scores(hidden_states, attention_mask)
            
        # Collect outputs
        outputs = {
            "importance_scores": importance_scores,
            "layer_config": self.get_layer_config(),
        }
        
        return hidden_states, outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"num_layers={self.num_layers}, "
            f"num_levels={self.num_attention_levels}, "
            f"adaptive={self.adaptive_computation}, "
            f"layer_counts={self.layer_counts}"
        )
