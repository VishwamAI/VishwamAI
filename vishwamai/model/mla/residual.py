"""Residual connection handlers for MLA layers."""

from typing import Optional, Tuple, Dict, Union

import torch
import torch.nn as nn

class MLAResidual(nn.Module):
    """Multi-level residual connection handler."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_levels: int,
        residual_scale: Optional[float] = None,
        level_scale_factor: float = 0.5,
        use_layer_scale: bool = True,
        init_scale: float = 0.1,
        dropout_prob: float = 0.1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize MLA residual handler.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_levels: Number of attention levels
            residual_scale: Optional fixed scale for residuals
            level_scale_factor: Scaling factor between attention levels
            use_layer_scale: Whether to use learnable layer scaling
            init_scale: Initial value for layer scale parameters
            dropout_prob: Residual dropout probability
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_levels = num_attention_levels
        self.residual_scale = residual_scale
        self.level_scale_factor = level_scale_factor
        self.use_layer_scale = use_layer_scale
        
        # Layer scaling parameters
        if use_layer_scale:
            self.layer_scale = nn.ParameterList([
                nn.Parameter(
                    torch.full(
                        (hidden_size,),
                        init_scale * (level_scale_factor ** level),
                        **factory_kwargs
                    )
                )
                for level in range(num_attention_levels)
            ])
        else:
            self.layer_scale = None
            
        # Residual dropout
        self.dropout = nn.Dropout(dropout_prob)
        
    def get_level_scale(self, level: int) -> float:
        """Get scaling factor for attention level.
        
        Args:
            level: Attention level index
            
        Returns:
            Scaling factor for level
        """
        if self.residual_scale is not None:
            return self.residual_scale
        else:
            return self.level_scale_factor ** level
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attention_level: int,
        pre_norm: bool = True,
        post_add: bool = False,
    ) -> torch.Tensor:
        """Apply residual connection with appropriate scaling.
        
        Args:
            hidden_states: Main path tensor
            residual: Residual path tensor
            attention_level: Current attention level
            pre_norm: Whether this is a pre-norm residual
            post_add: Whether to apply post-residual addition
            
        Returns:
            Output tensor after residual connection
        """
        if not pre_norm:
            # Apply layer scaling if enabled
            if self.use_layer_scale:
                hidden_states = hidden_states * self.layer_scale[attention_level].unsqueeze(0).unsqueeze(0)
                
            # Apply dropout
            hidden_states = self.dropout(hidden_states)
            
            # Add scaled residual
            output = residual + hidden_states * self.get_level_scale(attention_level)
            
            # Apply post-residual addition if needed
            if post_add:
                output = output + hidden_states
                
        else:
            # For pre-norm, apply operations in different order
            if self.use_layer_scale:
                hidden_states = hidden_states * self.layer_scale[attention_level].unsqueeze(0).unsqueeze(0)
                
            # Add scaled residual
            output = hidden_states + residual * self.get_level_scale(attention_level)
            
            # Apply dropout
            output = self.dropout(output)
            
            # Apply post-residual addition if needed
            if post_add:
                output = output + hidden_states * self.get_level_scale(attention_level)
                
        return output
    
    def merge_residuals(
        self,
        residuals: List[torch.Tensor],
        attention_levels: List[int]
    ) -> torch.Tensor:
        """Merge multiple residual connections from different levels.
        
        Args:
            residuals: List of residual tensors
            attention_levels: List of corresponding attention levels
            
        Returns:
            Merged residual tensor
        """
        assert len(residuals) == len(attention_levels), "Number of residuals must match number of levels"
        
        # Initialize output with first residual
        output = residuals[0] * self.get_level_scale(attention_levels[0])
        
        # Add remaining scaled residuals
        for residual, level in zip(residuals[1:], attention_levels[1:]):
            scale = self.get_level_scale(level)
            if self.use_layer_scale:
                scale = scale * self.layer_scale[level]
            output = output + residual * scale
            
        return output

class AdaptiveResidual(MLAResidual):
    """Residual handler with adaptive scaling."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_levels: int,
        num_heads: int,
        temperature: float = 1.0,
        **kwargs
    ):
        """Initialize adaptive residual handler.
        
        Args:
            hidden_size: Size of hidden dimension
            num_attention_levels: Number of attention levels
            num_heads: Number of attention heads
            temperature: Temperature for attention scores
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(hidden_size, num_attention_levels, **kwargs)
        
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Adaptive scaling attention
        self.scale_query = nn.Linear(hidden_size, num_heads, bias=False)
        self.scale_key = nn.Linear(hidden_size, num_heads, bias=False)
        
        # Initialize weights
        nn.init.normal_(self.scale_query.weight, std=0.02)
        nn.init.normal_(self.scale_key.weight, std=0.02)
        
    def compute_adaptive_scale(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor
    ) -> torch.Tensor:
        """Compute adaptive scaling factor using attention mechanism.
        
        Args:
            hidden_states: Main path tensor
            residual: Residual path tensor
            
        Returns:
            Scaling factor tensor
        """
        # Compute query and key
        query = self.scale_query(hidden_states)  # [B, L, H]
        key = self.scale_key(residual)  # [B, L, H]
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.num_heads)
        
        # Apply temperature and softmax
        scale = torch.softmax(scores / self.temperature, dim=-1)
        
        return scale
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        attention_level: int,
        pre_norm: bool = True,
        post_add: bool = False,
    ) -> torch.Tensor:
        """Apply residual connection with adaptive scaling.
        
        Args:
            hidden_states: Main path tensor
            residual: Residual path tensor
            attention_level: Current attention level
            pre_norm: Whether this is a pre-norm residual
            post_add: Whether to apply post-residual addition
            
        Returns:
            Output tensor after residual connection
        """
        # Compute adaptive scale
        adaptive_scale = self.compute_adaptive_scale(hidden_states, residual)
        
        # Get base scale for level
        level_scale = self.get_level_scale(attention_level)
        
        # Combine scales
        scale = adaptive_scale * level_scale
        
        if not pre_norm:
            if self.use_layer_scale:
                hidden_states = hidden_states * self.layer_scale[attention_level].unsqueeze(0).unsqueeze(0)
                
            hidden_states = self.dropout(hidden_states)
            output = residual + hidden_states * scale
            
            if post_add:
                output = output + hidden_states
                
        else:
            if self.use_layer_scale:
                hidden_states = hidden_states * self.layer_scale[attention_level].unsqueeze(0).unsqueeze(0)
                
            output = hidden_states + residual * scale
            output = self.dropout(output)
            
            if post_add:
                output = output + hidden_states * scale
                
        return output
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_levels={self.num_attention_levels}, "
            f"num_heads={self.num_heads}, "
            f"temperature={self.temperature}"
        )
