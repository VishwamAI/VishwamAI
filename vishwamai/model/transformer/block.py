"""Transformer block implementations."""

from typing import Optional, Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn

from .config import TransformerConfig
from .layer import TransformerLayer, get_transformer_layer

class TransformerBlock(nn.Module):
    """Base transformer block combining multiple layers."""
    
    def __init__(
        self,
        config: TransformerConfig,
        block_idx: int,
        num_layers: Optional[int] = None,
        share_layers: bool = False,
        pre_norm: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize transformer block.
        
        Args:
            config: Transformer configuration
            block_idx: Index of this block
            num_layers: Optional number of layers (defaults to config value)
            share_layers: Whether to share layer parameters
            pre_norm: Whether to use pre-normalization architecture
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.block_idx = block_idx
        self.num_layers = num_layers or config.num_hidden_layers
        self.share_layers = share_layers
        self.pre_norm = pre_norm
        
        # Create transformer layers
        if share_layers:
            # Create single shared layer
            self.layer = get_transformer_layer(
                config=config,
                layer_idx=0,
                pre_norm=pre_norm,
                **factory_kwargs
            )
        else:
            # Create multiple layers
            self.layers = nn.ModuleList([
                get_transformer_layer(
                    config=config,
                    layer_idx=i,
                    pre_norm=pre_norm,
                    **factory_kwargs
                )
                for i in range(self.num_layers)
            ])
            
        # Optional final layer normalization
        if not pre_norm:
            self.final_norm = nn.LayerNorm(
                config.hidden_size,
                eps=config.layer_norm_eps,
                **factory_kwargs
            )
        else:
            self.final_norm = None
            
    def _get_layer(self, layer_idx: int) -> TransformerLayer:
        """Get layer by index.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Transformer layer
        """
        if self.share_layers:
            return self.layer
        else:
            return self.layers[layer_idx]
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass through transformer block.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask tensor
            position_embeddings: Optional tuple of (cos, sin) rotary position embeddings
            past_key_values: Optional list of cached (key, value) tensors per layer
            use_cache: Whether to return key/value tensors for incremental decoding
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return all hidden states
            **kwargs: Additional arguments passed to layers
            
        Returns:
            Tuple containing:
            - Output tensor
            - Optional list of cached (key, value) tensors per layer
            - Optional list of attention weights per layer
            - Optional list of hidden states per layer
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        
        # Process through layers
        for i in range(self.num_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Get past key/value states for this layer
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Process through layer
            layer_outputs = self._get_layer(i)(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            hidden_states = layer_outputs[0]
            
            # Cache key/values if needed
            if use_cache:
                all_past_key_values = all_past_key_values + (layer_outputs[1],)
                
            # Cache attention weights if needed
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[-1],)
                
        # Apply final normalization if needed
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (all_past_key_values,)
            
        if output_hidden_states:
            outputs += (all_hidden_states,)
            
        if output_attentions:
            outputs += (all_attentions,)
            
        return outputs
    
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"block_idx={self.block_idx}, "
            f"num_layers={self.num_layers}, "
            f"shared={self.share_layers}, "
            f"pre_norm={self.pre_norm}"
        )

class ParallelTransformerBlock(TransformerBlock):
    """Transformer block optimized for parallel computation."""
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with parallel layer computation where possible."""
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        
        # Compute shared components once
        if self.share_layers:
            # Use single layer for all positions
            all_outputs = []
            for i in range(self.num_layers):
                past_key_value = past_key_values[i] if past_key_values is not None else None
                layer_outputs = self.layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )
                all_outputs.append(layer_outputs)
                hidden_states = layer_outputs[0]
                
            # Process outputs
            for i, layer_outputs in enumerate(all_outputs):
                if output_hidden_states and i < len(all_outputs) - 1:
                    all_hidden_states = all_hidden_states + (layer_outputs[0],)
                    
                if use_cache:
                    all_past_key_values = all_past_key_values + (layer_outputs[1],)
                    
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[-1],)
                    
        else:
            # Process through layers normally
            for i in range(self.num_layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                    
                past_key_value = past_key_values[i] if past_key_values is not None else None
                layer_outputs = self.layers[i](
                    hidden_states,
                    attention_mask=attention_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    **kwargs
                )
                hidden_states = layer_outputs[0]
                
                if use_cache:
                    all_past_key_values = all_past_key_values + (layer_outputs[1],)
                    
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[-1],)
                    
        # Apply final normalization if needed
        if self.final_norm is not None:
            hidden_states = self.final_norm(hidden_states)
            
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Prepare outputs
        outputs = (hidden_states,)
        
        if use_cache:
            outputs += (all_past_key_values,)
            
        if output_hidden_states:
            outputs += (all_hidden_states,)
            
        if output_attentions:
            outputs += (all_attentions,)
            
        return outputs
