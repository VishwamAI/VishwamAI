"""Transformer model implementation."""

from typing import Optional, Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn

from ..embeddings import EmbeddingLayer
from .config import TransformerConfig, MoEMLAConfig
from .block import TransformerBlock, ParallelTransformerBlock
from .moe_mla_block import MoEMLABlock

class TransformerModel(nn.Module):
    """Base transformer language model."""
    
    def __init__(
        self,
        config: TransformerConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Initialize transformer model.
        
        Args:
            config: Transformer configuration
            device: Device to create tensors on
            dtype: Data type for parameters
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.embeddings = EmbeddingLayer(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            max_seq_length=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            positional_encoding=config.position_embedding_type,
            tie_weights=config.tie_word_embeddings,
            **factory_kwargs
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config=config,
                block_idx=i,
                pre_norm=True,
                **factory_kwargs
            )
            for i in range(config.num_hidden_layers)
        ])
        
        # Output head
        self.output_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False if config.tie_word_embeddings else True,
            **factory_kwargs
        )
        
        # Initialize weights
        self._init_weights()
        
        # Tie weights if enabled
        if config.tie_word_embeddings:
            self.tie_weights()
            
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize embedding weights
        self.embeddings.token_embedding._init_weights()
        
        # Initialize output head
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.output_head.weight, std=self.config.initializer_range)
            if self.output_head.bias is not None:
                nn.init.zeros_(self.output_head.bias)
                
    def tie_weights(self):
        """Tie embedding and output layer weights."""
        self.output_head.weight = self.embeddings.token_embedding.weight
        
    def get_input_embeddings(self) -> nn.Module:
        """Get input embeddings layer."""
        return self.embeddings.token_embedding
        
    def set_input_embeddings(self, embeddings: nn.Module):
        """Set input embeddings layer.
        
        Args:
            embeddings: New embeddings module
        """
        self.embeddings.token_embedding = embeddings
        
    def get_output_embeddings(self) -> nn.Module:
        """Get output embeddings layer."""
        return self.output_head
        
    def set_output_embeddings(self, embeddings: nn.Module):
        """Set output embeddings layer.
        
        Args:
            embeddings: New embeddings module
        """
        self.output_head = embeddings
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Forward pass through transformer model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_length]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_values: Optional cached key/value states for incremental decoding
            labels: Optional target token IDs for computing loss
            use_cache: Whether to return key/value states for incremental decoding
            output_attentions: Whether to return attention weights
            output_hidden_states: Whether to return hidden states
            return_dict: Whether to return outputs as dictionary
            
        Returns:
            Model outputs either as tuple or dictionary
        """
        # Get input embeddings
        hidden_states = self.embeddings(input_ids, position_ids)
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        
        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                
            # Get past key/values for this block
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Process through block
            block_outputs = block(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=None,  # Handled by embedding layer
                past_key_values=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = block_outputs[0]
            
            if use_cache:
                all_past_key_values = all_past_key_values + (block_outputs[1],)
                
            if output_attentions:
                all_attentions = all_attentions + (block_outputs[-1],)
                
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            
        # Compute logits
        logits = self.output_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        if not return_dict:
            # Return tuple of tensors
            outputs = (logits, loss) if loss is not None else (logits,)
            if use_cache:
                outputs += (all_past_key_values,)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_attentions,)
            return outputs
            
        # Return outputs as dictionary
        return {
            "logits": logits,
            "loss": loss,
            "past_key_values": all_past_key_values,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }
        
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for text generation.
        
        Args:
            input_ids: Input token IDs
            past_key_values: Optional cached key/value states
            attention_mask: Optional attention mask
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of model inputs
        """
        # Only last token for inputs_ids if past is defined
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
        
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device

class MoEMLATransformerModel(TransformerModel):
    """Transformer model with MoE and MLA mechanisms."""
    
    def __init__(
        self,
        config: MoEMLAConfig,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """Initialize MoE-MLA transformer model."""
        super().__init__(config, device=device, dtype=dtype)
        
        # Replace standard blocks with MoE-MLA blocks
        self.blocks = nn.ModuleList([
            MoEMLABlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                num_attention_levels=config.num_attention_levels,
                num_experts=config.num_experts,
                expert_capacity_factor=config.expert_capacity_factor,
                moe_layer_position=config.moe_layer_position,
                use_expert_choice=config.use_expert_choice,
                share_expert_params=config.share_expert_params,
                intermediate_size=config.intermediate_size,
                activation=config.hidden_act,
                attention_dropout_prob=config.attention_dropout_prob,
                hidden_dropout_prob=config.hidden_dropout_prob,
                layer_norm_eps=config.layer_norm_eps,
                use_adaptive_residual=config.use_adaptive_residual,
                use_layer_scale=config.use_layer_scale,
                layer_scale_init=config.layer_scale_init,
                level_scale_factor=config.level_scale_factor,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.num_hidden_layers)
        ])
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Forward pass with MoE-MLA specific handling."""
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,  # Always get dict for processing
        )
        
        if output_router_logits:
            # Collect router logits from MoE layers
            router_logits = []
            for block in self.blocks:
                if hasattr(block, "moe"):
                    router_logits.append(block.moe.router.weight)
            outputs["router_logits"] = router_logits
            
        if not return_dict:
            # Convert dict to tuple
            return tuple(outputs.values())
            
        return outputs
