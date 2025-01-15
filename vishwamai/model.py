import math
from typing import Optional, List, Tuple, Dict, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from dataclasses import dataclass
from transformers import PreTrainedModel, BertConfig

from .architecture import (
    VishwamaiConfig,
    TransformerBlock,
    RMSNorm,
    precompute_freqs_cis
)

class VishwamaiModel(PreTrainedModel):
    """
    Main Vishwamai model implementation compatible with Hugging Face's Transformers.
    """
    config_class = VishwamaiConfig

    def __init__(self, config: VishwamaiConfig):
        super().__init__(config)
        retries = 0
        while retries < 3:
            try:
                # Core components
                self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
                
                # Transformer layers
                self.layers = nn.ModuleList([
                    TransformerBlock(config) for _ in range(config.n_layers)
                ])
                
                # Output components
                self.norm = RMSNorm(config.dim, config.norm_eps)
                self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
                
                # Position embeddings
                self.freqs_cis = precompute_freqs_cis(
                    config.dim // config.n_heads,
                    config.max_seq_len * 2,
                    config.rope_theta,
                    config.rope_scaling
                )
                
                # Initialize weights
                self.apply(self._init_weights)
                
                # Add memory optimization flags
                self.gradient_checkpointing = False
                self.use_activation_checkpointing = False
                break
            except RuntimeError:
                retries += 1
                if retries == 3:
                    raise

    def _init_weights(self, module):
        """Initialize model weights."""
        try:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        except Exception as e:
            logging.error(f"Error in _init_weights: {e}")
            raise

    def enable_memory_efficient_training(self):
        """Enable memory optimizations."""
        try:
            self.gradient_checkpointing = True
            self.use_activation_checkpointing = True
            for layer in self.layers:
                layer.gradient_checkpointing = True
                layer.attention.use_flash_attention = True
        except Exception as e:
            logging.error(f"Error enabling memory efficient training: {e}")
            raise

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        for layer in self.layers:
            if hasattr(layer, 'gradient_checkpointing'):
                layer.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            past_key_values: Optional cached key/value states for efficient inference
            use_cache: Whether to use cached key/values
            start_pos: Starting position for position embeddings
            return_dict: Whether to return outputs as dictionary
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs either as tensor or dictionary
        """
        # Input validation
        try:
            if input_ids.size(0) == 0:
                raise ValueError("Batch size cannot be zero.")
            if input_ids.dim() != 2:
                raise ValueError("input_ids must be a 2D tensor [batch_size, sequence_length]")
            
            if (input_ids >= self.config.vocab_size).any():
                raise ValueError("Token indices exceed vocabulary size.")
            
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            
            # Ensure tokens are within vocabulary size
            input_ids = torch.clamp(input_ids, max=self.config.vocab_size - 1)
            
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length), 
                    dtype=torch.bool,
                    device=device
                )
            
            # Create causal mask
            causal_mask = torch.triu(
                torch.full(
                    (seq_length, seq_length), 
                    float('-inf'),
                    device=device
                ),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) * causal_mask
            
            # Initialize outputs
            hidden_states = []
            attentions = []
            present_key_values = [] if use_cache else None
            
            # Get input embeddings
            h = self.tok_embeddings(input_ids)
            
            # Process through transformer layers
            for idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    hidden_states.append(h)
                    
                past_kv = past_key_values[idx] if past_key_values is not None else None
                
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    h = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        h,
                        start_pos,
                        self.freqs_cis,
                        attention_mask
                    )
                else:
                    h = layer(h, start_pos, self.freqs_cis, attention_mask)
                
                if use_cache:
                    present_key_values.append(layer.attention.get_kv_cache())
                    
                if output_attentions:
                    attentions.append(layer.attention.get_attention_weights())
                    
            # Final normalization
            h = self.norm(h)
            
            # Get logits
            logits = self.output(h)
            
            if not return_dict:
                return logits
                
            return {
                'logits': logits,
                'hidden_states': hidden_states if output_hidden_states else None,
                'attentions': attentions if output_attentions else None,
                'past_key_values': present_key_values
            }
        except Exception as e:
            logging.error(f"Real-world error in VishwamaiModel.forward: {e}")
            raise e

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        start_pos: int = 0,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
            past_key_values: Optional cached key/value states for efficient inference
            use_cache: Whether to use cached key/values
            start_pos: Starting position for position embeddings
            return_dict: Whether to return outputs as dictionary
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights
            
        Returns:
            Model outputs either as tensor or dictionary
        """
        # Input validation
        try:
            if input_ids.size(0) == 0:
                raise ValueError("Batch size cannot be zero.")
            if input_ids.dim() != 2:
                raise ValueError("input_ids must be a 2D tensor [batch_size, sequence_length]")
            
            if (input_ids >= self.config.vocab_size).any():
                raise ValueError("Token indices exceed vocabulary size.")
            
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
            
            # Ensure tokens are within vocabulary size
            input_ids = torch.clamp(input_ids, max=self.config.vocab_size - 1)
            
            # Create attention mask if not provided
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length), 
                    dtype=torch.bool,
                    device=device
                )
            
            # Create causal mask
            causal_mask = torch.triu(
                torch.full(
                    (seq_length, seq_length), 
                    float('-inf'),
                    device=device
                ),
                diagonal=1
            )
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1) * causal_mask
            
            # Initialize outputs
            hidden_states = []
            attentions = []
            present_key_values = [] if use_cache else None
            
            # Get input embeddings
            h = self.tok_embeddings(input_ids)
            
            # Process through transformer layers
            for idx, layer in enumerate(self.layers):
                if output_hidden_states:
                    hidden_states.append(h)
                    
                past_kv = past_key_values[idx] if past_key_values is not None else None
                
                h = layer(
                    h,
                    start_pos=start_pos,
                    freqs_cis=self.freqs_cis,
                    mask=attention_mask
                )
                
                if use_cache:
                    present_key_values.append(layer.attention.get_kv_cache())
                    
                if output_attentions:
                    attentions.append(layer.attention.get_attention_weights())
                    
            # Final normalization
            h = self.norm(h)
            
            # Get logits
            logits = self.output(h)
            
            if not return_dict:
                return logits
                
            return {
                'logits': logits,
                'hidden_states': hidden_states if output_hidden_states else None,
                'attentions': attentions if output_attentions else None,
                'past_key_values': present_key_values
            }
        except Exception as e:
            logging.error(f"Real-world error in VishwamaiModel.forward: {e}")
            raise e

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text using the model.
        
        Args:
            input_ids: Initial input tokens
            max_length: Maximum length to generate
            min_length: Minimum length to generate
            do_sample: Whether to use sampling vs greedy decoding
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            repetition_penalty: Penalty for repeating tokens
            pad_token_id: Token ID for padding
            eos_token_id: Token ID for end of sequence
            use_cache: Whether to use KV caching
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs
        """
        # Setup generation parameters
        max_length = max_length or self.config.max_seq_len
        min_length = min_length or 0
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Track unfinished sequences
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        cur_len = input_ids.shape[1]
        past_key_values = None
        
        while cur_len < max_length:
            # Forward pass
            model_inputs = {
                'input_ids': input_ids if past_key_values is None else input_ids[:, -1:],
                'past_key_values': past_key_values,
                'use_cache': use_cache,
                'start_pos': cur_len - 1 if past_key_values is not None else 0
            }
            
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs['logits'][:, -1, :]
            past_key_values = outputs['past_key_values']

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
                        
            # Handle minimum length
            if min_length and cur_len < min_length:
                next_token_logits[:, eos_token_id] = float('-inf')
                
            if do_sample:
                # Apply top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(batch_size):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = float('-inf')
                        
                # Sample next tokens
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1)
                
            # Finished sequences should have their next token be a padding token
            if eos_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                
            # Update input_ids
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = input_ids.shape[1]
            
            # Update unfinished sequences
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.ne(eos_token_id).long()
                )
                
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
                
        return input_ids
        
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare inputs for generation step."""
        # Only last token for inputs_ids if past is defined
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask
        }

    def adjust_logits_during_generation(
        self,
        logits: torch.Tensor,
        cur_len: int,
        min_length: int,
        max_length: int,
        **kwargs
    ) -> torch.Tensor:
        """Adjust logits during generation for constraints."""
        if cur_len < min_length:
            # Disable EOS token if minimum length not reached
            logits[:, self.config.eos_token_id] = float('-inf')
        return logits

def init_model(config: VishwamaiConfig) -> VishwamaiModel:
    """Initialize a new Vishwamai model."""
    try:
        model = VishwamaiModel(config)
        # Initialize with proper scaling
        for name, param in model.named_parameters():
            if param.ndim == 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        # Ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
        return model
    except Exception as e:
        logging.error(f"Error initializing model: {e}")
        raise