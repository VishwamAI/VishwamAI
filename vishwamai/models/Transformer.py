"""
Core Transformer implementation for VishwamAI
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, Dict, Any, List, Union, Tuple

from vishwamai.utils.config import ModelConfig
from vishwamai.models.base_layers import Linear, LayerNorm, Embedding
from vishwamai.models.MLA import MLA
from vishwamai.models.MLP import MLP
from vishwamai.utils.parallel import model_parallel_forward

class TransformerConfig:
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 2048,
        num_layers: int = 24,
        num_heads: int = 16,
        intermediate_size: int = 8192,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        use_mla: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_mla = use_mla

class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Multi-Level Attention if enabled
        if config.use_mla:
            self.attention = MLA(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads
            )
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
            
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout
        )
        
        self.ln1 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln2 = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-LayerNorm architecture
        h = self.ln1(x)
        
        if self.config.use_mla:
            h = self.attention(h, attention_mask=attention_mask)
        else:
            h, _ = self.attention(h, h, h, key_padding_mask=attention_mask)
            
        x = x + self.dropout(h)
        
        h = self.ln2(x)
        h = self.mlp(h)
        x = x + self.dropout(h)
        
        return x

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = Embedding(config.vocab_size, config.hidden_size)
        self.wpe = Embedding(config.max_position_embeddings, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        self.ln_f = LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (Linear, Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def get_input_embeddings(self) -> nn.Module:
        return self.wte
        
    def set_input_embeddings(self, new_embeddings: nn.Module):
        self.wte = new_embeddings
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        device = input_ids.device
        input_shape = input_ids.size()
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        
        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            
        # Prepare attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            
        # Embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
            
        hidden_states = self.drop(hidden_states)
        
        # Transformer blocks
        all_hidden_states = [hidden_states]
        attention_weights = []
        
        for block in self.blocks:
            hidden_states = model_parallel_forward(
                block,
                hidden_states,
                attention_mask=attention_mask
            )
            all_hidden_states.append(hidden_states)
            
        hidden_states = self.ln_f(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'hidden_states': all_hidden_states,
            'attentions': attention_weights
        }
