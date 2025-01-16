import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
from .fp8_cast_bf16 import fp8_cast

@dataclass
class VishwamaiConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

class VishwamaiModel(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.config = config
        
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, config.num_hidden_layers)
        
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
        self._device = "cpu"
        
    @property
    def device(self):
        return self._device
        
    def to(self, device):
        self._device = device if isinstance(device, str) else device.type
        return super().to(device)
        
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).expand(batch_size, -1)
        
        # Get embeddings
        inputs_embeds = self.embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = inputs_embeds + position_embeds
        
        # Apply transformer layers
        if attention_mask is None:
            attention_mask = torch.zeros((batch_size, seq_length), dtype=torch.bool, device=input_ids.device)
        else:
            attention_mask = (attention_mask == 0)
        
        hidden_states = self.encoder(hidden_states, src_key_padding_mask=attention_mask)
        
        # Apply final layer norm and head
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)
        
        return logits