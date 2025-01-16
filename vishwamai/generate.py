import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class GenerationConfig:
    max_length: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_return_sequences: int = 1

class VishwamaiGenerator:
    def __init__(self, model, tokenizer, config: Optional[GenerationConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
    def generate(self, text: str) -> List[str]:
        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        
        with torch.no_grad():
            output_ids = self._generate_tokens(input_ids)
            
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    def _generate_tokens(self, input_ids):
        batch_size = input_ids.shape[0]
        
        for _ in range(self.config.max_length):
            outputs = self.model(input_ids)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / self.config.temperature
            
            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.config.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for batch_idx in range(batch_size):
                next_token_logits[batch_idx, sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]] = float('-inf')
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Check if all sequences hit the EOS token
            if (next_tokens == self.tokenizer.eos_token_id).all():
                break
                
        return input_ids