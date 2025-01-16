import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Union, Tuple
from dataclasses import dataclass
import numpy as np
from .model import VishwamaiModel  # Corrected import
from .conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

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
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.model.device)
        
        with torch.no_grad():
            output_ids = self._generate_tokens(input_ids)
            
        # Convert tensor output to list for tokenizer
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.cpu().tolist()
            
        return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    
    def _generate_tokens(self, input_ids):
        # Ensure we don't exceed max_length
        if input_ids.shape[1] >= self.config.max_length:
            return input_ids[:, :self.config.max_length]
        
        batch_size = input_ids.shape[0]
        
        for _ in range(self.config.max_length - input_ids.shape[1]):
            outputs = self.model(input_ids)
            if outputs is None:
                return input_ids  # Handle None outputs gracefully
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
            if (next_tokens == self.tokenizer.config.eos_id).all():
                break
            
            # Enforce max_length
            if input_ids.shape[1] >= self.config.max_length:
                break
        
        # Ensure EOS token is appended if not present
        eos_present = (input_ids[:, -1] == self.tokenizer.config.eos_id)
        for i in range(batch_size):
            if not eos_present[i]:
                if input_ids.shape[1] < self.config.max_length:
                    input_ids[i] = torch.cat([input_ids[i], torch.tensor([self.tokenizer.config.eos_id], device=input_ids.device)])
                else:
                    # Replace the last token with EOS if max_length is reached
                    input_ids[i, -1] = self.tokenizer.config.eos_id
        
        return input_ids