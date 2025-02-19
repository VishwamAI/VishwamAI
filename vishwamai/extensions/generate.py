"""
Text generation utilities and implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np

from transformers import PreTrainedTokenizer, AutoTokenizer

from vishwamai.models.Transformer import Transformer
from vishwamai.utils.config import ModelConfig
from vishwamai.models.tokenizer import Tokenizer

@dataclass
class ModelArgs:
    """Arguments for model generation."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_length: int = 2048
    min_length: int = 0
    num_return_sequences: int = 1
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = False
    use_cache: bool = True
    length_penalty: float = 1.0
    bad_words_ids: Optional[List[List[int]]] = None

class Generator:
    """Text generation with various decoding strategies."""
    
    def __init__(
        self,
        model: Transformer,
        tokenizer: Union[Tokenizer, PreTrainedTokenizer],
        max_length: int = 2048,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def generate(
        self,
        input_text: Union[str, List[str]],
        max_length: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from input prompt.
        
        Args:
            input_text: Input prompt(s)
            max_length: Maximum length of generated sequence
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text or list of texts
        """
        # Encode input text
        if isinstance(input_text, str):
            input_text = [input_text]
            
        encoded = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Set up generation config
        gen_kwargs = {
            'max_length': max_length or self.max_length,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'attention_mask': attention_mask,
            **kwargs
        }
        
        # Generate
        with torch.no_grad():
            output_sequences = self._generate(
                input_ids=input_ids,
                **gen_kwargs
            )
            
        # Decode output sequences
        generated_texts = self.tokenizer.batch_decode(
            output_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        if len(generated_texts) == 1:
            return generated_texts[0]
        return generated_texts
        
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        **kwargs
    ) -> torch.Tensor:
        """Internal generation method."""
        batch_size = input_ids.shape[0]
        
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat(num_return_sequences, 1)
                
        unfinished_sequences = torch.ones(
            input_ids.shape[0], 1,
            device=self.device,
            dtype=torch.long
        )
        
        while input_ids.shape[-1] < max_length:
            # Get model outputs
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True
            )
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty
                        
            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                    
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)[:, None]
                
            # Append next tokens
            next_tokens = next_tokens * unfinished_sequences
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), device=self.device)
                ], dim=-1)
                
            # Update finished sequences mask
            unfinished_sequences = unfinished_sequences.mul(
                (next_tokens != self.tokenizer.eos_token_id).long()
            )
            
            # Stop if all sequences are finished
            if unfinished_sequences.max() == 0:
                break
                
        return input_ids
