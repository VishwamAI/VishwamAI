"""
Text Generation Utilities
========================

This module provides utilities for text generation using the VishwamAI model.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple

def sample_top_p(logits: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Sample from the top-p (nucleus) distribution of logits.
    
    Args:
        logits: Shape (batch_size, vocab_size)
        p: Cumulative probability threshold
        
    Returns:
        Sampled token indices
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum_probs <= p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = True
    
    probs[~mask] = 0
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    return torch.multinomial(probs, num_samples=1)

def sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample from logits using temperature scaling.
    
    Args:
        logits: Shape (batch_size, vocab_size)
        temperature: Sampling temperature (higher = more random)
        
    Returns:
        Sampled token indices
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)

def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_length: int = 100,
    min_length: int = 0,
    do_sample: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.9,
    num_beams: int = 1,
    pad_token_id: int = 0,
    eos_token_id: int = 2,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Generate text using various decoding strategies.
    
    Args:
        model: The VishwamAI model
        input_ids: Input token IDs
        max_length: Maximum generation length
        min_length: Minimum generation length
        do_sample: Whether to use sampling (if False, uses greedy decoding)
        temperature: Sampling temperature
        top_p: Nucleus sampling probability threshold
        num_beams: Number of beams for beam search
        pad_token_id: Padding token ID
        eos_token_id: End of sequence token ID
        attention_mask: Optional attention mask
        
    Returns:
        Generated token IDs
    """
    if num_beams > 1:
        return _beam_search(
            model, input_ids, max_length, num_beams,
            pad_token_id, eos_token_id, attention_mask
        )
    
    device = input_ids.device
    batch_size = input_ids.shape[0]
    cur_len = input_ids.shape[1]
    
    while cur_len < max_length:
        if attention_mask is None:
            attention_mask = input_ids.new_ones(batch_size, cur_len)
            
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :]
            
            # Prevent EOS before min_length
            if cur_len < min_length:
                next_token_logits[:, eos_token_id] = float('-inf')
                
            if do_sample:
                if temperature != 1.0:
                    next_token = sample_temperature(next_token_logits, temperature)
                else:
                    next_token = sample_top_p(next_token_logits, top_p)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            cur_len += 1
            
            # Stop if all sequences have generated EOS token
            if (input_ids[:, -1] == eos_token_id).all():
                break
                
    return input_ids

def _beam_search(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_length: int,
    num_beams: int,
    pad_token_id: int,
    eos_token_id: int,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Perform beam search decoding.
    
    Args:
        model: The VishwamAI model
        input_ids: Input token IDs
        max_length: Maximum generation length
        num_beams: Number of beams
        pad_token_id: Padding token ID
        eos_token_id: End of sequence token ID
        attention_mask: Optional attention mask
        
    Returns:
        Generated token IDs
    """
    batch_size = input_ids.shape[0]
    vocab_size = model.config.vocab_size
    
    # Expand input to num_beams
    input_ids = input_ids.unsqueeze(1).expand(batch_size, num_beams, -1)
    input_ids = input_ids.contiguous().view(batch_size * num_beams, -1)
    
    if attention_mask is not None:
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, num_beams, -1)
        attention_mask = attention_mask.contiguous().view(batch_size * num_beams, -1)
    
    # Initialize scores for each beam
    beam_scores = torch.zeros((batch_size, num_beams), device=input_ids.device)
    beam_scores[:, 1:] = float('-inf')
    beam_scores = beam_scores.view(-1)
    
    while input_ids.shape[-1] < max_length:
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :]
            
        # Calculate log probabilities
        next_scores = F.log_softmax(next_token_logits, dim=-1)
        
        # Add beam scores to next token scores
        next_scores = next_scores + beam_scores[:, None]
        
        # Reshape for easier handling
        next_scores = next_scores.view(batch_size, num_beams * vocab_size)
        
        # Get top-k scores and indices
        next_scores, next_tokens = torch.topk(
            next_scores, num_beams, dim=1, largest=True, sorted=True
        )
        
        # Convert indices to token IDs and beam indices
        next_beam_tokens = next_tokens % vocab_size
        next_beam_indices = next_tokens // vocab_size
        
        # Build next input_ids
        next_input_ids = []
        for batch_idx in range(batch_size):
            next_input_ids.append([])
            for beam_idx in next_beam_indices[batch_idx]:
                next_input_ids[-1].append(
                    torch.cat([
                        input_ids[batch_idx * num_beams + beam_idx],
                        next_beam_tokens[batch_idx, beam_idx].unsqueeze(0)
                    ])
                )
        
        input_ids = torch.stack([
            torch.stack(batch) for batch in next_input_ids
        ]).view(batch_size * num_beams, -1)
        
        beam_scores = next_scores.view(-1)
        
        # Check if all beams have generated EOS
        if (input_ids[:, -1] == eos_token_id).all():
            break
            
    # Select the best beam for each batch
    output_ids = []
    for batch_idx in range(batch_size):
        output_ids.append(input_ids[batch_idx * num_beams])
        
    return torch.stack(output_ids)
