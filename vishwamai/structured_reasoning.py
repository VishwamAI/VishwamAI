import torch
import torch.nn as nn
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    premise: torch.Tensor
    inference: torch.Tensor
    confidence: float

class StructuredReasoning(nn.Module):
    """Structured reasoning module that generates step-by-step reasoning"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Premise encoder
        self.premise_encoder = nn.Linear(hidden_size, hidden_size)
        
        # Inference generator with attention
        self.inference_generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        
        # Confidence scorer
        self.confidence_scorer = nn.Linear(hidden_size, 1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        step_context: Optional[torch.Tensor] = None
    ) -> ReasoningStep:
        """
        Forward pass for structured reasoning
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask tensor of shape (batch_size, seq_len)
            step_context: Optional context from previous reasoning steps
        
        Returns:
            ReasoningStep containing the premise, inference and confidence
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Generate premise encoding
        premise = self.premise_encoder(hidden_states)
        if attention_mask is not None:
            premise = premise * attention_mask.unsqueeze(-1)
        
        # Combine with step context if provided
        if step_context is not None:
            context_vector = step_context
        else:
            context_vector = torch.zeros_like(hidden_states)
            
        # Generate inference incorporating context
        combined = torch.cat([premise, context_vector], dim=-1)
        inference = self.inference_generator(combined)
        
        # Compute confidence score
        confidence = torch.sigmoid(self.confidence_scorer(inference)).mean()
        
        return ReasoningStep(
            premise=premise,
            inference=inference,
            confidence=confidence.item()
        )
