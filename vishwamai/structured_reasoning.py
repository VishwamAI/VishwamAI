import torch
import torch.nn as nn
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class ReasoningStep:
    premise: str
    inference: str
    confidence: float
    
class StructuredReasoning(nn.Module):
    def __init__(self, hidden_size: int, num_reasoning_steps: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_steps = num_reasoning_steps
        
        # Reasoning components
        self.premise_encoder = nn.Linear(hidden_size, hidden_size)
        self.inference_generator = nn.TransformerDecoderLayer(hidden_size, nhead=8)
        self.confidence_scorer = nn.Linear(hidden_size, 1)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[ReasoningStep]:
        batch_size = hidden_states.size(0)
        reasoning_steps = []
        
        current_state = hidden_states
        for _ in range(self.num_steps):
            # Encode current understanding
            premise_repr = self.premise_encoder(current_state)
            
            # Generate inference
            inference_repr = self.inference_generator(
                premise_repr,
                hidden_states,
                tgt_mask=attention_mask
            )
            
            # Score confidence
            confidence = torch.sigmoid(self.confidence_scorer(inference_repr))
            
            # Store reasoning step
            reasoning_steps.append(ReasoningStep(
                premise=premise_repr,
                inference=inference_repr,
                confidence=confidence
            ))
            
            # Update state for next step
            current_state = inference_repr
            
        return reasoning_steps
