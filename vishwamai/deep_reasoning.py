import torch
import torch.nn as nn
from typing import Dict, List, Optional
from dataclasses import dataclass

from .structured_reasoning import StructuredReasoning
from .verification import SelfVerification 
from .metacognition import MetaCognition

@dataclass
class ReasoningOutput:
    steps: List[Dict]
    verification: Dict[str, torch.Tensor]
    meta_analysis: Dict[str, torch.Tensor]
    final_answer: str
    confidence: float

class DeepReasoning(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        
        self.structured_reasoning = StructuredReasoning(hidden_size)
        self.verification = SelfVerification(hidden_size)
        self.metacognition = MetaCognition(hidden_size)
        
        # Integration layers
        self.reasoning_merger = nn.Linear(hidden_size * 2, hidden_size)
        self.final_classifier = nn.Linear(hidden_size, config.vocab_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        knowledge_context: Optional[torch.Tensor] = None
    ) -> ReasoningOutput:
        # Step 1: Structured reasoning
        reasoning_steps = self.structured_reasoning(
            hidden_states, 
            attention_mask
        )
        
        # Step 2: Self-verification
        verification_results = self.verification(
            hidden_states,
            knowledge_context or hidden_states
        )
        
        # Step 3: Metacognition
        meta_results = self.metacognition(hidden_states)
        
        # Integrate all insights
        final_repr = self.reasoning_merger(torch.cat([
            hidden_states,
            reasoning_steps[-1].inference
        ], dim=-1))
        
        logits = self.final_classifier(final_repr)
        
        return ReasoningOutput(
            steps=[{
                "premise": step.premise.tolist(),
                "inference": step.inference.tolist(),
                "confidence": step.confidence.item()
            } for step in reasoning_steps],
            verification=verification_results,
            meta_analysis=meta_results,
            final_answer=logits.argmax(-1),
            confidence=meta_results["confidence"].mean().item()
        )
