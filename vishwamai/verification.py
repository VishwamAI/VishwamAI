import torch
import torch.nn as nn
from typing import Dict, List, Tuple

class SelfVerification(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.consistency_checker = nn.Linear(hidden_size * 2, 1)
        self.fact_validator = nn.Linear(hidden_size, 1)
        self.uncertainty_estimator = nn.Linear(hidden_size, 1)
        
    def forward(
        self,
        generated_output: torch.Tensor,
        knowledge_context: torch.Tensor,
        reasoning_history: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Check internal consistency
        batch_size = generated_output.size(0)
        # Reshape and expand tensors properly
        reshaped_output = generated_output.view(batch_size, -1, generated_output.size(-1))
        pairs = torch.cat([
            reshaped_output.unsqueeze(1).expand(-1, batch_size, -1, -1),
            reshaped_output.unsqueeze(0).expand(batch_size, -1, -1, -1)
        ], dim=-1)
        consistency_scores = torch.sigmoid(self.consistency_checker(pairs))
        
        # Validate against knowledge and reasoning history
        fact_scores = torch.sigmoid(self.fact_validator(
            (generated_output * knowledge_context) + reasoning_history
        ))
        
        # Estimate uncertainty considering reasoning history
        joint_state = generated_output + reasoning_history
        uncertainty = torch.sigmoid(self.uncertainty_estimator(joint_state))
        
        return {
            "consistency": consistency_scores,
            "factual_validity": fact_scores,
            "uncertainty": uncertainty
        }
