import torch
import torch.nn as nn
from typing import Dict, Optional

class MetaCognition(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.strategy_selector = nn.Linear(hidden_size, 3)  # 3 strategies
        self.complexity_estimator = nn.Linear(hidden_size, 1)
        self.confidence_predictor = nn.Linear(hidden_size, 1)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        task_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Select reasoning strategy
        strategy_logits = self.strategy_selector(hidden_states)
        strategy_probs = torch.softmax(strategy_logits, dim=-1)
        
        # Estimate task complexity
        complexity = torch.sigmoid(self.complexity_estimator(hidden_states))
        
        # Predict confidence
        confidence = torch.sigmoid(self.confidence_predictor(hidden_states))
        
        return {
            "strategy": strategy_probs,
            "complexity": complexity,
            "confidence": confidence
        }
