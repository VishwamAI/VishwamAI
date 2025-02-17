import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

@dataclass
class RewardConfig:
    """Configuration for reward function."""
    hidden_size: int = 1024
    num_layers: int = 3
    dropout: float = 0.1
    reward_scales: Dict[str, float] = None
    
    def __post_init__(self):
        if self.reward_scales is None:
            self.reward_scales = {
                "relevance": 1.0,
                "coherence": 0.8,
                "factuality": 1.0,
                "depth": 0.6,
                "creativity": 0.4,
                "mathematical_reasoning": 1.0,
                "step_validity": 0.8,
                "thought_consistency": 0.7,
                "solution_completeness": 0.9
            }

class MathStepEvaluator(nn.Module):
    """Evaluates mathematical reasoning steps."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.op_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 5)  # Scores for +, -, *, /, =
        )
        
        self.step_validator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, current_state: torch.Tensor, next_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate mathematical step validity and operation scores."""
        # Score operations
        op_scores = self.op_scorer(current_state)
        
        # Validate step
        combined = torch.cat([current_state, next_state], dim=-1)
        validity = self.step_validator(combined)
        
        return op_scores, validity

class RewardNetwork(nn.Module):
    """Enhanced neural network for learning reward functions with mathematical reasoning."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__()
        self.config = config or RewardConfig()
        
        # Math reasoning components
        self.math_evaluator = MathStepEvaluator(self.config.hidden_size)
        
        # Thought path evaluation
        self.thought_evaluator = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Feature extractors for different reward aspects
        self.feature_extractors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_size, self.config.hidden_size // 2)
            )
            for name in self.config.reward_scales.keys()
        })
        
        # Reward heads for different aspects
        self.reward_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.config.hidden_size // 2, self.config.hidden_size // 4),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_size // 4, 1)
            )
            for name in self.config.reward_scales.keys()
        })
        
        # Value prediction head
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size // 2, 1)
        )
        
    def evaluate_thought_path(self, states: List[torch.Tensor]) -> torch.Tensor:
        """Evaluate coherence and validity of a thought path."""
        if len(states) < 2:
            return torch.tensor(0.0, device=states[0].device)
            
        path_scores = []
        for i in range(len(states) - 1):
            combined = torch.cat([states[i], states[i + 1]], dim=-1)
            score = self.thought_evaluator(combined)
            path_scores.append(score)
            
        return torch.mean(torch.stack(path_scores))
    
    def evaluate_math_reasoning(
        self,
        states: List[torch.Tensor],
        operations: List[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Evaluate mathematical reasoning steps."""
        if not states or not operations:
            return {
                'operation_scores': torch.tensor(0.0),
                'step_validity': torch.tensor(0.0),
                'reasoning_score': torch.tensor(0.0)
            }
            
        op_scores_list = []
        validity_scores = []
        
        for i in range(len(states) - 1):
            op_scores, validity = self.math_evaluator(states[i], states[i + 1])
            target_op = operations[i].argmax(dim=-1)
            op_score = op_scores[torch.arange(op_scores.size(0)), target_op]
            
            op_scores_list.append(op_score)
            validity_scores.append(validity)
            
        avg_op_score = torch.mean(torch.stack(op_scores_list))
        avg_validity = torch.mean(torch.stack(validity_scores))
        
        return {
            'operation_scores': avg_op_score,
            'step_validity': avg_validity,
            'reasoning_score': (avg_op_score + avg_validity) / 2
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        thought_path: Optional[List[torch.Tensor]] = None,
        math_operations: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute rewards for different aspects including mathematical reasoning."""
        rewards = {}
        
        # Base rewards
        rewards = {}
        for name, extractor in self.feature_extractors.items():
            features = extractor(hidden_states)
            reward = self.reward_heads[name](features)
            rewards[name] = reward * self.config.reward_scales[name]
        
        # Thought path evaluation
        if thought_path:
            rewards['thought_consistency'] = (
                self.evaluate_thought_path(thought_path) * 
                self.config.reward_scales['thought_consistency']
            )
        
        # Mathematical reasoning evaluation
        if thought_path and math_operations:
            math_rewards = self.evaluate_math_reasoning(thought_path, math_operations)
            for name, value in math_rewards.items():
                rewards[name] = value * self.config.reward_scales['mathematical_reasoning']
        
        # Calculate overall value incorporating all components
        value = self.value_head(hidden_states)
        rewards['value'] = value
        
        return rewards

class SLAP(nn.Module):
    """Scaled Learning through Action Prediction."""
    
    def __init__(self, reward_net: RewardNetwork):
        super().__init__()
        self.reward_net = reward_net
        
        # Action prediction networks
        self.action_encoder = nn.Sequential(
            nn.Linear(reward_net.config.hidden_size, reward_net.config.hidden_size),
            nn.GELU(),
            nn.Dropout(reward_net.config.dropout),
            nn.Linear(reward_net.config.hidden_size, reward_net.config.hidden_size // 2)
        )
        
        self.action_predictor = nn.Sequential(
            nn.Linear(reward_net.config.hidden_size // 2, reward_net.config.hidden_size // 4),
            nn.GELU(), 
            nn.Dropout(reward_net.config.dropout),
            nn.Linear(reward_net.config.hidden_size // 4, reward_net.config.hidden_size // 8),
            nn.GELU(),
            nn.Linear(reward_net.config.hidden_size // 8, 1)
        )
        
    def forward(self, 
                hidden_states: torch.Tensor,
                actions: Optional[torch.Tensor] = None) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute rewards and predict optimal actions.
        Args:
            hidden_states: Input hidden states
            actions: Optional ground truth actions for training
        Returns:
            rewards: Dictionary of computed rewards
            action_loss: Action prediction loss if actions provided
        """
        # Get rewards from reward network
        rewards = self.reward_net(hidden_states)
        
        # Predict actions
        action_features = self.action_encoder(hidden_states)
        predicted_actions = self.action_predictor(action_features)
        
        action_loss = None
        if actions is not None:
            # Compute action prediction loss
            action_loss = F.mse_loss(predicted_actions, actions)
            
        return rewards, action_loss

class RewardTrainer:
    """Trainer for reward function and SLAP."""
    
    def __init__(self,
                 slap_module: SLAP,
                 learning_rate: float = 1e-4,
                 max_grad_norm: float = 1.0):
        self.slap = slap_module
        self.optimizer = torch.optim.AdamW(slap_module.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm
        
    def train_step(self,
                  hidden_states: torch.Tensor,
                  target_rewards: Dict[str, torch.Tensor],
                  actions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Single training step."""
        self.slap.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        rewards, action_loss = self.slap(hidden_states, actions)
        
        # Compute rewards loss
        reward_losses = {
            name: F.mse_loss(rewards[name], target)
            for name, target in target_rewards.items()
        }
        
        # Combine losses
        total_loss = sum(reward_losses.values())
        if action_loss is not None:
            total_loss = total_loss + action_loss
            
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.slap.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Return losses
        loss_dict = {f"{name}_loss": loss.item() for name, loss in reward_losses.items()}
        if action_loss is not None:
            loss_dict["action_loss"] = action_loss.item()
        loss_dict["total_loss"] = total_loss.item()
        
        return loss_dict
        
    @torch.no_grad()
    def evaluate(self,
                hidden_states: torch.Tensor,
                target_rewards: Dict[str, torch.Tensor],
                actions: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Evaluate reward predictions."""
        self.slap.eval()
        rewards, action_loss = self.slap(hidden_states, actions)
        
        metrics = {}
        for name, target in target_rewards.items():
            metrics[f"{name}_mse"] = F.mse_loss(rewards[name], target).item()
            metrics[f"{name}_mae"] = F.l1_loss(rewards[name], target).item()
            
        if action_loss is not None:
            metrics["action_loss"] = action_loss.item()
            
        return metrics

    def save_checkpoint(self, path: str):
        """Save trainer checkpoint."""
        torch.save({
            'model_state_dict': self.slap.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load trainer checkpoint."""
        checkpoint = torch.load(path)
        self.slap.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
