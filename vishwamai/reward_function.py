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
                "creativity": 0.4
            }

class RewardNetwork(nn.Module):
    """Neural network for learning reward functions."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        super().__init__()
        self.config = config or RewardConfig()
        
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
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute rewards for different aspects and overall value."""
        rewards = {}
        
        # Calculate individual reward components
        for name, extractor in self.feature_extractors.items():
            features = extractor(hidden_states)
            reward = self.reward_heads[name](features)
            rewards[name] = reward * self.config.reward_scales[name]
        
        # Calculate overall value
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
