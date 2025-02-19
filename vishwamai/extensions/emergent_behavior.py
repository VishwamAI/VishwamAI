import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class EmergentConfig:
    """Configuration for emergent behavior module."""
    novelty_threshold: float = 0.7
    curiosity_weight: float = 0.3
    self_play_buffer_size: int = 1000
    min_novelty_count: int = 50
    exploration_decay: float = 0.995
    innovation_threshold: float = 0.5
    max_complexity: float = 5.0
    adaptation_rate: float = 0.1

class EmergentBehaviorModule(nn.Module):
    """
    Module for encouraging emergent behavior through self-play and intrinsic motivation.
    
    This module implements mechanisms for:
    - Self-generated task creation
    - Novelty detection and rewards
    - Complexity growth tracking
    - Adaptive exploration strategies
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[EmergentConfig] = None
    ):
        super().__init__()
        self.config = config or EmergentConfig()
        self.hidden_dim = hidden_dim
        
        # Neural networks for novelty detection and task generation
        self.novelty_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.task_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # State tracking
        self.experience_buffer = []
        self.novelty_history = []
        self.complexity_curve = []
        self.current_exploration_rate = 1.0
        
    def forward(
        self,
        current_state: torch.Tensor,
        action_history: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process current state for emergent behavior detection and generation.
        
        Args:
            current_state: Current hidden state tensor
            action_history: History of previous actions
            
        Returns:
            Tuple of (generated task embedding, metrics dictionary)
        """
        # Calculate novelty score
        novelty_score = self.detect_novelty(current_state)
        
        # Update experience buffer
        if len(self.experience_buffer) >= self.config.self_play_buffer_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append(current_state.detach())
        
        # Generate new task based on current state and history
        task_context = self._create_task_context(current_state, action_history)
        generated_task = self.task_generator(task_context)
        
        # Update exploration rate
        self.current_exploration_rate *= self.config.exploration_decay
        
        # Calculate complexity metrics
        complexity_score = self._calculate_complexity(current_state)
        self.complexity_curve.append(complexity_score)
        
        metrics = {
            'novelty_score': novelty_score.item(),
            'complexity_score': complexity_score,
            'exploration_rate': self.current_exploration_rate,
            'buffer_size': len(self.experience_buffer)
        }
        
        return generated_task, metrics
    
    def detect_novelty(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate novelty score for current state."""
        if not self.experience_buffer:
            return torch.tensor(1.0, device=state.device)
            
        # Compare with previous experiences
        similarities = []
        for past_state in self.experience_buffer[-self.config.min_novelty_count:]:
            sim = F.cosine_similarity(state.flatten(), past_state.flatten(), dim=0)
            similarities.append(sim)
            
        avg_similarity = torch.mean(torch.stack(similarities))
        novelty_score = 1.0 - avg_similarity
        
        self.novelty_history.append(novelty_score.item())
        return novelty_score
    
    def _create_task_context(
        self,
        current_state: torch.Tensor,
        action_history: List[torch.Tensor]
    ) -> torch.Tensor:
        """Create context for task generation."""
        # Combine current state with action history
        if action_history:
            history_tensor = torch.stack(action_history[-5:], dim=0).mean(dim=0)
        else:
            history_tensor = torch.zeros_like(current_state)
            
        return torch.cat([current_state, history_tensor], dim=-1)
    
    def _calculate_complexity(self, state: torch.Tensor) -> float:
        """Calculate complexity score of current state."""
        # Measure complexity through various metrics
        entropy = -torch.sum(F.softmax(state, dim=-1) * F.log_softmax(state, dim=-1))
        gradient_norm = torch.norm(torch.gradient(state)[0])
        
        complexity = (entropy * gradient_norm).item()
        return min(complexity, self.config.max_complexity)
    
    def generate_intrinsic_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor
    ) -> torch.Tensor:
        """Generate intrinsic motivation reward."""
        # Calculate intrinsic reward based on novelty and learning progress
        novelty_reward = self.detect_novelty(next_state)
        
        # Measure learning progress
        if len(self.complexity_curve) > 1:
            progress = (self.complexity_curve[-1] - self.complexity_curve[-2])
            progress_reward = torch.tensor(max(0, progress))
        else:
            progress_reward = torch.tensor(0.0)
            
        # Combine rewards
        intrinsic_reward = (
            novelty_reward * self.config.curiosity_weight +
            progress_reward * (1 - self.config.curiosity_weight)
        )
        
        return intrinsic_reward
    
    def should_explore(self) -> bool:
        """Determine if system should explore new behaviors."""
        return (
            torch.rand(1).item() < self.current_exploration_rate or
            len(self.experience_buffer) < self.config.min_novelty_count
        )
    
    def get_learning_trajectory(self) -> Dict[str, List[float]]:
        """Get learning trajectory metrics."""
        return {
            'complexity_curve': self.complexity_curve,
            'novelty_history': self.novelty_history,
            'exploration_rate': [
                1.0 * (self.config.exploration_decay ** i)
                for i in range(len(self.complexity_curve))
            ]
        }
        
    def adapt_parameters(self, performance_metrics: Dict[str, float]):
        """Adapt module parameters based on performance."""
        # Adjust novelty threshold based on performance
        if 'task_success_rate' in performance_metrics:
            success_rate = performance_metrics['task_success_rate']
            self.config.novelty_threshold += (
                self.config.adaptation_rate * 
                (success_rate - 0.5)  # Adjust towards optimal success rate
            )
            self.config.novelty_threshold = max(0.1, min(0.9, self.config.novelty_threshold))
            
        # Adjust curiosity weight based on learning progress
        if len(self.complexity_curve) > 10:
            recent_progress = np.mean(np.diff(self.complexity_curve[-10:]))
            if recent_progress < 0.01:  # Learning plateau
                self.config.curiosity_weight *= 1.1  # Increase exploration
            else:
                self.config.curiosity_weight *= 0.95  # Reduce exploration
            
            self.config.curiosity_weight = max(0.1, min(0.9, self.config.curiosity_weight))
    
    def integrate_with_trainer(self, trainer):
        """Integrate emergent behavior module with advanced trainer."""
        self.trainer = trainer
        self.trainer.emergent_behavior = self

    def update_forward(self, current_state: torch.Tensor, action_history: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Updated forward method to include new advancements.
        
        Args:
            current_state: Current hidden state tensor
            action_history: History of previous actions
            
        Returns:
            Tuple of (generated task embedding, metrics dictionary)
        """
        # Calculate novelty score
        novelty_score = self.detect_novelty(current_state)
        
        # Update experience buffer
        if len(self.experience_buffer) >= self.config.self_play_buffer_size:
            self.experience_buffer.pop(0)
        self.experience_buffer.append(current_state.detach())
        
        # Generate new task based on current state and history
        task_context = self._create_task_context(current_state, action_history)
        generated_task = self.task_generator(task_context)
        
        # Update exploration rate
        self.current_exploration_rate *= self.config.exploration_decay
        
        # Calculate complexity metrics
        complexity_score = self._calculate_complexity(current_state)
        self.complexity_curve.append(complexity_score)
        
        # Calculate intrinsic reward
        intrinsic_reward = self.generate_intrinsic_reward(current_state, action_history[-1] if action_history else torch.zeros_like(current_state), generated_task)
        
        metrics = {
            'novelty_score': novelty_score.item(),
            'complexity_score': complexity_score,
            'exploration_rate': self.current_exploration_rate,
            'buffer_size': len(self.experience_buffer),
            'intrinsic_reward': intrinsic_reward.item()
        }
        
        return generated_task, metrics
