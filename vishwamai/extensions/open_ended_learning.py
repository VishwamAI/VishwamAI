import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto

class ExplorationStrategy(Enum):
    RANDOM = auto()
    NOVELTY = auto()
    CURIOSITY = auto()
    COMPLEXITY = auto()
    DIVERSITY = auto()

@dataclass
class OpenEndedConfig:
    """Configuration for open-ended learning."""
    min_novelty_threshold: float = 0.3
    max_complexity: float = 5.0
    exploration_rate: float = 0.1
    diversity_weight: float = 0.5
    memory_capacity: int = 10000
    min_task_count: int = 10
    max_task_count: int = 1000
    evolution_rate: float = 0.01
    generation_temperature: float = 0.8
    adaptation_rate: float = 0.1
    task_difficulty_range: Tuple[float, float] = (0.1, 1.0)
    strategy: ExplorationStrategy = ExplorationStrategy.NOVELTY

class OpenEndedLearning(nn.Module):
    """
    Module for implementing open-ended learning and continuous evolution.
    
    This module enables:
    - Continuous task generation
    - Novelty-driven exploration
    - Complexity growth tracking
    - Task space evolution
    - Adaptive difficulty
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[OpenEndedConfig] = None
    ):
        super().__init__()
        self.config = config or OpenEndedConfig()
        self.hidden_dim = hidden_dim
        
        # Task generation networks
        self.task_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.complexity_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.novelty_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # State tracking
        self.task_history = []
        self.complexity_history = []
        self.novelty_history = []
        self.task_space = set()
        self.current_generation = 0
        
        # Task evolution tracking
        self.evolution_metrics = {
            'task_complexity': [],
            'diversity': [],
            'success_rate': []
        }
        
    def forward(
        self,
        current_state: torch.Tensor,
        performance_history: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate new tasks and learning opportunities.
        
        Args:
            current_state: Current model state
            performance_history: Optional history of performance metrics
            
        Returns:
            Tuple of (generated task, metrics dictionary)
        """
        batch_size = current_state.size(0)
        
        # Generate task embedding based on current state and history
        task_context = self._create_task_context(current_state)
        generated_task = self.task_generator(task_context)
        
        # Estimate task complexity
        complexity = self.complexity_estimator(generated_task)
        
        # Measure novelty
        novelty_score = self._calculate_novelty(generated_task)
        
        # Evolve task space if needed
        if self._should_evolve():
            self._evolve_task_space()
        
        # Track metrics
        self._update_metrics(generated_task, complexity, novelty_score)
        
        metrics = {
            'complexity': complexity.mean().item(),
            'novelty': novelty_score.mean().item(),
            'task_space_size': len(self.task_space),
            'generation': self.current_generation,
            'evolution_metrics': self.get_evolution_metrics()
        }
        
        return generated_task, metrics
    
    def _create_task_context(self, current_state: torch.Tensor) -> torch.Tensor:
        """Create context for task generation."""
        # Combine current state with history summary
        if self.task_history:
            history_tensor = torch.stack(
                [task.detach() for task in self.task_history[-10:]],
                dim=0
            ).mean(dim=0)
        else:
            history_tensor = torch.zeros_like(current_state)
            
        return torch.cat([current_state, history_tensor], dim=-1)
        
    def _calculate_novelty(self, task: torch.Tensor) -> torch.Tensor:
        """Calculate novelty score for a task."""
        if not self.task_history:
            return torch.ones(task.size(0), 1, device=task.device)
            
        # Compare with recent tasks
        similarities = []
        for past_task in self.task_history[-50:]:
            sim = F.cosine_similarity(
                task.flatten(1),
                past_task.flatten(1).unsqueeze(0).expand(task.size(0), -1),
                dim=1
            )
            similarities.append(sim)
            
        similarity = torch.stack(similarities, dim=1).mean(dim=1)
        novelty = 1.0 - similarity
        
        return novelty.unsqueeze(1)
    
    def _should_evolve(self) -> bool:
        """Determine if task space should evolve."""
        if len(self.task_history) < self.config.min_task_count:
            return False
            
        # Check various evolution triggers
        complexity_plateau = self._check_complexity_plateau()
        diversity_low = self._check_diversity()
        performance_good = self._check_performance()
        
        return (complexity_plateau or diversity_low) and performance_good
    
    def _check_complexity_plateau(self) -> bool:
        """Check if complexity growth has plateaued."""
        if len(self.complexity_history) < 50:
            return False
            
        recent_complexity = np.array(self.complexity_history[-50:])
        slope = np.polyfit(np.arange(50), recent_complexity, 1)[0]
        return slope < 0.001
    
    def _check_diversity(self) -> bool:
        """Check if task diversity is too low."""
        if len(self.task_history) < 50:
            return False
            
        recent_tasks = torch.stack(self.task_history[-50:])
        distances = torch.cdist(recent_tasks, recent_tasks)
        diversity = distances.mean().item()
        
        return diversity < self.config.diversity_weight
    
    def _check_performance(self) -> bool:
        """Check if performance is good enough for evolution."""
        if not hasattr(self, 'performance_history') or not self.performance_history:
            return True
            
        recent_performance = np.mean(self.performance_history[-20:])
        return recent_performance > 0.7
    
    def _evolve_task_space(self):
        """Evolve the task space for continued learning."""
        self.current_generation += 1
        
        # Generate new tasks through mutation and combination
        new_tasks = []
        
        # Mutation
        for task in self.task_history[-10:]:
            mutated = self._mutate_task(task)
            new_tasks.append(mutated)
            
        # Combination
        for _ in range(5):
            if len(self.task_history) >= 2:
                task1, task2 = np.random.choice(self.task_history[-20:], size=2)
                combined = self._combine_tasks(task1, task2)
                new_tasks.append(combined)
                
        # Add successful variations to task space
        for task in new_tasks:
            task_hash = self._hash_task(task)
            self.task_space.add(task_hash)
            
        # Prune task space if too large
        if len(self.task_space) > self.config.max_task_count:
            self._prune_task_space()
    
    def _mutate_task(self, task: torch.Tensor) -> torch.Tensor:
        """Apply mutation to a task."""
        noise = torch.randn_like(task) * self.config.evolution_rate
        return task + noise
    
    def _combine_tasks(self, task1: torch.Tensor, task2: torch.Tensor) -> torch.Tensor:
        """Combine two tasks to create a new one."""
        alpha = np.random.beta(0.5, 0.5)
        return alpha * task1 + (1 - alpha) * task2
    
    def _hash_task(self, task: torch.Tensor) -> str:
        """Create a hash for a task for uniqueness tracking."""
        return str(hash(task.detach().cpu().numpy().tobytes()))
    
    def _prune_task_space(self):
        """Prune task space to maintain diversity and manageable size."""
        if len(self.task_space) <= self.config.min_task_count:
            return
            
        # Remove least novel or successful tasks
        tasks = list(self.task_space)
        novelties = [self._calculate_novelty(torch.tensor(task)).item() for task in tasks]
        
        # Keep most novel tasks
        keep_indices = np.argsort(novelties)[-self.config.max_task_count:]
        self.task_space = {tasks[i] for i in keep_indices}
    
    def _update_metrics(
        self,
        task: torch.Tensor,
        complexity: torch.Tensor,
        novelty: torch.Tensor
    ):
        """Update tracking metrics."""
        self.task_history.append(task.detach())
        self.complexity_history.append(complexity.mean().item())
        self.novelty_history.append(novelty.mean().item())
        
        # Maintain history size
        if len(self.task_history) > self.config.memory_capacity:
            self.task_history = self.task_history[-self.config.memory_capacity:]
            self.complexity_history = self.complexity_history[-self.config.memory_capacity:]
            self.novelty_history = self.novelty_history[-self.config.memory_capacity:]
    
    def get_evolution_metrics(self) -> Dict[str, List[float]]:
        """Get metrics about task space evolution."""
        recent_window = 100
        
        metrics = {
            'complexity_trend': self.complexity_history[-recent_window:],
            'novelty_trend': self.novelty_history[-recent_window:],
            'task_space_size': [len(self.task_space)],
            'generation': self.current_generation
        }
        
        # Calculate diversity
        if len(self.task_history) >= 2:
            recent_tasks = torch.stack(self.task_history[-recent_window:])
            distances = torch.cdist(recent_tasks, recent_tasks)
            diversity = distances.mean().item()
            metrics['diversity'] = diversity
            
        return metrics
    
    def adapt_difficulty(self, performance: float) -> None:
        """Adapt task generation difficulty based on performance."""
        current_range = self.config.task_difficulty_range
        
        if performance > 0.8:  # Too easy
            new_min = min(current_range[0] + self.config.adaptation_rate, current_range[1])
            new_max = min(current_range[1] + self.config.adaptation_rate, 1.0)
            self.config.task_difficulty_range = (new_min, new_max)
            
        elif performance < 0.4:  # Too hard
            new_min = max(current_range[0] - self.config.adaptation_rate, 0.1)
            new_max = max(current_range[1] - self.config.adaptation_rate, new_min + 0.1)
            self.config.task_difficulty_range = (new_min, new_max)
