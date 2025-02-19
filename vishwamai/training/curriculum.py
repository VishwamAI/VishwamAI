"""
Curriculum Learning implementation for VishwamAI.

This module provides functionality for curriculum learning, allowing the model
to learn from progressively more difficult examples during training.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Iterator, Tuple
from dataclasses import dataclass
import numpy as np

from vishwamai.utils.config import ModelConfig, TrainingConfig

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    difficulty_metric: str = 'length'  # length, complexity, or custom
    start_difficulty: float = 0.2
    end_difficulty: float = 1.0
    difficulty_step: float = 0.1
    steps_per_difficulty: int = 1000
    scoring_function: Optional[str] = None
    custom_metric: Optional[callable] = None

class CurriculumLearning:
    """
    Implements curriculum learning strategies.
    """
    
    def __init__(
        self,
        config: ModelConfig,
        training_config: TrainingConfig,
        curriculum_config: Optional[CurriculumConfig] = None
    ):
        self.config = config
        self.training_config = training_config
        self.curriculum_config = curriculum_config or CurriculumConfig()
        
        # Initialize difficulty tracking
        self.current_difficulty = self.curriculum_config.start_difficulty
        self.steps_at_difficulty = 0
        self.total_steps = 0
        
        # Set up difficulty metric function
        self.difficulty_fn = self._get_difficulty_function()
        
        # Cache for sample difficulties
        self.sample_difficulties: Dict[int, float] = {}
        
    def _get_difficulty_function(self) -> callable:
        """Get the appropriate difficulty scoring function."""
        if self.curriculum_config.difficulty_metric == 'length':
            return self._length_difficulty
        elif self.curriculum_config.difficulty_metric == 'complexity':
            return self._complexity_difficulty
        elif self.curriculum_config.difficulty_metric == 'custom':
            if self.curriculum_config.custom_metric is None:
                raise ValueError("Custom metric function not provided")
            return self.curriculum_config.custom_metric
        else:
            raise ValueError(
                f"Unknown difficulty metric: {self.curriculum_config.difficulty_metric}"
            )
            
    def _length_difficulty(self, sample: Dict[str, torch.Tensor]) -> float:
        """Compute difficulty based on sequence length."""
        return len(sample['input_ids']) / self.config.max_position_embeddings
        
    def _complexity_difficulty(self, sample: Dict[str, torch.Tensor]) -> float:
        """
        Compute difficulty based on sequence complexity.
        Uses entropy and structural metrics.
        """
        input_ids = sample['input_ids']
        
        # Compute token entropy
        unique, counts = np.unique(input_ids, return_counts=True)
        probs = counts / len(input_ids)
        entropy = -np.sum(probs * np.log2(probs))
        
        # Normalize entropy
        max_entropy = np.log2(len(unique))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Consider sequence length as well
        length_factor = len(input_ids) / self.config.max_position_embeddings
        
        # Combine metrics
        complexity = 0.7 * normalized_entropy + 0.3 * length_factor
        return complexity
        
    def get_difficulty(self, sample: Dict[str, torch.Tensor]) -> float:
        """Get difficulty score for a sample."""
        # Check cache first
        sample_id = hash(tuple(sample['input_ids'].tolist()))
        if sample_id in self.sample_difficulties:
            return self.sample_difficulties[sample_id]
            
        # Compute difficulty
        difficulty = self.difficulty_fn(sample)
        
        # Cache result
        self.sample_difficulties[sample_id] = difficulty
        
        return difficulty
        
    def should_increase_difficulty(self) -> bool:
        """Determine if difficulty should be increased."""
        if self.steps_at_difficulty >= self.curriculum_config.steps_per_difficulty:
            if self.current_difficulty < self.curriculum_config.end_difficulty:
                return True
        return False
        
    def increase_difficulty(self):
        """Increase the current difficulty level."""
        self.current_difficulty = min(
            self.current_difficulty + self.curriculum_config.difficulty_step,
            self.curriculum_config.end_difficulty
        )
        self.steps_at_difficulty = 0
        
    def filter_samples(
        self,
        samples: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Filter samples based on current difficulty."""
        filtered = []
        for sample in samples:
            difficulty = self.get_difficulty(sample)
            if difficulty <= self.current_difficulty:
                filtered.append(sample)
        return filtered
        
    def get_samples(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Get samples for current curriculum stage.
        
        Args:
            dataloader: Original dataloader
            epoch: Current training epoch
            
        Returns:
            Iterator over filtered samples
        """
        for batch in dataloader:
            # Update steps
            self.total_steps += 1
            self.steps_at_difficulty += 1
            
            # Check if difficulty should be increased
            if self.should_increase_difficulty():
                self.increase_difficulty()
                
            # Filter batch based on difficulty
            if isinstance(batch, dict):
                difficulties = [
                    self.get_difficulty({'input_ids': ids})
                    for ids in batch['input_ids']
                ]
                mask = [d <= self.current_difficulty for d in difficulties]
                filtered_batch = {
                    k: v[mask] for k, v in batch.items()
                    if isinstance(v, torch.Tensor)
                }
                
                if len(filtered_batch['input_ids']) > 0:
                    yield filtered_batch
            else:
                # Handle non-dictionary batch types if needed
                yield batch
                
    def get_curriculum_stats(self) -> Dict[str, float]:
        """Get current curriculum learning statistics."""
        return {
            'current_difficulty': self.current_difficulty,
            'steps_at_difficulty': self.steps_at_difficulty,
            'total_steps': self.total_steps,
            'progress': (
                self.current_difficulty - self.curriculum_config.start_difficulty
            ) / (
                self.curriculum_config.end_difficulty - self.curriculum_config.start_difficulty
            )
        }
        
    def save_state(self) -> Dict[str, Any]:
        """Save curriculum learning state."""
        return {
            'current_difficulty': self.current_difficulty,
            'steps_at_difficulty': self.steps_at_difficulty,
            'total_steps': self.total_steps,
            'sample_difficulties': self.sample_difficulties,
            'config': vars(self.curriculum_config)
        }
        
    def load_state(self, state: Dict[str, Any]):
        """Load curriculum learning state."""
        self.current_difficulty = state['current_difficulty']
        self.steps_at_difficulty = state['steps_at_difficulty']
        self.total_steps = state['total_steps']
        self.sample_difficulties = state['sample_difficulties']
        
        # Update config if provided
        if 'config' in state:
            self.curriculum_config = CurriculumConfig(**state['config'])
