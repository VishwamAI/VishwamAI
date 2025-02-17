import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import math

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    # Curriculum progression settings
    min_sequence_length: int = 32
    max_sequence_length: int = 512
    min_vocab_complexity: float = 0.3
    max_vocab_complexity: float = 1.0
    min_reasoning_steps: int = 1
    max_reasoning_steps: int = 8
    
    # Pacing function parameters
    pacing_function: str = 'root'  # Options: 'linear', 'root', 'step'
    total_curriculum_steps: int = 10000
    step_sizes: List[float] = None  # Required only for 'step' pacing
    
    # Adaptive pacing settings
    performance_threshold: float = 0.8
    min_samples_before_advance: int = 100
    smoothing_factor: float = 0.95

class CurriculumScheduler:
    """
    Manages curriculum learning progression based on training dynamics.
    
    This scheduler dynamically adjusts task difficulty based on model performance
    and training progress, implementing various pacing functions and adaptive
    progression strategies.
    """
    
    def __init__(self, config: CurriculumConfig):
        """
        Initialize curriculum scheduler.
        
        Args:
            config: Curriculum learning configuration
        """
        self.config = config
        self.current_step = 0
        self.current_difficulty = 0.0
        self.performance_history = []
        self.samples_at_level = 0
        
        # Validate and set pacing function
        if config.pacing_function not in ['linear', 'root', 'step']:
            raise ValueError(f"Unknown pacing function: {config.pacing_function}")
            
        if config.pacing_function == 'step' and not config.step_sizes:
            raise ValueError("step_sizes required for step pacing function")
            
        # Initialize difficulty metrics
        self.sequence_length_range = config.max_sequence_length - config.min_sequence_length
        self.vocab_complexity_range = config.max_vocab_complexity - config.min_vocab_complexity
        self.reasoning_steps_range = config.max_reasoning_steps - config.min_reasoning_steps
        
    def get_current_difficulty(self) -> float:
        """Get current curriculum difficulty (0 to 1)."""
        if self.config.pacing_function == 'linear':
            return min(1.0, self.current_step / self.config.total_curriculum_steps)
            
        elif self.config.pacing_function == 'root':
            return min(1.0, math.sqrt(self.current_step / self.config.total_curriculum_steps))
            
        else:  # step
            step_idx = min(
                len(self.config.step_sizes) - 1,
                self.current_step // (self.config.total_curriculum_steps // len(self.config.step_sizes))
            )
            return self.config.step_sizes[step_idx]
    
    def get_curriculum_params(self) -> Dict[str, Any]:
        """
        Get current curriculum parameters.
        
        Returns:
            Dictionary containing curriculum parameters for the current difficulty level
        """
        difficulty = self.get_current_difficulty()
        
        return {
            'sequence_length': int(
                self.config.min_sequence_length +
                difficulty * self.sequence_length_range
            ),
            'vocab_complexity': float(
                self.config.min_vocab_complexity +
                difficulty * self.vocab_complexity_range
            ),
            'reasoning_steps': int(
                self.config.min_reasoning_steps +
                difficulty * self.reasoning_steps_range
            )
        }
    
    def update(self, performance_metrics: Dict[str, float]) -> bool:
        """
        Update curriculum state based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
            
        Returns:
            Boolean indicating whether curriculum level has advanced
        """
        # Track performance
        avg_performance = np.mean(list(performance_metrics.values()))
        self.performance_history.append(avg_performance)
        self.samples_at_level += 1
        
        # Check if we should advance difficulty
        advanced = False
        if self.samples_at_level >= self.config.min_samples_before_advance:
            recent_performance = np.mean(self.performance_history[-self.config.min_samples_before_advance:])
            
            if recent_performance >= self.config.performance_threshold:
                self.current_step += 1
                self.samples_at_level = 0
                advanced = True
                
                # Clear old performance history when advancing
                self.performance_history = self.performance_history[-self.config.min_samples_before_advance:]
        
        return advanced
    
    def estimate_task_difficulty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate task difficulty from input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (overall difficulty score, difficulty metrics dictionary)
        """
        metrics = {}
        
        # Sequence length difficulty
        seq_len = input_ids.size(1)
        seq_diff = (seq_len - self.config.min_sequence_length) / self.sequence_length_range
        metrics['sequence_length'] = min(1.0, max(0.0, seq_diff))
        
        # Vocabulary complexity
        if attention_mask is not None:
            vocab_size = input_ids.max().item() + 1
            vocab_diff = (
                torch.unique(input_ids[attention_mask.bool()]).size(0) / vocab_size
            )
            metrics['vocabulary'] = min(1.0, max(0.0, vocab_diff))
        else:
            vocab_diff = (torch.unique(input_ids).size(0) / (input_ids.max().item() + 1))
            metrics['vocabulary'] = min(1.0, max(0.0, vocab_diff))
        
        # Reasoning steps difficulty
        reasoning_steps = input_ids.size(1) // 10  # Example heuristic
        reasoning_diff = (reasoning_steps - self.config.min_reasoning_steps) / self.reasoning_steps_range
        metrics['reasoning_steps'] = min(1.0, max(0.0, reasoning_diff))
        
        # Overall difficulty score (weighted average)
        weights = {'sequence_length': 0.4, 'vocabulary': 0.3, 'reasoning_steps': 0.3}
        difficulty_score = sum(
            metric * weights[name] for name, metric in metrics.items()
        )
        
        return difficulty_score, metrics
    
    def state_dict(self) -> Dict[str, Any]:
        """Get curriculum scheduler state."""
        return {
            'current_step': self.current_step,
            'current_difficulty': self.current_difficulty,
            'performance_history': self.performance_history,
            'samples_at_level': self.samples_at_level
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load curriculum scheduler state."""
        self.current_step = state_dict['current_step']
        self.current_difficulty = state_dict['current_difficulty']
        self.performance_history = state_dict['performance_history']
        self.samples_at_level = state_dict['samples_at_level']
