"""Early stopping callbacks for training."""
from typing import Optional, Dict, List
import torch
import numpy as np

class EarlyStopping:
    """Early stopping callback to prevent overfitting.
    
    Args:
        monitor: Metric to monitor
        min_delta: Minimum change in monitored value
        patience: Epochs without improvement before stopping
        mode: One of ['min', 'max']
        strict: Whether to be strict about metric improvements
        verbose: Whether to print stopping info
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 3,
        mode: str = 'min',
        strict: bool = False,
        verbose: bool = True
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.strict = strict
        self.verbose = verbose
        
        # Initialize tracking
        self.wait = 0
        self.stopped_epoch = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.should_stop = False
        
    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> bool:
        """Check if training should stop.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            
        Returns:
            Whether training should stop
        """
        if self.monitor not in metrics:
            return False
            
        current = metrics[self.monitor]
        
        if self.mode == 'min':
            score = current
        else:
            score = -current
            
        if self._is_improvement(score, self.best_score):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.should_stop = True
                if self.verbose:
                    print(f'Early stopping triggered at epoch {epoch}')
                return True
                
        return False
        
    def _is_improvement(
        self,
        current: float,
        best: float
    ) -> bool:
        """Check if current score is an improvement.
        
        Args:
            current: Current score
            best: Best score so far
            
        Returns:
            Whether current score is an improvement
        """
        if self.strict:
            return current < (best - self.min_delta)
        return current <= (best - self.min_delta)

class ExpertEarlyStopping(EarlyStopping):
    """Early stopping for expert-parallel training.
    
    Adds expert monitoring and handles distributed metrics.
    
    Args:
        monitor: Metric to monitor
        min_delta: Minimum change in monitored value
        patience: Epochs without improvement before stopping
        mode: One of ['min', 'max']
        expert_metrics: Expert-specific metrics to monitor
        expert_patience: Patience for expert metrics
        expert_threshold: Threshold for expert metrics
        strict: Whether to be strict about metric improvements
        verbose: Whether to print stopping info
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        min_delta: float = 0.0,
        patience: int = 3,
        mode: str = 'min',
        expert_metrics: Optional[List[str]] = None,
        expert_patience: int = 2,
        expert_threshold: float = 0.1,
        strict: bool = False,
        verbose: bool = True
    ):
        super().__init__(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            mode=mode,
            strict=strict,
            verbose=verbose
        )
        self.expert_metrics = expert_metrics or ['expert_load_std', 'expert_usage']
        self.expert_patience = expert_patience
        self.expert_threshold = expert_threshold
        
        # Expert tracking
        self.expert_wait = {metric: 0 for metric in self.expert_metrics}
        self.best_expert_scores = {
            metric: float('inf') if mode == 'min' else float('-inf')
            for metric in self.expert_metrics
        }
        
    def __call__(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> bool:
        """Check both overall and expert-specific stopping conditions.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            
        Returns:
            Whether training should stop
        """
        # Check main stopping condition
        should_stop = super().__call__(epoch, metrics)
        if should_stop:
            return True
            
        # Check expert metrics
        for metric in self.expert_metrics:
            if metric not in metrics:
                continue
                
            current = metrics[metric]
            if self._check_expert_metric(current, metric):
                self.expert_wait[metric] += 1
                if self.expert_wait[metric] >= self.expert_patience:
                    if self.verbose:
                        print(f'Expert early stopping triggered at epoch {epoch}'
                              f' due to {metric}')
                    self.stopped_epoch = epoch
                    self.should_stop = True
                    return True
            else:
                self.expert_wait[metric] = 0
                self.best_expert_scores[metric] = current
                
        return False
        
    def _check_expert_metric(
        self,
        current: float,
        metric: str
    ) -> bool:
        """Check if expert metric exceeds threshold.
        
        Args:
            current: Current metric value
            metric: Metric name
            
        Returns:
            Whether metric exceeds threshold
        """
        # For load std, check if variance too high
        if 'load_std' in metric:
            return current > self.expert_threshold
            
        # For usage, check if too imbalanced
        if 'usage' in metric:
            return current < self.expert_threshold
            
        # Default to standard comparison
        if self.mode == 'min':
            return current > (self.best_expert_scores[metric] + self.min_delta)
        return current < (self.best_expert_scores[metric] - self.min_delta)
        
    def reset(self):
        """Reset tracking state."""
        super().reset()
        self.expert_wait = {metric: 0 for metric in self.expert_metrics}
        self.best_expert_scores = {
            metric: float('inf') if self.mode == 'min' else float('-inf')
            for metric in self.expert_metrics
        }
