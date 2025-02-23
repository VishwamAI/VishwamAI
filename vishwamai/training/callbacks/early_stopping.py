"""Early stopping callback for training monitoring."""

from typing import Dict, Optional, List
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping.
    
    Attributes:
        monitor (str): Metric to monitor
        mode (str): One of {min, max} for monitoring
        patience (int): Number of checks with no improvement after which training will stop
        min_delta (float): Minimum change in monitored quantity to qualify as an improvement
        check_interval (int): Number of epochs between checks
    """
    monitor: str = "val_loss"
    mode: str = "min"
    patience: int = 3
    min_delta: float = 0.0
    check_interval: int = 1

class EarlyStopping:
    """Callback for early stopping based on monitored metric.
    
    This callback tracks a specified metric and stops training when there's no
    improvement for a configured number of checks.
    
    Args:
        config (EarlyStoppingConfig): Early stopping configuration
    """
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.best_score: Optional[float] = None
        self.counter = 0
        self.stopped_epoch = 0
        self.history: List[float] = []
        
        if config.mode not in ["min", "max"]:
            raise ValueError(f"mode {config.mode} is not supported")
            
        if config.check_interval < 1:
            raise ValueError(
                f"check_interval must be >= 1, got {config.check_interval}"
            )
            
        self._init_is_better()
        
    def _init_is_better(self) -> None:
        """Initialize comparison function based on mode."""
        if self.config.mode == "min":
            self.is_better = lambda x, y: x < y - self.config.min_delta
        else:
            self.is_better = lambda x, y: x > y + self.config.min_delta
            
    def state_dict(self) -> Dict:
        """Get current state for checkpointing.
        
        Returns:
            Dict: State dictionary
        """
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'stopped_epoch': self.stopped_epoch,
            'history': self.history
        }
        
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state from checkpoint.
        
        Args:
            state_dict (Dict): State dictionary
        """
        self.best_score = state_dict['best_score']
        self.counter = state_dict['counter']
        self.stopped_epoch = state_dict['stopped_epoch']
        self.history = state_dict['history']
        
    def on_validation_end(
        self,
        metrics: Dict[str, float],
        epoch: int
    ) -> bool:
        """Called after validation phase.
        
        Args:
            metrics (Dict[str, float]): Validation metrics
            epoch (int): Current epoch
            
        Returns:
            bool: True if training should stop
        """
        if epoch % self.config.check_interval != 0:
            return False
            
        monitor_val = metrics.get(self.config.monitor)
        if monitor_val is None:
            logger.warning(f"Metric {self.config.monitor} not found in metrics")
            return False
            
        self.history.append(monitor_val)
        
        if self.best_score is None:
            # First epoch
            self.best_score = monitor_val
            return False
            
        if self.is_better(monitor_val, self.best_score):
            # Score improved
            self.best_score = monitor_val
            self.counter = 0
            return False
            
        # Score did not improve
        self.counter += 1
        if self.counter >= self.config.patience:
            self.stopped_epoch = epoch
            logger.info(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best {self.config.monitor}: {self.best_score:.4f}"
            )
            return True
            
        return False
        
    def get_trend(self, window: int = 5) -> Optional[float]:
        """Get trend of monitored metric over recent checks.
        
        Args:
            window (int, optional): Window size for trend calculation. Defaults to 5.
            
        Returns:
            Optional[float]: Trend value (positive = increasing, negative = decreasing)
        """
        if len(self.history) < window:
            return None
            
        recent = np.array(self.history[-window:])
        x = np.arange(window)
        slope, _ = np.polyfit(x, recent, 1)
        
        return slope
        
    def get_best_score(self) -> Optional[float]:
        """Get best score achieved.
        
        Returns:
            Optional[float]: Best score
        """
        return self.best_score
        
    def get_wait_epochs(self) -> int:
        """Get number of epochs without improvement.
        
        Returns:
            int: Number of epochs without improvement
        """
        return self.counter * self.config.check_interval
        
    def reset(self) -> None:
        """Reset internal states."""
        self.best_score = None
        self.counter = 0
        self.stopped_epoch = 0
        self.history.clear()
