"""Training visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """Visualizes training metrics and statistics."""
    
    def __init__(self):
        """Initialize the training visualizer."""
        plt.style.use('seaborn')
        self.metrics_history: Dict[str, List[float]] = {}
        self.steps: List[int] = []
        
    def add_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Add metrics for a training step.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            step (int): Current training step
        """
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append(value)
            
        self.steps.append(step)
        
    def plot_metric(
        self,
        metric_name: str,
        figsize: Tuple[int, int] = (10, 6),
        title: Optional[str] = None,
        xlabel: str = "Step",
        ylabel: Optional[str] = None,
        window_size: Optional[int] = None
    ) -> None:
        """Plot a single training metric over time.
        
        Args:
            metric_name (str): Name of metric to plot
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (Optional[str], optional): Plot title. Defaults to None.
            xlabel (str, optional): X-axis label. Defaults to "Step".
            ylabel (Optional[str], optional): Y-axis label. Defaults to None.
            window_size (Optional[int], optional): Window size for smoothing. Defaults to None.
        """
        if metric_name not in self.metrics_history:
            logger.warning(f"Metric {metric_name} not found in history")
            return
            
        plt.figure(figsize=figsize)
        values = self.metrics_history[metric_name]
        
        if window_size:
            values = np.convolve(values, 
                               np.ones(window_size)/window_size,
                               mode='valid')
            steps = self.steps[window_size-1:]
        else:
            steps = self.steps
            
        plt.plot(steps, values)
        plt.title(title or f"{metric_name} over time")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel or metric_name)
        plt.grid(True)
        
    def plot_metrics_comparison(
        self,
        metric_names: List[str],
        figsize: Tuple[int, int] = (12, 6),
        title: str = "Metrics Comparison",
        xlabel: str = "Step",
        normalize: bool = False
    ) -> None:
        """Plot multiple metrics for comparison.
        
        Args:
            metric_names (List[str]): List of metric names to compare
            figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
            title (str, optional): Plot title. Defaults to "Metrics Comparison".
            xlabel (str, optional): X-axis label. Defaults to "Step".
            normalize (bool, optional): Whether to normalize values. Defaults to False.
        """
        plt.figure(figsize=figsize)
        
        for name in metric_names:
            if name not in self.metrics_history:
                logger.warning(f"Metric {name} not found in history")
                continue
                
            values = np.array(self.metrics_history[name])
            if normalize:
                values = (values - np.min(values)) / (np.max(values) - np.min(values))
                
            plt.plot(self.steps, values, label=name)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Normalized Value" if normalize else "Value")
        plt.legend()
        plt.grid(True)
        
    def plot_learning_rate(
        self,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Learning Rate Schedule"
    ) -> None:
        """Plot the learning rate schedule.
        
        Args:
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            title (str, optional): Plot title. Defaults to "Learning Rate Schedule".
        """
        if 'learning_rate' not in self.metrics_history:
            logger.warning("Learning rate history not found")
            return
            
        plt.figure(figsize=figsize)
        plt.plot(self.steps, self.metrics_history['learning_rate'])
        plt.title(title)
        plt.xlabel("Step")
        plt.ylabel("Learning Rate")
        plt.yscale('log')
        plt.grid(True)
        
    def plot_loss_distribution(
        self,
        loss_name: str = 'loss',
        figsize: Tuple[int, int] = (10, 6),
        bins: int = 50
    ) -> None:
        """Plot the distribution of loss values.
        
        Args:
            loss_name (str, optional): Name of loss metric. Defaults to 'loss'.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
            bins (int, optional): Number of histogram bins. Defaults to 50.
        """
        if loss_name not in self.metrics_history:
            logger.warning(f"Loss metric {loss_name} not found in history")
            return
            
        plt.figure(figsize=figsize)
        sns.histplot(self.metrics_history[loss_name], bins=bins)
        plt.title(f"{loss_name.capitalize()} Distribution")
        plt.xlabel(loss_name.capitalize())
        plt.ylabel("Count")
