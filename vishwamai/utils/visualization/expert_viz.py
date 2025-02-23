"""Expert routing visualization utilities."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class ExpertRoutingVisualizer:
    """Visualizes MoE expert routing patterns and statistics."""
    
    def __init__(self):
        """Initialize the expert routing visualizer."""
        plt.style.use('seaborn')
        
    def plot_routing_distribution(
        self,
        routing_probs: Union[torch.Tensor, np.ndarray],
        expert_ids: Optional[List[int]] = None,
        figsize: Tuple[int, int] = (10, 6),
        title: str = "Token Routing Distribution"
    ) -> None:
        """Plot the distribution of tokens across experts.
        
        Args:
            routing_probs: Routing probabilities [num_tokens, num_experts]
            expert_ids (Optional[List[int]]): List of expert IDs for labels
            figsize (Tuple[int, int]): Figure size. Defaults to (10, 6)
            title (str): Plot title
        """
        if isinstance(routing_probs, torch.Tensor):
            routing_probs = routing_probs.detach().cpu().numpy()
            
        num_experts = routing_probs.shape[1]
        expert_ids = expert_ids or list(range(num_experts))
        
        # Calculate routing distribution
        expert_loads = routing_probs.sum(axis=0)
        expert_loads = expert_loads / expert_loads.sum()  # Normalize
        
        plt.figure(figsize=figsize)
        plt.bar(range(num_experts), expert_loads)
        plt.xticks(range(num_experts), [f"Expert {i}" for i in expert_ids])
        plt.title(title)
        plt.xlabel("Expert")
        plt.ylabel("Fraction of Tokens")
        plt.grid(True, alpha=0.3)
        
    def plot_expert_usage_heatmap(
        self,
        usage_matrix: Union[torch.Tensor, np.ndarray],
        layer_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'YlOrRd'
    ) -> None:
        """Plot expert usage patterns as a heatmap.
        
        Args:
            usage_matrix: Expert usage [num_tokens, num_experts]
            layer_idx (Optional[int]): Layer index for title
            figsize (Tuple[int, int]): Figure size. Defaults to (12, 8)
            cmap (str): Colormap for heatmap. Defaults to 'YlOrRd'
        """
        if isinstance(usage_matrix, torch.Tensor):
            usage_matrix = usage_matrix.detach().cpu().numpy()
            
        plt.figure(figsize=figsize)
        sns.heatmap(
            usage_matrix,
            cmap=cmap,
            cbar_kws={'label': 'Usage Intensity'},
            xticklabels=[f"Expert {i}" for i in range(usage_matrix.shape[1])],
            yticklabels=False
        )
        
        title = "Expert Usage Patterns"
        if layer_idx is not None:
            title += f" (Layer {layer_idx})"
            
        plt.title(title)
        plt.xlabel("Expert")
        plt.ylabel("Token Position")
        
    def plot_balance_metrics(
        self,
        balance_metrics: Dict[str, List[float]],
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Plot expert balance metrics over training steps.
        
        Args:
            balance_metrics (Dict[str, List[float]]): Dictionary of balance metrics
            figsize (Tuple[int, int]): Figure size. Defaults to (12, 6)
        """
        plt.figure(figsize=figsize)
        
        for metric_name, values in balance_metrics.items():
            plt.plot(values, label=metric_name)
            
        plt.title("Expert Balance Metrics Over Time")
        plt.xlabel("Training Step")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        
    def plot_expert_specialization(
        self,
        token_counts: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        expert_idx: int,
        top_k: int = 20,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Visualize token specialization for a specific expert.
        
        Args:
            token_counts: Token count matrix [vocab_size, num_experts]
            tokens (List[str]): List of token strings
            expert_idx (int): Expert index to visualize
            top_k (int): Number of top tokens to show. Defaults to 20
            figsize (Tuple[int, int]): Figure size. Defaults to (12, 6)
        """
        if isinstance(token_counts, torch.Tensor):
            token_counts = token_counts.detach().cpu().numpy()
            
        expert_counts = token_counts[:, expert_idx]
        top_indices = np.argsort(-expert_counts)[:top_k]
        
        plt.figure(figsize=figsize)
        plt.bar(range(top_k), expert_counts[top_indices])
        plt.xticks(range(top_k), [tokens[i] for i in top_indices], rotation=45, ha='right')
        plt.title(f"Top {top_k} Tokens for Expert {expert_idx}")
        plt.xlabel("Token")
        plt.ylabel("Usage Count")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    def plot_routing_confidence(
        self,
        routing_probs: Union[torch.Tensor, np.ndarray],
        bins: int = 50,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """Plot distribution of routing confidence scores.
        
        Args:
            routing_probs: Routing probabilities [num_tokens, num_experts]
            bins (int): Number of histogram bins. Defaults to 50
            figsize (Tuple[int, int]): Figure size. Defaults to (10, 6)
        """
        if isinstance(routing_probs, torch.Tensor):
            routing_probs = routing_probs.detach().cpu().numpy()
            
        max_probs = routing_probs.max(axis=1)
        
        plt.figure(figsize=figsize)
        plt.hist(max_probs, bins=bins, density=True)
        plt.title("Distribution of Routing Confidence Scores")
        plt.xlabel("Confidence Score")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
