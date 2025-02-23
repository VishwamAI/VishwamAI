"""Attention visualization utilities."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class AttentionVisualizer:
    """Visualizes attention patterns and weights."""
    
    def __init__(self):
        """Initialize the attention visualizer."""
        plt.style.use('seaborn')
        
    def plot_attention_weights(
        self,
        attention_weights: Union[torch.Tensor, np.ndarray],
        tokens: Optional[List[str]] = None,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        title: Optional[str] = None
    ) -> None:
        """Plot attention weights as a heatmap.
        
        Args:
            attention_weights: Attention weights tensor/array [seq_len, seq_len]
            tokens (Optional[List[str]]): Input tokens for axis labels
            layer_idx (Optional[int]): Layer index for title
            head_idx (Optional[int]): Attention head index for title
            figsize (Tuple[int, int]): Figure size. Defaults to (10, 8)
            cmap (str): Colormap for heatmap. Defaults to 'viridis'
            title (Optional[str]): Custom title for plot
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
            
        plt.figure(figsize=figsize)
        sns.heatmap(
            attention_weights,
            cmap=cmap,
            xticklabels=tokens if tokens else 'auto',
            yticklabels=tokens if tokens else 'auto',
            cbar=True
        )
        
        if title is None:
            title = "Attention Weights"
            if layer_idx is not None:
                title += f" (Layer {layer_idx}"
                if head_idx is not None:
                    title += f", Head {head_idx})"
                else:
                    title += ")"
                    
        plt.title(title)
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        
    def plot_attention_patterns(
        self,
        attention_weights: Union[torch.Tensor, np.ndarray],
        num_heads: Optional[int] = None,
        tokens: Optional[List[str]] = None,
        layer_idx: Optional[int] = None,
        max_seq_len: int = 100,
        figsize: Optional[Tuple[int, int]] = None
    ) -> None:
        """Plot attention patterns for multiple heads.
        
        Args:
            attention_weights: Attention weights [num_heads, seq_len, seq_len]
            num_heads (Optional[int]): Number of attention heads to plot
            tokens (Optional[List[str]]): Input tokens for axis labels
            layer_idx (Optional[int]): Layer index for title
            max_seq_len (int): Maximum sequence length to plot
            figsize (Optional[Tuple[int, int]]): Figure size
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
            
        # Truncate sequence length if needed
        if attention_weights.shape[1] > max_seq_len:
            attention_weights = attention_weights[:, :max_seq_len, :max_seq_len]
            if tokens:
                tokens = tokens[:max_seq_len]
                
        num_heads = num_heads or attention_weights.shape[0]
        nrows = int(np.ceil(num_heads / 2))
        figsize = figsize or (15, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, 2, figsize=figsize)
        if nrows == 1:
            axes = axes.reshape(1, -1)
            
        layer_str = f" (Layer {layer_idx})" if layer_idx is not None else ""
        fig.suptitle(f"Attention Patterns{layer_str}")
        
        for head_idx in range(num_heads):
            row = head_idx // 2 
            col = head_idx % 2
            
            sns.heatmap(
                attention_weights[head_idx],
                cmap='viridis',
                xticklabels=tokens if tokens else 'auto',
                yticklabels=tokens if tokens else 'auto',
                cbar=True,
                ax=axes[row, col]
            )
            
            axes[row, col].set_title(f"Head {head_idx}")
            axes[row, col].set_xlabel("Key Tokens")
            axes[row, col].set_ylabel("Query Tokens")
            
        # Hide empty subplots
        for idx in range(num_heads, nrows * 2):
            row = idx // 2
            col = idx % 2
            axes[row, col].axis('off')
            
        plt.tight_layout()
        
    def plot_attention_flow(
        self,
        attention_weights: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        focus_position: int,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Visualize attention flow from a specific position.
        
        Args:
            attention_weights: Attention weights tensor/array
            tokens (List[str]): Input tokens
            focus_position (int): Position to visualize attention from
            layer_idx (Optional[int]): Layer index for title
            head_idx (Optional[int]): Attention head index for title
            figsize (Tuple[int, int]): Figure size. Defaults to (12, 6)
        """
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()
            
        plt.figure(figsize=figsize)
        
        attention_from_position = attention_weights[focus_position]
        positions = np.arange(len(tokens))
        
        plt.bar(positions, attention_from_position)
        plt.xticks(positions, tokens, rotation=45, ha='right')
        
        title = f"Attention Flow from '{tokens[focus_position]}'"
        if layer_idx is not None:
            title += f" (Layer {layer_idx}"
            if head_idx is not None:
                title += f", Head {head_idx})"
            else:
                title += ")"
                
        plt.title(title)
        plt.xlabel("Tokens")
        plt.ylabel("Attention Weight")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
