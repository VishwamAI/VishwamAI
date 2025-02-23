"""Visualization utilities for model analysis and training monitoring."""
from typing import Dict, List, Optional, Union, Tuple
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

def plot_training_history(
    history: Dict[str, List[Dict[str, float]]],
    metrics: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot training metrics over time.
    
    Args:
        history: Training history dict
        metrics: Optional list of metrics to plot
        save_dir: Optional directory to save plots
        show: Whether to display plots
    """
    if metrics is None:
        metrics = list(history['train'][0].keys())
        metrics.remove('step')
        
    fig, axes = plt.subplots(
        math.ceil(len(metrics) / 2), 2,
        figsize=(15, 5 * math.ceil(len(metrics) / 2))
    )
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Plot train metrics
        train_steps = [x['step'] for x in history['train']]
        train_values = [x[metric] for x in history['train']]
        ax.plot(train_steps, train_values, label='train')
        
        # Plot val metrics if available
        if history.get('val'):
            val_steps = [x['step'] for x in history['val']]
            val_values = [x[metric] for x in history['val']]
            ax.plot(val_steps, val_values, label='val')
            
        ax.set_title(metric)
        ax.set_xlabel('Step')
        ax.legend()
        
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'training_history.png'
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_expert_usage(
    usage_stats: Dict[str, List[float]],
    save_dir: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot expert usage statistics.
    
    Args:
        usage_stats: Expert usage statistics
        save_dir: Optional directory to save plots
        show: Whether to display plots
    """
    num_experts = len(usage_stats['expert_usage'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Plot usage distribution
    sns.barplot(
        x=list(range(num_experts)),
        y=usage_stats['expert_usage'],
        ax=axes[0,0]
    )
    axes[0,0].set_title('Expert Usage Distribution')
    axes[0,0].set_xlabel('Expert Index')
    axes[0,0].set_ylabel('Usage Fraction')
    
    # Plot timing stats
    axes[0,1].errorbar(
        x=list(range(num_experts)),
        y=usage_stats['expert_mean_time'],
        yerr=usage_stats['expert_std_time'],
        fmt='o'
    )
    axes[0,1].set_title('Expert Timing')
    axes[0,1].set_xlabel('Expert Index')
    axes[0,1].set_ylabel('Mean Time (s)')
    
    # Plot token distribution
    sns.barplot(
        x=list(range(num_experts)),
        y=usage_stats['expert_token_frac'],
        ax=axes[1,0]
    )
    axes[1,0].set_title('Token Distribution')
    axes[1,0].set_xlabel('Expert Index')
    axes[1,0].set_ylabel('Token Fraction')
    
    # Plot call counts
    sns.barplot(
        x=list(range(num_experts)),
        y=usage_stats['expert_calls'],
        ax=axes[1,1]
    )
    axes[1,1].set_title('Expert Call Counts')
    axes[1,1].set_xlabel('Expert Index')
    axes[1,1].set_ylabel('Number of Calls')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'expert_usage.png'
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()

def plot_attention_patterns(
    attention_weights: torch.Tensor,
    save_dir: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot attention pattern heatmaps.
    
    Args:
        attention_weights: Attention weights [batch, heads, seq, seq]
        save_dir: Optional directory to save plots
        show: Whether to display plots
    """
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    for batch_idx in range(min(batch_size, 4)):
        fig, axes = plt.subplots(
            math.ceil(num_heads / 2), 2,
            figsize=(15, 5 * math.ceil(num_heads / 2))
        )
        axes = axes.flatten()
        
        for head_idx in range(num_heads):
            ax = axes[head_idx]
            sns.heatmap(
                attention_weights[batch_idx, head_idx].cpu(),
                ax=ax,
                cmap='viridis'
            )
            ax.set_title(f'Head {head_idx}')
            
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / f'attention_batch{batch_idx}.png'
            plt.savefig(save_path)
            
        if show:
            plt.show()
        else:
            plt.close()

def plot_loss_landscape(
    model: torch.nn.Module,
    loss_fn: callable,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_points: int = 20,
    delta: float = 1.0,
    save_dir: Optional[str] = None,
    show: bool = True
) -> None:
    """Plot 2D loss landscape around current parameters.
    
    Args:
        model: Model to analyze
        loss_fn: Loss function
        inputs: Input batch
        targets: Target batch
        num_points: Number of points per dimension
        delta: Maximum perturbation magnitude
        save_dir: Optional directory to save plots
        show: Whether to display plots
    """
    # Get random directions
    theta = torch.cat([p.flatten() for p in model.parameters()])
    d1 = torch.randn_like(theta)
    d2 = torch.randn_like(theta)
    d1 = d1 / d1.norm()
    d2 = d2 / d2.norm()
    
    # Generate grid
    alphas = np.linspace(-delta, delta, num_points)
    betas = np.linspace(-delta, delta, num_points)
    losses = np.zeros((num_points, num_points))
    
    # Evaluate loss landscape
    original_theta = theta.clone()
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            # Update parameters
            new_theta = original_theta + alpha * d1 + beta * d2
            idx = 0
            for p in model.parameters():
                num_params = p.numel()
                p.data = new_theta[idx:idx+num_params].reshape(p.shape)
                idx += num_params
                
            # Evaluate loss
            with torch.no_grad():
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                losses[i,j] = loss.item()
                
    # Reset parameters
    idx = 0
    for p in model.parameters():
        num_params = p.numel()
        p.data = original_theta[idx:idx+num_params].reshape(p.shape)
        idx += num_params
        
    # Plot landscape
    plt.figure(figsize=(10, 8))
    plt.contour(alphas, betas, losses, levels=50)
    plt.colorbar(label='Loss')
    plt.xlabel('Direction 1')
    plt.ylabel('Direction 2')
    plt.title('Loss Landscape')
    
    if save_dir:
        save_path = Path(save_dir) / 'loss_landscape.png'
        plt.savefig(save_path)
        
    if show:
        plt.show()
    else:
        plt.close()
