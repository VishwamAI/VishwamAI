"""Logging utilities for training."""
from typing import Any, Dict, List, Optional, Union
import os
import json
import time
import logging
from pathlib import Path
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

class MetricLogger:
    """Logging class for training metrics.
    
    Supports multiple logging backends:
    - Local files
    - TensorBoard
    - Weights & Biases
    - Console
    
    Args:
        log_dir: Directory for logs
        experiment_name: Name of experiment
        backends: List of logging backends to use
        use_wandb: Whether to use W&B
        wandb_project: W&B project name
        wandb_entity: W&B entity name
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        backends: List[str] = ['file', 'tensorboard'],
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.backends = backends
        self.use_wandb = use_wandb
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        if 'file' in backends:
            self.log_file = self.log_dir / 'training.log'
            logging.basicConfig(
                filename=str(self.log_file),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s'
            )
            
        # Setup TensorBoard
        if 'tensorboard' in backends:
            self.writer = SummaryWriter(
                log_dir=str(self.log_dir / 'tensorboard')
            )
            
        # Setup W&B
        if use_wandb:
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=experiment_name,
                dir=str(self.log_dir / 'wandb')
            )
            
        # Metric history
        self.history = {
            'train': [],
            'val': [],
            'test': []
        }
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ) -> None:
        """Log metrics for current step.
        
        Args:
            metrics: Dictionary of metrics
            step: Current step
            phase: Training phase ['train', 'val', 'test']
        """
        # Add to history
        self.history[phase].append({
            'step': step,
            **metrics
        })
        
        # File logging
        if 'file' in self.backends:
            metrics_str = ' '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logging.info(f"Step {step} ({phase}): {metrics_str}")
            
        # TensorBoard
        if 'tensorboard' in self.backends:
            for name, value in metrics.items():
                self.writer.add_scalar(
                    f"{phase}/{name}",
                    value,
                    step
                )
                
        # W&B
        if self.use_wandb:
            wandb.log({
                f"{phase}/{k}": v for k, v in metrics.items()
            }, step=step)
            
    def log_model_info(
        self,
        model: torch.nn.Module,
        input_shape: Optional[tuple] = None
    ) -> None:
        """Log model architecture and stats.
        
        Args:
            model: PyTorch model
            input_shape: Optional shape for model summary
        """
        # Get model stats
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        model_info = {
            'num_parameters': num_params,
            'trainable_parameters': trainable_params,
            'architecture': str(model)
        }
        
        # Save to file
        if 'file' in self.backends:
            info_file = self.log_dir / 'model_info.json'
            with open(info_file, 'w') as f:
                json.dump(model_info, f, indent=2)
                
        # TensorBoard
        if 'tensorboard' in self.backends and input_shape:
            self.writer.add_graph(
                model,
                torch.randn(input_shape).to(next(model.parameters()).device)
            )
            
        # W&B
        if self.use_wandb:
            wandb.config.update(model_info)
            
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        # Save to file
        if 'file' in self.backends:
            hparams_file = self.log_dir / 'hyperparameters.json'
            with open(hparams_file, 'w') as f:
                json.dump(hparams, f, indent=2)
                
        # TensorBoard
        if 'tensorboard' in self.backends:
            self.writer.add_hparams(hparams, {})
            
        # W&B
        if self.use_wandb:
            wandb.config.update(hparams)
            
    def log_expert_stats(
        self,
        expert_metrics: Dict[str, List[float]],
        step: int
    ) -> None:
        """Log expert-specific metrics.
        
        Args:
            expert_metrics: Dictionary of expert metrics
            step: Current step
        """
        # Calculate summary stats
        stats = {}
        for name, values in expert_metrics.items():
            stats.update({
                f"{name}_mean": sum(values) / len(values),
                f"{name}_min": min(values),
                f"{name}_max": max(values),
                f"{name}_std": torch.tensor(values).std().item()
            })
            
        # Log metrics
        self.log_metrics(stats, step)
        
        # Expert-specific logging
        if 'file' in self.backends:
            expert_file = self.log_dir / f"expert_stats_{step}.json"
            with open(expert_file, 'w') as f:
                json.dump(expert_metrics, f, indent=2)
                
    def save_history(self) -> None:
        """Save metric history to file."""
        history_file = self.log_dir / 'metric_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def close(self) -> None:
        """Close logging connections."""
        if 'tensorboard' in self.backends:
            self.writer.close()
            
        if self.use_wandb:
            wandb.finish()
            
class DistributedLogger(MetricLogger):
    """Logger for distributed training.
    
    Only rank 0 performs logging.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        **kwargs: Arguments passed to MetricLogger
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        **kwargs
    ):
        self.rank = rank
        self.world_size = world_size
        
        # Only create logger on rank 0
        if rank == 0:
            super().__init__(**kwargs)
            
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        phase: str = 'train'
    ) -> None:
        """Log metrics only from rank 0."""
        if self.rank == 0:
            super().log_metrics(metrics, step, phase)
            
    def log_model_info(
        self,
        model: torch.nn.Module,
        input_shape: Optional[tuple] = None
    ) -> None:
        """Log model info only from rank 0."""
        if self.rank == 0:
            super().log_model_info(model, input_shape)
            
    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters only from rank 0."""
        if self.rank == 0:
            super().log_hyperparameters(hparams)
            
    def log_expert_stats(
        self,
        expert_metrics: Dict[str, List[float]],
        step: int
    ) -> None:
        """Log expert stats only from rank 0."""
        if self.rank == 0:
            super().log_expert_stats(expert_metrics, step)
            
    def save_history(self) -> None:
        """Save history only from rank 0."""
        if self.rank == 0:
            super().save_history()
            
    def close(self) -> None:
        """Close logger only from rank 0."""
        if self.rank == 0:
            super().close()
