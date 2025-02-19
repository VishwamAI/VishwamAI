"""
Advanced training utilities and features for VishwamAI models.
Includes support for continuous training with A100 GPU optimization and Tree of Thoughts integration.
"""

import os
import time
import signal
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, List, Union, Tuple
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset, Dataset
from torch.cuda.amp import autocast, GradScaler

from vishwamai.extensions.neural_memory import NeuralMemory
from vishwamai.extensions.tree_of_thoughts import TreeOfThoughts, TreeConfig
from vishwamai.utils.config import ModelConfig, TrainingConfig, OpenEndedConfig
from vishwamai.models.Transformer import Transformer
from vishwamai.training.curriculum import CurriculumLearning
from vishwamai.utils.parallel import model_parallel_forward

class AdvancedTrainer:
    """Advanced training with multi-dataset support and A100 optimization."""
    
    def __init__(
        self,
        model: Transformer,
        config: ModelConfig,
        training_config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        use_tree_search: bool = True
    ):
        self.model = model 
        self.config = config
        self.training_config = training_config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize advanced features
        if config.use_memory:
            self.memory = NeuralMemory(config)
            
        if getattr(config, 'use_curriculum', False):
            self.curriculum = CurriculumLearning(
                config=config,
                training_config=self.training_config
            )
            
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup A100 optimizations
        self.scaler = GradScaler()  # For mixed precision training
        self.batch_accumulation = 4  # Gradient accumulation steps
        
        # Configure model for A100
        if torch.cuda.get_device_name().find('A100') != -1:
            # Enable tensor cores
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set optimal memory allocation
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
            
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize Tree of Thoughts if enabled
        if use_tree_search:
            self.tree_of_thoughts = TreeOfThoughts(
                model=self.model,
                config=TreeConfig(
                    beam_width=4,
                    max_depth=3,
                    temperature=0.7
                )
            )
        
        # Initialize training state
        self.running = True
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = self.training_config.checkpoint_interval or 3600  # Default 1 hour
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training monitoring."""
        logger = logging.getLogger('VishwamAI_Training')
        logger.setLevel(logging.INFO)
        
        os.makedirs('logs', exist_ok=True)
        
        fh = logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
        
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with weight decay and large batch optimizations."""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.training_config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_params,
            lr=self.training_config.learning_rate,
            betas=(0.9, 0.95),  # Adjusted for large batch training
            eps=1e-8,
            fused=True  # Enable fused implementation for A100
        )
        
        return optimizer
        
    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler with warmup."""
        warmup_steps = self.training_config.warmup_steps or 1000
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return 1.0
            
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received shutdown signal {signum}. Saving checkpoint...")
        self.running = False
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        use_tree_search: bool = False
    ) -> Dict[str, float]:
        """Single training step with mixed precision and gradient accumulation."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with autocast(device_type='cuda', dtype=torch.float16):
            if use_tree_search and hasattr(self, 'tree_of_thoughts'):
                outputs = self.tree_of_thoughts.forward(batch)
            else:
                outputs = self.model(**batch)
                
            loss = outputs['loss']
            
            # Add memory loss if using memory
            if self.config.use_memory:
                memory_outputs = self.memory(outputs['hidden_states'])
                loss = loss + 0.1 * memory_outputs['stats']['access_loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.batch_accumulation
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        metrics = {
            'loss': loss.item() * self.batch_accumulation,
            'perplexity': torch.exp(loss * self.batch_accumulation).item()
        }
        
        # Gradient accumulation
        if self.optimizer._step_count % self.batch_accumulation == 0:
            # Gradient clipping
            if self.training_config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config.max_grad_norm
                )
            
            # Optimizer step with gradient scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
        return metrics
        
    def _should_checkpoint(self) -> bool:
        """Determine if it's time to create a checkpoint."""
        return time.time() - self.last_checkpoint_time >= self.checkpoint_interval
        
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        epoch: int,
        use_curriculum: bool = False,
        checkpoint_dir: str = 'checkpoints'
    ) -> Dict[str, float]:
        """Train for one epoch with progress monitoring."""
        total_loss = 0
        total_samples = 0
        
        # Get curriculum samples if using curriculum learning
        if use_curriculum and hasattr(self, 'curriculum'):
            dataloader = self.curriculum.get_samples(
                dataloader,
                epoch=epoch
            )
            
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')
        for batch in progress_bar:
            # Determine whether to use tree search
            use_tree_search = (
                epoch >= self.config.tree_search_start_epoch
                if hasattr(self.config, 'tree_search_start_epoch')
                else False
            )
            
            # Training step
            step_results = self.train_step(
                batch,
                use_tree_search=use_tree_search
            )
            
            # Update metrics
            batch_size = batch['input_ids'].size(0)
            total_loss += step_results['loss'] * batch_size
            total_samples += batch_size
            
            # Check for checkpointing
            if self._should_checkpoint():
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f'checkpoint_epoch{epoch}_step{total_samples}.pt'
                )
                self.save_checkpoint(checkpoint_path, epoch)
                self.last_checkpoint_time = time.time()
                self.logger.info(f"Created checkpoint: {checkpoint_path}")
                
            # Log metrics
            self.logger.info(
                f"Step {total_samples}: loss={step_results['loss']:.4f}, "
                f"perplexity={step_results['perplexity']:.4f}, "
                f"lr={self.get_learning_rate():.2e}"
            )
            
            # Update progress bar
            progress_bar.set_postfix(step_results)
            
            # Update learning rate
            self.scheduler.step()
            
            # Check if we should stop
            if not self.running:
                self.logger.info("Stopping training due to shutdown signal")
                break
            
        # Calculate epoch metrics
        epoch_metrics = {
            'loss': total_loss / total_samples,
            'perplexity': torch.exp(torch.tensor(total_loss / total_samples)).item()
        }
        
        return epoch_metrics
        
    def save_checkpoint(
        self,
        save_path: str,
        epoch: int,
        optimizer_state: Optional[dict] = None,
        include_memory: bool = True
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),  # Save gradient scaler state
            'config': self.config.__dict__,
            'training_config': self.training_config.__dict__
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        if include_memory and hasattr(self, 'memory'):
            checkpoint['memory_state'] = self.memory.state_dict()
            
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint to {save_path}")
        
    def load_checkpoint(self, checkpoint_path: str, load_memory: bool = True):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        if load_memory and 'memory_state' in checkpoint and hasattr(self, 'memory'):
            self.memory.load_state_dict(checkpoint['memory_state'])
            
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('epoch', -1)
        
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
        
    def get_memory_stats(self) -> Optional[Dict[str, float]]:
        """Get neural memory statistics if enabled."""
        if hasattr(self, 'memory'):
            return self.memory.get_stats()
        return None
