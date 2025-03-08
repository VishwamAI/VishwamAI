"""
Standard training script for VishwamAI models.
Handles distributed training, mixed precision, and experiment tracking with Aim.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import logging
from tqdm import tqdm
from typing import Dict, Optional, Any
import math

from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.training.dataset_loader import VishwamAIDataset, create_dataloader
from vishwamai.optimisation.memory_optimization import MemoryOptimizer
from vishwamai.utils.logging import AimLogger

logger = logging.getLogger(__name__)

class Trainer:
    """
    Base trainer for VishwamAI models.
    Supports distributed training, mixed precision, and Aim logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataset: VishwamAIDataset,
        val_dataset: Optional[VishwamAIDataset] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize trainer"""
        # Default configuration
        default_config = {
            'batch_size': 32,
            'grad_acc_steps': 1,
            'learning_rate': 1e-4,
            'warmup_steps': 1000,
            'max_steps': 50000,
            'save_steps': 1000,
            'eval_steps': 500,
            'logging_steps': 100,
            'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
            'fp16': True,
            'checkpoint_dir': 'checkpoints',
            'experiment_name': 'vishwamai_training'
        }
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Initialize distributed training
        if self.config['local_rank'] != -1:
            torch.cuda.set_device(self.config['local_rank'])
            dist.init_process_group(backend='nccl')
            
        self.is_main_process = self.config['local_rank'] in [-1, 0]
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = model.to(self.device)
        if self.config['local_rank'] != -1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.config['local_rank']],
                output_device=self.config['local_rank']
            )
            
        # Create dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_dataloader = create_dataloader(
            train_dataset,
            batch_size=self.config['batch_size'],
            local_rank=self.config['local_rank']
        )
        if val_dataset:
            self.val_dataloader = create_dataloader(
                val_dataset,
                batch_size=self.config['batch_size'],
                local_rank=self.config['local_rank']
            )
            
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Initialize scheduler with warmup
        num_training_steps = self.config['max_steps']
        num_warmup_steps = self.config['warmup_steps']
        
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda
        )
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if self.config['fp16'] else None
        
        # Initialize memory optimization
        self.memory_optimizer = MemoryOptimizer(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device
        )
        
        # Initialize Aim logging
        if self.is_main_process:
            self.logger = AimLogger(
                experiment_name=self.config['experiment_name'],
                hparams=self.config,
                log_system_params=True
            )
            
            # Log model graph
            if isinstance(model, VishwamAITransformer):
                dummy_shape = (1, 32)  # Batch size 1, sequence length 32
                self.logger.log_model_graph(model, dummy_shape)
                
        # Initialize training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
    def compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute loss for prediction.
        
        Args:
            logits: Model output logits
            target_ids: Target token IDs
            
        Returns:
            loss: Computed loss value
        """
        return torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100
        )
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Execute single training step.
        
        Args:
            batch: Batch of training data
            
        Returns:
            loss: Training loss for this step
        """
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config['fp16']):
            logits = self.model(input_ids)
            loss = self.compute_loss(logits, target_ids)
            loss = loss / self.config['grad_acc_steps']
            
        # Backward pass with gradient scaling
        if self.config['fp16']:
            self.scaler.scale(loss).backward()
            if (self.step + 1) % self.config['grad_acc_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (self.step + 1) % self.config['grad_acc_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        # Log metrics
        if self.is_main_process and self.step % self.config['logging_steps'] == 0:
            self.logger.log_metrics(
                {
                    'loss': loss.item() * self.config['grad_acc_steps'],
                    'learning_rate': self.scheduler.get_last_lr()[0]
                },
                step=self.step
            )
            
        return loss.item() * self.config['grad_acc_steps']
        
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run evaluation and compute validation loss.
        
        Returns:
            avg_loss: Average validation loss
        """
        if not self.val_dataset:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        total_steps = 0
        
        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            
            with autocast(enabled=self.config['fp16']):
                logits = self.model(input_ids)
                loss = self.compute_loss(logits, target_ids)
                
            total_loss += loss.item()
            total_steps += 1
            
        avg_loss = total_loss / total_steps
        
        # Log validation metrics
        if self.is_main_process:
            self.logger.log_metrics(
                {'val_loss': avg_loss},
                step=self.step,
                context="validation"
            )
            
        return avg_loss
        
    def save_checkpoint(self, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        if not self.is_main_process:
            return
            
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        
        # Save model
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(
            self.config['checkpoint_dir'],
            f"checkpoint_{self.step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                "checkpoint_best.pt"
            )
            torch.save(checkpoint, best_path)
            
    def train(self) -> None:
        """Execute main training loop"""
        logger.info("Starting training")
        
        while self.step < self.config['max_steps']:
            epoch_loss = 0
            epoch_steps = 0
            
            # Enable memory optimization
            self.memory_optimizer.enable_gradient_checkpointing()
            if self.config['fp16']:
                self.memory_optimizer.enable_mixed_precision(self.scaler)
                
            # Training epoch
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}"):
                loss = self.train_step(batch)
                epoch_loss += loss
                epoch_steps += 1
                self.step += 1
                
                # Update learning rate
                self.scheduler.step()
                
                # Evaluation
                if self.step % self.config['eval_steps'] == 0:
                    val_loss = self.evaluate()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        if self.is_main_process:
                            self.save_checkpoint(is_best=True)
                            
                # Regular checkpoint saving
                if self.step % self.config['save_steps'] == 0 and self.is_main_process:
                    self.save_checkpoint()
                    
                if self.step >= self.config['max_steps']:
                    break
                    
            self.epoch += 1
            
            # Log epoch metrics
            if self.is_main_process:
                self.logger.log_metrics(
                    {'epoch_loss': epoch_loss / epoch_steps},
                    epoch=self.epoch
                )
                
        # Final checkpoint and cleanup
        if self.is_main_process:
            self.save_checkpoint()
            self.logger.close()
            
        logger.info("Training completed")

def main():
    """Main training function"""
    # Load config
    config = {
        'batch_size': 32,
        'grad_acc_steps': 1,
        'learning_rate': 1e-4,
        'warmup_steps': 1000,
        'max_steps': 50000,
        'save_steps': 1000,
        'eval_steps': 500,
        'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
        'fp16': True,
        'checkpoint_dir': 'checkpoints',
        'experiment_name': 'vishwamai_training'
    }
    
    # Initialize model
    model = VishwamAITransformer(
        vocab_size=50000,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072
    )
    
    # Load datasets
    train_dataset = VishwamAIDataset(
        data_path='path/to/train.json',
        tokenizer=None,  # Add your tokenizer here
        mode='normal'
    )
    
    val_dataset = VishwamAIDataset(
        data_path='path/to/val.json',
        tokenizer=None,  # Add your tokenizer here
        mode='normal'
    )
    
    # Initialize trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()