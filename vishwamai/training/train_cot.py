"""
Chain of Thought (CoT) training script for VishwamAI.
Extends standard training with thought-aware losses and evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import aim
import logging
from tqdm import tqdm
from typing import Dict, Optional, Any, Tuple

from vishwamai.models.cot_model import CoTModel
from vishwamai.training.dataset_loader import VishwamAIDataset, create_dataloader
from vishwamai.training.train_normal import Trainer as BaseTrainer
from vishwamai.optimisation.memory_optimization import MemoryOptimizer

logger = logging.getLogger(__name__)

class CoTTrainer(BaseTrainer):
    """
    Specialized trainer for Chain of Thought models.
    Extends base trainer with thought-aware training.
    """
    
    def __init__(
        self,
        model: CoTModel,
        train_dataset: VishwamAIDataset,
        val_dataset: Optional[VishwamAIDataset] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize CoT trainer"""
        # Update default config with CoT-specific settings
        default_config = {
            'thought_loss_weight': 0.5,  # Weight for thought coherence loss
            'max_thought_length': 256,    # Maximum length for thought sequences
            'min_thought_length': 16,     # Minimum length for valid thoughts
            'thought_temperature': 0.8,   # Temperature for thought generation
            'thought_top_p': 0.95,        # Top-p for thought sampling
            'experiment_name': 'vishwamai_cot_training'  # Aim experiment name
        }
        if config:
            default_config.update(config)
            
        super().__init__(model, train_dataset, val_dataset, default_config)
        
        # Initialize thought-specific metrics
        self.thought_metrics = {
            'thought_loss': 0.0,
            'answer_loss': 0.0,
            'thought_length': 0.0,
            'thought_coherence': 0.0
        }
        
        self.setup_aim_logging()
        
    def setup_aim_logging(self):
        """Initialize Aim experiment tracking with CoT-specific settings"""
        self.aim_run = aim.Run(
            experiment=self.config['experiment_name'],
            log_system_params=True
        )
        # Log configuration including CoT-specific parameters
        self.aim_run["hparams"] = self.config
        
        # Set descriptive run name
        self.aim_run.name = f"cot_{self.config['experiment_name']}_{aim.Run.generate_run_hash()}"
        
        # Create metric contexts for different types of metrics
        self.aim_run.create_context("thought_metrics")
        self.aim_run.create_context("answer_metrics")
        
    def compute_loss(
        self, 
        logits: torch.Tensor, 
        target_ids: torch.Tensor,
        thought_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CoT-specific loss with separate thought and answer components.
        
        Args:
            logits: Model output logits (batch_size, seq_len, vocab_size)
            target_ids: Target token IDs (batch_size, seq_len)
            thought_mask: Mask indicating thought regions (batch_size, seq_len)
            
        Returns:
            total_loss: Combined loss value
            metrics: Dictionary of component losses and metrics
        """
        batch_size, seq_len, vocab_size = logits.size()
        
        # Reshape for loss computation
        logits = logits.view(-1, vocab_size)
        target_ids = target_ids.view(-1)
        
        # Base loss
        base_loss = nn.functional.cross_entropy(
            logits, 
            target_ids, 
            ignore_index=-1,
            reduction='none'
        ).view(batch_size, seq_len)
        
        # Separate thought and answer losses if mask provided
        if thought_mask is not None:
            # Thought region loss
            thought_loss = (base_loss * thought_mask).sum() / (thought_mask.sum() + 1e-6)
            
            # Answer region loss
            answer_mask = 1 - thought_mask
            answer_loss = (base_loss * answer_mask).sum() / (answer_mask.sum() + 1e-6)
            
            # Combine with weighting
            total_loss = (
                self.config['thought_loss_weight'] * thought_loss +
                (1 - self.config['thought_loss_weight']) * answer_loss
            )
            
            # Compute thought coherence metric
            thought_coherence = self._compute_thought_coherence(logits, target_ids, thought_mask)
            
            metrics = {
                'thought_loss': thought_loss.item(),
                'answer_loss': answer_loss.item(),
                'thought_coherence': thought_coherence
            }
        else:
            # Fallback to standard loss if no mask
            total_loss = base_loss.mean()
            metrics = {}
            
        return total_loss, metrics
        
    def _compute_thought_coherence(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        thought_mask: torch.Tensor
    ) -> float:
        """
        Compute coherence metric for generated thoughts.
        Uses next token prediction accuracy within thought regions.
        """
        with torch.no_grad():
            # Get predictions
            preds = logits.argmax(dim=-1)
            
            # Mask for thought regions only
            masked_correct = (preds == target_ids) * thought_mask.view(-1)
            
            # Compute accuracy in thought regions
            coherence = masked_correct.sum() / (thought_mask.sum() + 1e-6)
            
        return coherence.item()
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single CoT training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        # Get thought mask if available in batch
        thought_mask = batch.get('thought_mask', None)
        if thought_mask is not None:
            thought_mask = thought_mask.to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config['fp16']):
            logits = self.model(input_ids)
            loss, step_metrics = self.compute_loss(logits, target_ids, thought_mask)
            loss = loss / self.config['grad_acc_steps']
            
        # Update thought metrics
        for k, v in step_metrics.items():
            self.thought_metrics[k] = v
            if self.is_main_process:
                # Log detailed metrics with Aim under appropriate contexts
                if 'thought' in k:
                    context = "thought_metrics"
                else:
                    context = "answer_metrics"
                    
                self.aim_run.track(
                    v,
                    name=f'train/{k}',
                    step=self.step,
                    epoch=self.epoch,
                    context=context
                )
            
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
                
        return loss.item() * self.config['grad_acc_steps']
        
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run evaluation with thought-specific metrics.
        Returns average loss.
        """
        if not self.val_dataset:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        total_metrics = {k: 0.0 for k in self.thought_metrics.keys()}
        total_steps = 0
        
        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            thought_mask = batch.get('thought_mask', None)
            if thought_mask is not None:
                thought_mask = thought_mask.to(self.device)
            
            with autocast(enabled=self.config['fp16']):
                logits = self.model(input_ids)
                loss, step_metrics = self.compute_loss(logits, target_ids, thought_mask)
                
            total_loss += loss.item()
            for k, v in step_metrics.items():
                total_metrics[k] += v
            total_steps += 1
            
        # Compute averages
        avg_loss = total_loss / total_steps
        avg_metrics = {k: v / total_steps for k, v in total_metrics.items()}
        
        # Log validation metrics with Aim
        if self.is_main_process:
            self.aim_run.track(
                avg_loss,
                name='val/loss',
                step=self.step,
                epoch=self.epoch
            )
            
            # Log detailed validation metrics under appropriate contexts
            for k, v in avg_metrics.items():
                context = "thought_metrics" if 'thought' in k else "answer_metrics"
                self.aim_run.track(
                    v,
                    name=f'val/{k}',
                    step=self.step,
                    epoch=self.epoch,
                    context=context
                )
            
        return avg_loss
        
    def train(self):
        """Main training loop with CoT-specific logging"""
        logger.info("Starting CoT training")
        
        while self.step < self.config['max_steps']:
            epoch_loss = 0
            epoch_metrics = {k: 0.0 for k in self.thought_metrics.keys()}
            epoch_steps = 0
            
            # Enable memory optimization
            self.memory_optimizer.enable_gradient_checkpointing()
            if self.config['fp16']:
                self.memory_optimizer.enable_mixed_precision(self.scaler)
                
            # Training epoch
            for batch in tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}"):
                loss = self.train_step(batch)
                epoch_loss += loss
                for k, v in self.thought_metrics.items():
                    epoch_metrics[k] += v
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
            
            # Log epoch metrics with Aim
            if self.is_main_process:
                avg_metrics = {k: v / epoch_steps for k, v in epoch_metrics.items()}
                
                # Log overall epoch metrics
                self.aim_run.track(
                    epoch_loss / epoch_steps,
                    name='train/epoch_loss',
                    epoch=self.epoch
                )
                
                # Log detailed epoch metrics
                for k, v in avg_metrics.items():
                    context = "thought_metrics" if 'thought' in k else "answer_metrics"
                    self.aim_run.track(
                        v,
                        name=f'train/epoch_{k}',
                        epoch=self.epoch,
                        context=context
                    )
                
        # Final checkpoint and cleanup
        if self.is_main_process:
            self.save_checkpoint()
            self.aim_run.close()
            
        logger.info("CoT training completed")

def main():
    """Main training function"""
    # Load config
    config = {
        'batch_size': 16,  # Smaller batch size for CoT
        'grad_acc_steps': 4,
        'learning_rate': 5e-5,  # Lower learning rate for stability
        'warmup_steps': 2000,
        'max_steps': 100000,
        'save_steps': 1000,
        'eval_steps': 500,
        'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
        'fp16': True,
        'checkpoint_dir': 'checkpoints',
        'thought_loss_weight': 0.5,
        'max_thought_length': 256,
        'min_thought_length': 16,
        'experiment_name': 'vishwamai_cot_training'
    }
    
    # Initialize model
    model = CoTModel(
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072,
        vocab_size=50000
    )
    
    # Load datasets
    train_dataset = VishwamAIDataset(
        data_path='path/to/train.json',
        tokenizer=None,  # Add your tokenizer here
        mode='cot'
    )
    
    val_dataset = VishwamAIDataset(
        data_path='path/to/val.json',
        tokenizer=None,  # Add your tokenizer here
        mode='cot'
    )
    
    # Initialize trainer
    trainer = CoTTrainer(model, train_dataset, val_dataset, config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()