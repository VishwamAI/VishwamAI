"""
Model distillation utilities for VishwamAI.
Supports knowledge distillation, feature-based distillation, and attention distillation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Any, List, Tuple
from dataclasses import dataclass
import yaml
import os

from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.training.dataset_loader import VishwamAIDataset, create_dataloader
from vishwamai.optimisation.memory_optimization import MemoryOptimizer
from vishwamai.training.parallel_training import ParallelTrainer, ParallelConfig

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    """Configuration for model distillation"""
    temperature: float = 2.0          # Temperature for softening logits
    alpha: float = 0.5               # Weight for distillation loss
    beta: float = 0.3                # Weight for feature matching loss
    gamma: float = 0.2               # Weight for attention matching loss
    feature_layers: List[int] = None # Layers for feature matching
    use_attention_distill: bool = True  # Whether to distill attention maps
    max_grad_norm: float = 1.0       # Gradient clipping norm
    mixed_precision: bool = True      # Whether to use mixed precision

class DistillationLoss(nn.Module):
    """
    Combined loss function for knowledge distillation.
    Includes KL divergence, feature matching, and attention matching.
    """
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_features: Optional[List[torch.Tensor]] = None,
        teacher_features: Optional[List[torch.Tensor]] = None,
        student_attentions: Optional[List[torch.Tensor]] = None,
        teacher_attentions: Optional[List[torch.Tensor]] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            student_features: Hidden states from student model
            teacher_features: Hidden states from teacher model
            student_attentions: Attention maps from student model
            teacher_attentions: Attention maps from teacher model
            labels: Ground truth labels for supervised loss
            
        Returns:
            total_loss: Combined distillation loss
            metrics: Dictionary of component losses
        """
        metrics = {}
        
        # Knowledge distillation loss
        soft_student = F.log_softmax(student_logits / self.config.temperature, dim=-1)
        soft_teacher = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        distill_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        distill_loss *= (self.config.temperature ** 2)
        metrics['distill_loss'] = distill_loss.item()
        
        # Hard loss with ground truth if available
        if labels is not None:
            hard_loss = F.cross_entropy(student_logits, labels)
            metrics['hard_loss'] = hard_loss.item()
        else:
            hard_loss = 0
            
        # Feature matching loss
        feature_loss = 0
        if student_features and teacher_features:
            for student_feat, teacher_feat in zip(student_features, teacher_features):
                feature_loss += F.mse_loss(student_feat, teacher_feat)
            metrics['feature_loss'] = feature_loss.item()
            
        # Attention matching loss
        attention_loss = 0
        if self.config.use_attention_distill and student_attentions and teacher_attentions:
            for student_attn, teacher_attn in zip(student_attentions, teacher_attentions):
                attention_loss += F.mse_loss(student_attn, teacher_attn)
            metrics['attention_loss'] = attention_loss.item()
            
        # Combine losses
        total_loss = (
            (1 - self.config.alpha) * hard_loss +
            self.config.alpha * distill_loss +
            self.config.beta * feature_loss +
            self.config.gamma * attention_loss
        )
        metrics['total_loss'] = total_loss.item()
        
        return total_loss, metrics

class DistillationTrainer(ParallelTrainer):
    """
    Trainer for model distillation, extending parallel training capabilities.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_dataset: VishwamAIDataset,
        val_dataset: Optional[VishwamAIDataset] = None,
        distill_config: Optional[DistillationConfig] = None,
        parallel_config: Optional[ParallelConfig] = None,
        training_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize distillation trainer"""
        super().__init__(
            model_class=type(student_model),
            model_config={},  # Empty since we pass instantiated model
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            parallel_config=parallel_config,
            training_config=training_config
        )
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.distill_config = distill_config or DistillationConfig()
        
        # Initialize loss function
        self.distill_loss_fn = DistillationLoss(self.distill_config)
        
        # Set models to appropriate modes
        self.teacher_model.eval()
        self.student_model.train()
        
        # Move models to device
        self.teacher_model = self.teacher_model.to(self.device)
        self.student_model = self.student_model.to(self.device)
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single distillation training step"""
        self.student_model.train()
        self.teacher_model.eval()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', None)
        if labels is not None:
            labels = labels.to(self.device)
            
        # Teacher forward pass
        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids,
                output_hidden_states=True,
                output_attentions=self.distill_config.use_attention_distill
            )
            
        # Student forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.distill_config.mixed_precision):
            student_outputs = self.student_model(
                input_ids,
                output_hidden_states=True,
                output_attentions=self.distill_config.use_attention_distill
            )
            
            # Compute loss
            loss, metrics = self.distill_loss_fn(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_outputs.logits,
                student_features=student_outputs.hidden_states,
                teacher_features=teacher_outputs.hidden_states,
                student_attentions=student_outputs.attentions,
                teacher_attentions=teacher_outputs.attentions,
                labels=labels
            )
            
            loss = loss / self.parallel_config.grad_accumulation
            
        # Backward pass
        if self.distill_config.mixed_precision:
            self.scaler.scale(loss).backward()
            if (self.global_step + 1) % self.parallel_config.grad_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.distill_config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (self.global_step + 1) % self.parallel_config.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.student_model.parameters(),
                    self.distill_config.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return loss.item(), metrics

    def train(self, num_epochs: int):
        """Main distillation training loop"""
        logger.info("Starting distillation training")
        
        self.global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_metrics = {}
            num_steps = 0
            
            # Enable memory optimization
            if self.parallel_config.optimize_memory:
                self.memory_optimizer.enable_gradient_checkpointing()
                if self.distill_config.mixed_precision:
                    self.memory_optimizer.enable_mixed_precision(self.scaler)
                    
            # Training epoch
            for batch in self.train_dataloader:
                loss, step_metrics = self.train_step(batch)
                epoch_loss += loss
                
                # Accumulate metrics
                for k, v in step_metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0) + v
                    
                num_steps += 1
                self.global_step += 1
                
                # Validation
                if self.val_dataset and self.global_step % self.training_config.get('eval_steps', 500) == 0:
                    val_loss = self.evaluate()
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if self.is_main_process:
                            self.save_student_checkpoint(is_best=True)
                            
            # Log epoch metrics
            if self.is_main_process:
                avg_metrics = {
                    k: v / num_steps for k, v in epoch_metrics.items()
                }
                logger.info(
                    f"Epoch {epoch + 1}, Loss: {epoch_loss / num_steps:.4f}, "
                    f"Metrics: {avg_metrics}"
                )
                
            # Synchronize processes
            if self.parallel_config.local_rank != -1:
                torch.distributed.barrier()
                
        # Final save
        if self.is_main_process:
            self.save_student_checkpoint()
            
    def save_student_checkpoint(self, is_best: bool = False):
        """Save student model checkpoint"""
        checkpoint = {
            'step': self.global_step,
            'student_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'distill_config': self.distill_config,
            'parallel_config': self.parallel_config,
            'training_config': self.training_config
        }
        
        save_path = os.path.join(
            self.training_config.get('checkpoint_dir', 'checkpoints'),
            'student_model.pt'
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = os.path.join(
                self.training_config.get('checkpoint_dir', 'checkpoints'),
                'student_model_best.pt'
            )
            torch.save(checkpoint, best_path)
            
    @classmethod
    def from_pretrained(
        cls,
        teacher_path: str,
        student_config_path: str,
        train_dataset: VishwamAIDataset,
        val_dataset: Optional[VishwamAIDataset] = None,
        distill_config: Optional[DistillationConfig] = None,
        parallel_config: Optional[ParallelConfig] = None,
        training_config: Optional[Dict[str, Any]] = None
    ) -> 'DistillationTrainer':
        """
        Initialize trainer from pretrained teacher and student config.
        
        Args:
            teacher_path: Path to pretrained teacher model
            student_config_path: Path to student model config
            train_dataset: Training dataset
            val_dataset: Validation dataset
            distill_config: Distillation configuration
            parallel_config: Parallel training configuration
            training_config: Training hyperparameters
            
        Returns:
            Initialized trainer instance
        """
        # Load teacher model
        teacher_model = VishwamAITransformer.from_pretrained(teacher_path)
        
        # Load student config and initialize model
        with open(student_config_path) as f:
            student_config = yaml.safe_load(f)
        student_model = VishwamAITransformer(**student_config)
        
        return cls(
            teacher_model=teacher_model,
            student_model=student_model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            distill_config=distill_config,
            parallel_config=parallel_config,
            training_config=training_config
        )

def main():
    """Main function to run distillation"""
    # Load teacher model
    teacher_model = VishwamAITransformer.from_pretrained('path/to/teacher/model')
    
    # Initialize smaller student model
    student_config = {
        'vocab_size': 50000,
        'embed_dim': 384,   # Half the teacher's dimension
        'num_layers': 6,    # Half the teacher's layers
        'num_heads': 6,
        'ff_dim': 1536
    }
    student_model = VishwamAITransformer(**student_config)
    
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
    
    # Configure distillation
    distill_config = DistillationConfig(
        temperature=2.0,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        feature_layers=[1, 3, 5],  # Layers to match
        use_attention_distill=True
    )
    
    # Configure parallel training
    parallel_config = ParallelConfig(
        backend='nccl',
        world_size=torch.cuda.device_count(),
        model_parallel_size=1,
        pipeline_parallel_size=1,
        mixed_precision=True,
        optimize_memory=True
    )
    
    # Training config
    training_config = {
        'batch_size': 32,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_steps': 1000,
        'total_steps': 100000,
        'eval_steps': 500,
        'save_steps': 1000,
        'checkpoint_dir': 'distillation_checkpoints'
    }
    
    # Initialize trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        distill_config=distill_config,
        parallel_config=parallel_config,
        training_config=training_config
    )
    
    # Run distillation
    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main()