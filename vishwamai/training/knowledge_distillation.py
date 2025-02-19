import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict

class DistillationTrainer:
    """Knowledge distillation trainer for model compression."""
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.teacher_model.eval()
        
        # Setup optimizer for student
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute distillation loss with optional hard labels."""
        # Compute soft targets distillation
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # If labels provided, compute hard targets loss
        if labels is not None:
            task_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * task_loss + (1 - self.alpha) * distillation_loss
        else:
            total_loss = distillation_loss
            task_loss = torch.tensor(0.0)
            
        return {
            'total_loss': total_loss,
            'distillation_loss': distillation_loss,
            'task_loss': task_loss
        }
        
    def train(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str = "student_checkpoints"
    ):
        """Train student model using knowledge distillation."""
        import os
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        device = next(self.student_model.parameters()).device
        
        for epoch in range(num_epochs):
            self.student_model.train()
            total_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**batch)
                    
                # Get student predictions
                student_outputs = self.student_model(**batch)
                
                # Compute losses
                losses = self.compute_distillation_loss(
                    student_outputs['logits'],
                    teacher_outputs['logits'],
                    batch.get('labels')
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                losses['total_loss'].backward()
                self.optimizer.step()
                
                total_loss += losses['total_loss'].item()
                
            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"student_model_epoch_{epoch+1}.pt")
                torch.save(self.student_model.state_dict(), checkpoint_path)
                
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
