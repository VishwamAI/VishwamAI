import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union
from tqdm import tqdm
import os
from pathlib import Path
from .conceptual_tokenizer import ConceptualTokenizer

class VishwamaiTrainer:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: ConceptualTokenizer,
        train_dataset: DataLoader,
        eval_dataset: Optional[DataLoader] = None,
        device: str = "cuda",
        optimizer_class: Any = torch.optim.AdamW,
        scheduler_class: Any = torch.optim.lr_scheduler.CosineAnnealingLR,
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = device
        self.optimizer = optimizer_class(model.parameters())
        self.scheduler = scheduler_class(self.optimizer, T_max=1000)
        self.use_wandb = use_wandb
        # Add support for tracking best model
        self.best_eval_loss = float('inf')
        self.best_model_path = None
        
    def train(
        self,
        num_epochs: int,
        save_dir: Union[str, Path],
        evaluation_steps: int = 100,
        save_steps: int = 1000,
        logging_steps: int = 10,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        fp16: bool = True,
    ):
        """Train the model"""
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed precision training
        scaler = torch.amp.GradScaler('cuda') if fp16 else None
        
        global_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            with tqdm(self.train_dataset, desc=f"Epoch {epoch+1}") as pbar:
                for step, batch in enumerate(pbar):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Compute loss using the compute_loss method
                    loss = self.compute_loss(batch)
                    
                    if fp16:
                        with torch.amp.autocast('cuda'):
                            loss = self.compute_loss(batch)
                            loss = loss / gradient_accumulation_steps
                            
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0:
                            scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            
                    epoch_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item()})
                    
                    # Improved logging
                    log_data = {
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "global_step": global_step
                    }

                    
                    # Evaluation
                    if global_step % evaluation_steps == 0:
                        eval_results = self.evaluate()
                        self.model.train()
                        
                        # Save best model
                        if eval_results and eval_results["eval_loss"] < self.best_eval_loss:
                            self.best_eval_loss = eval_results["eval_loss"]
                            self.best_model_path = save_dir / f"best_model"
                            self.save_model(self.best_model_path)
                    
                    # Save model
                    if global_step % save_steps == 0:
                        self.save_model(save_dir / f"checkpoint-{global_step}")
                        
                    global_step += 1
                    
            # End of epoch
            epoch_loss = epoch_loss / len(self.train_dataset)
            print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}")
            
            # Update learning rate
            self.scheduler.step()
            
        # Save final model
        self.save_model(save_dir / "final_model")
        
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model"""
        if self.eval_dataset is None:
            return {}
            
        self.model.eval()
        eval_loss = 0
        
        with torch.no_grad():
            for batch in self.eval_dataset:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.compute_loss(batch)
                eval_loss += loss.item()
                
        eval_loss = eval_loss / len(self.eval_dataset)
        return {"eval_loss": eval_loss}
    
    def save_model(self, path: Union[str, Path]):
        """Save model and tokenizer"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)  # Changed from 'save' to 'save_pretrained'
        
        # Save training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, path / "training_state.pt")
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform a single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device if needed
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        loss = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the loss for a batch of data"""
        # Get labels but keep a copy
        labels = batch['labels']
        batch_size, seq_length = labels.size()
        
        # Get required inputs
        model_inputs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }
        
        # Only add concept_ids if present
        if 'concept_ids' in batch:
            model_inputs['concept_ids'] = batch['concept_ids']
        
        # Forward pass
        outputs = self.model(**model_inputs)
        print(f"outputs shape before reshape: {outputs.shape}")
        print(f"labels shape before reshape: {labels.shape}")
        # Get sequence lengths
        batch_size, seq_length_output, vocab_size = outputs.size()
        batch_size_labels, seq_length_labels = labels.size()
        
        # Use the minimum sequence length between outputs and labels
        min_seq_length = min(seq_length_output, seq_length_labels)
        
        # Fix sequence length mismatch
        outputs = outputs[:, :min_seq_length, :]  # Truncate outputs
        labels = labels[:, :min_seq_length]       # Truncate labels
        
        # Reshape both outputs and labels
        outputs = outputs.reshape(-1, vocab_size)  # (batch_size * seq_length, vocab_size)
        labels = labels.reshape(-1)  # (batch_size * seq_length)
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        return loss
