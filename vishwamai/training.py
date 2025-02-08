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
        resume_from_checkpoint: Optional[Union[str, Path]] = None
    ):
        """Train the model"""
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize or resume training state
        global_step = 0
        best_eval_loss = float('inf')
        
        if resume_from_checkpoint:
            checkpoint_path = Path(resume_from_checkpoint)
            if checkpoint_path.exists():
                # Load model
                self.model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
                
                # Load training state
                training_state = torch.load(checkpoint_path / "training_state.pt")
                self.optimizer.load_state_dict(training_state["optimizer"])
                self.scheduler.load_state_dict(training_state["scheduler"])
                if "global_step" in training_state:
                    global_step = training_state["global_step"]
                if "best_eval_loss" in training_state:
                    best_eval_loss = training_state["best_eval_loss"]
                    
                print(f"Resumed from checkpoint: {checkpoint_path}")
                
        # Setup mixed precision training
        scaler = torch.amp.GradScaler(enabled=fp16 and torch.cuda.is_available())
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            with tqdm(self.train_dataset, desc=f"Epoch {epoch+1}") as pbar:
                for step, batch in enumerate(pbar):
                    # Compute loss using the compute_loss method which handles device transfers
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
                    
                    # Logging
                    
                    # Evaluation
                    if global_step % evaluation_steps == 0:
                        eval_results = self.evaluate()
                        self.model.train()
                    
                    # Evaluation and model saving
                    if global_step % evaluation_steps == 0:
                        eval_results = self.evaluate()
                        self.model.train()
                        
                        if eval_results:  # If evaluation was performed
                            current_eval_loss = eval_results["eval_loss"]
                            
                            # Save best model
                            if current_eval_loss < best_eval_loss:
                                best_eval_loss = current_eval_loss
                                self.save_model(
                                    save_dir / "best_model",
                                    {
                                        "global_step": global_step,
                                        "best_eval_loss": best_eval_loss
                                    }
                                )
                            
                    # Regular checkpoint saving
                    if global_step > 0 and global_step % save_steps == 0:
                        self.save_model(
                            save_dir / f"checkpoint-{global_step}",
                            {
                                "global_step": global_step,
                                "best_eval_loss": best_eval_loss
                            }
                        )
                        
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
                # compute_loss handles device transfers
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
        
        # Forward pass - compute_loss handles device transfers
        loss = self.compute_loss(batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
        
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the loss for a batch of data"""
        # Validate required fields
        required_fields = ['input_ids', 'attention_mask', 'labels']
        for field in required_fields:
            if field not in batch:
                raise ValueError(f"Missing required field: {field}")
        
        # Get labels and remove from inputs
        labels = batch.pop('labels')
        orig_shape = labels.shape
        labels = labels.view(-1)  # Flatten labels
        
        # Handle device transfers properly
        model_inputs = {}
        for key in ['input_ids', 'attention_mask', 'concept_ids']:
            if key in batch:
                # Pin memory first if on CPU and going to CUDA
                if self.device == 'cuda' and batch[key].device.type == 'cpu' and not batch[key].is_pinned():
                    batch[key] = batch[key].pin_memory()
                # Then transfer to device
                model_inputs[key] = batch[key].to(self.device, non_blocking=True)
        
        # Forward pass and get logits from output dictionary
        outputs = self.model(**model_inputs)
        logits = outputs['logits']
        
        # Handle shape mismatch
        if logits.size(1) != orig_shape[1]:
            # Reshape logits to match target shape
            logits = logits.view(orig_shape[0], orig_shape[1], -1)
        
        # Reshape for loss computation
        logits = logits.view(-1, logits.size(-1))  # (batch_size * seq_length, vocab_size)
        
        # Compute loss with shape validation
        if logits.size(0) != labels.size(0):
            raise ValueError(f"Logits and labels size mismatch: {logits.size()} vs {labels.size()}")
        
        loss = torch.nn.functional.cross_entropy(logits, labels)
        
        # Add labels back to batch for potential later use
        batch['labels'] = labels.view(model_inputs['input_ids'].size(0), -1)  # Restore original shape
        
        return loss
