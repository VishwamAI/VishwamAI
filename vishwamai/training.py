import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Union
import wandb
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
    ):
        """Train the model"""
        if self.use_wandb:
            wandb.init(project="vishwamai")
            
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup mixed precision training
        scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        global_step = 0
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            with tqdm(self.train_dataset, desc=f"Epoch {epoch+1}") as pbar:
                for step, batch in enumerate(pbar):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    with torch.cuda.amp.autocast(enabled=fp16):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                        loss = loss / gradient_accumulation_steps
                    
                    if fp16:
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
                    if global_step % logging_steps == 0:
                        if self.use_wandb:
                            wandb.log({
                                "loss": loss.item(),
                                "lr": self.scheduler.get_last_lr()[0],
                                "epoch": epoch,
                                "global_step": global_step
                            })
                    
                    # Evaluation
                    if global_step % evaluation_steps == 0:
                        eval_results = self.evaluate()
                        self.model.train()
                        if self.use_wandb:
                            wandb.log({"eval": eval_results})
                    
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
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
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
        self.tokenizer.save(path / "tokenizer.json")
        
        # Save training state
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
        }, path / "training_state.pt")
