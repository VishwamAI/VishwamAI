"""
Optimized training script for Google Colab A100 environments
"""
import os
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import wandb
from tqdm.auto import tqdm

from ..config.a100_config import A100TrainingConfig, VISHWAMAI_A100
from ..model import create_model
from ..utils.t4_utils import enable_t4_optimizations, get_device_capabilities

class ColabA100Trainer:
    def __init__(
        self,
        config: A100TrainingConfig,
        model_config: Optional[ModelArgs] = None,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.config = config
        self.model_config = model_config or VISHWAMAI_A100
        self.checkpoint_dir = checkpoint_dir
        
        # Setup device and optimizations
        self.setup_environment()
        
        # Initialize model and training components
        self.model = self.initialize_model()
        self.optimizer = self.create_optimizer()
        self.scaler = GradScaler() if self.config.fp8_training else None
        
    def setup_environment(self):
        """Configure training environment"""
        enable_t4_optimizations()
        capabilities = get_device_capabilities()
        
        if not capabilities.get("a100_available"):
            print("Warning: A100 GPU not detected. Performance may be suboptimal.")
            
        # Set memory optimization flags
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def initialize_model(self):
        """Initialize model with optimized settings"""
        model = create_model(config=self.model_config)
        model.cuda()
        
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        return model
        
    def create_optimizer(self):
        """Create AdamW optimizer with weight decay"""
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm"])],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm"])],
                "weight_decay": 0.0,
            },
        ]
        
        return torch.optim.AdamW(param_groups, lr=self.config.learning_rate)
        
    def train(self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None):
        """Main training loop"""
        self.model.train()
        progress_bar = tqdm(total=self.config.max_steps, desc="Training")
        
        for step, batch in enumerate(train_dataloader):
            if step >= self.config.max_steps:
                break
                
            # Forward pass with automatic mixed precision
            with autocast(dtype=torch.bfloat16):
                loss = self.training_step(batch)
                
            # Backward pass with gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
            # Logging and evaluation
            if step % self.config.logging_steps == 0:
                self.log_metrics({"loss": loss.item()}, step)
                
            if step % self.config.save_steps == 0:
                self.save_checkpoint(step)
                
            if eval_dataloader and step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataloader)
                self.log_metrics(eval_metrics, step)
                
            progress_bar.update(1)
            
    def training_step(self, batch):
        """Single training step"""
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = self.model(**batch)
        return outputs["loss"]
        
    def evaluate(self, eval_dataloader):
        """Evaluation loop"""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs["loss"].item()
                
        self.model.train()
        return {"eval_loss": total_loss / len(eval_dataloader)}
        
    def save_checkpoint(self, step):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-{step}")
        self.model.save_pretrained(checkpoint_path)
        
    def log_metrics(self, metrics, step):
        """Log metrics to WandB"""
        if wandb.run is not None:
            wandb.log(metrics, step=step)