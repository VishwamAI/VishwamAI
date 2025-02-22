"""
Trainer class for Vishwamai model
"""
import os
import time
from typing import Dict, Any, Optional, List, Tuple
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from ..model import VishwamaiModel
from ..config.model_config import PrecisionMode, PrecisionConfig
from .optimization import GradScaler, create_optimizer, create_scheduler
from .utils import setup_training, setup_logger, log_metrics, save_checkpoint

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer for Vishwamai model with precision options
    """
    def __init__(
        self,
        model: VishwamaiModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: str = "outputs"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        self.output_dir = output_dir
        
        # Setup distributed training
        self.setup_distributed()
        
        # Setup precision options
        self.setup_precision()
        
        # Create optimizer and scheduler
        num_training_steps = len(train_dataloader) * self.config.get("num_epochs", 1)
        self.optimizer = create_optimizer(model, self.config)
        self.scheduler = create_scheduler(
            self.optimizer,
            self.config,
            num_training_steps
        )
        
        # Setup logging
        self.logger = setup_logger(
            "vishwamai_trainer",
            os.path.join(output_dir, "train.log")
        )
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_eval_metric = float("inf")
        
    def setup_distributed(self) -> None:
        """Setup distributed training if needed"""
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        if self.world_size > 1:
            setup_training(self.local_rank, self.world_size)
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank
            )
            
    def setup_precision(self) -> None:
        """Setup precision options for training"""
        # Get precision config
        if "precision" not in self.config:
            self.config["precision"] = PrecisionConfig()
            
        precision_config = self.config["precision"]
        if not isinstance(precision_config, PrecisionConfig):
            precision_config = PrecisionConfig(**precision_config)
            
        self.precision_config = precision_config
        
        # Setup gradient scaler for mixed precision
        self.scaler = GradScaler(precision_config=precision_config)
        
        # Log precision settings
        logger.info(
            f"Training with precision mode: {precision_config.mode.value}, "
            f"mixed_precision: {precision_config.mixed_precision}, "
            f"gradient_precision: {precision_config.gradient_precision}"
        )
            
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            batch: Dictionary of input tensors
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Forward pass with automatic mixed precision
        with autocast(
            enabled=self.precision_config.mixed_precision,
            dtype=torch.float16 if self.precision_config.mode == PrecisionMode.FP16 else torch.bfloat16
        ):
            outputs = self.model(**batch)
            loss = outputs["loss"]
            
            # Scale loss for gradient accumulation
            if self.config.get("gradient_accumulation_steps", 1) > 1:
                loss = loss / self.config["gradient_accumulation_steps"]
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Update weights if gradient accumulation is done
        if (self.global_step + 1) % self.config.get("gradient_accumulation_steps", 1) == 0:
            # Clip gradients
            if self.config.get("max_grad_norm", None) is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"]
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Get metrics
        metrics = {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]["lr"],
            "grad_scale": self.scaler.get_scale()
        }
        
        return metrics
        
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.eval_dataloader is None:
            return {}
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            # Move batch to device
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            
            # Forward pass with appropriate precision
            with autocast(
                enabled=self.precision_config.mixed_precision,
                dtype=torch.float16 if self.precision_config.mode == PrecisionMode.FP16 else torch.bfloat16
            ):
                outputs = self.model(**batch)
                loss = outputs["loss"]
            
            total_loss += loss.item()
            num_batches += 1
            
        metrics = {"eval_loss": total_loss / num_batches}
        return metrics
        
    def save_model(self, save_dir: Optional[str] = None) -> None:
        """Save model checkpoint"""
        save_dir = save_dir or self.output_dir
        save_checkpoint(
            model=self.model.module if hasattr(self.model, "module") else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            metrics={"epoch": self.epoch},
            step=self.global_step,
            save_dir=save_dir
        )
        
    def train(self) -> Dict[str, List[float]]:
        """
        Train model
        
        Returns:
            Dictionary of training history
        """
        # Training history
        history = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": []
        }
        
        num_epochs = self.config.get("num_epochs", 1)
        logger.info(f"Starting {num_epochs} epochs of training...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_metrics = []
            epoch_start = time.time()
            
            # Training loop
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                # Training step
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)
                
                # Logging
                if (batch_idx + 1) % self.config.get("logging_steps", 100) == 0:
                    avg_metrics = {
                        k: sum(m[k] for m in epoch_metrics[-100:]) / len(epoch_metrics[-100:])
                        for k in metrics
                    }
                    log_metrics(avg_metrics, self.global_step, self.logger)
                    
                # Evaluation
                if (self.global_step + 1) % self.config.get("eval_steps", 500) == 0:
                    eval_metrics = self.evaluate()
                    log_metrics(eval_metrics, self.global_step, self.logger)
                    
                    # Save best model
                    if eval_metrics.get("eval_loss", float("inf")) < self.best_eval_metric:
                        self.best_eval_metric = eval_metrics["eval_loss"]
                        self.save_model(os.path.join(self.output_dir, "best"))
                        
                # Save checkpoint
                if (self.global_step + 1) % self.config.get("save_steps", 1000) == 0:
                    self.save_model()
                    
                self.global_step += 1
                
            # End of epoch
            avg_loss = sum(m["loss"] for m in epoch_metrics) / len(epoch_metrics)
            avg_lr = sum(m["lr"] for m in epoch_metrics) / len(epoch_metrics)
            eval_metrics = self.evaluate()
            
            history["train_loss"].append(avg_loss)
            history["eval_loss"].append(eval_metrics.get("eval_loss", 0.0))
            history["learning_rate"].append(avg_lr)
            
            # Log epoch metrics
            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} completed in {time.time() - epoch_start:.2f}s. "
                f"Train loss: {avg_loss:.4f}, Eval loss: {eval_metrics.get('eval_loss', 0.0):.4f}"
            )
            
            # Save epoch checkpoint
            self.save_model(os.path.join(self.output_dir, f"epoch_{epoch+1}"))
            
        return history
