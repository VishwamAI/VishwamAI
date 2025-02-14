import math
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from .model import VishwamaiModel, Block

@dataclass
class TrainingArgs:
    """Arguments for training configuration"""
    output_dir: str = "checkpoints"
    num_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 5000
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    logging_steps: int = 100
    use_fsdp: bool = True
    mixed_precision: bool = True
    cpu_offload: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42

class TrainerState:
    """Maintains training state and metrics"""
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.metrics: Dict[str, float] = {}
        self.optimizer_state: Optional[Dict] = None

    def save(self, path: Union[str, Path]):
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'metrics': self.metrics,
            'optimizer_state': self.optimizer_state
        }, path)

    def load(self, path: Union[str, Path]):
        state_dict = torch.load(path)
        self.epoch = state_dict['epoch']
        self.global_step = state_dict['global_step']
        self.best_loss = state_dict['best_loss']
        self.metrics = state_dict['metrics']
        self.optimizer_state = state_dict['optimizer_state']

class Trainer:
    def __init__(
        self,
        model: VishwamaiModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        args: Optional[TrainingArgs] = None
    ):
        self.args = args or TrainingArgs()
        self.state = TrainerState()
        
        # Set up distributed training
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        # Setup model with FSDP if enabled
        if self.args.use_fsdp and self.world_size > 1:
            mixed_precision_policy = None
            if self.args.mixed_precision:
                mixed_precision_policy = MixedPrecision(
                    param_dtype=torch.float8_e4m3fn if model.args.dtype == "fp8" else torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16
                )
            
            self.model = FSDP(
                model,
                auto_wrap_policy=transformer_auto_wrap_policy(transformer_layer_cls={Block}),
                mixed_precision=mixed_precision_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=CPUOffload(offload_params=self.args.cpu_offload),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                device_id=torch.cuda.current_device() if torch.cuda.is_available() else None
            )
        else:
            self.model = model

        # Enable gradient checkpointing if requested
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Setup optimizer
        decay_parameters = [p for p in self.model.parameters() if p.ndim >= 2]
        nodecay_parameters = [p for p in self.model.parameters() if p.ndim < 2]
        
        self.optimizer = optim.AdamW([
            {'params': decay_parameters, 'weight_decay': self.args.weight_decay},
            {'params': nodecay_parameters, 'weight_decay': 0.0}
        ], lr=self.args.learning_rate)

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Create output directory
        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_scheduler(self, num_training_steps: int):
        """Creates a learning rate scheduler"""
        def lr_lambda(current_step: int):
            if current_step < self.args.warmup_steps:
                return float(current_step) / float(max(1, self.args.warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - self.args.warmup_steps)
                ),
            )
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs a single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(batch['input_ids'])
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            batch['labels'].view(-1)
        )
        
        # Scale loss for gradient accumulation
        loss = loss / self.args.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.args.gradient_accumulation_steps

    def evaluation_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Performs a single evaluation step"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch['input_ids'])
            loss = nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch['labels'].view(-1)
            )
        return loss.item()

    def save_checkpoint(self, path: Union[str, Path]):
        """Saves a training checkpoint"""
        if isinstance(self.model, FSDP):
            self.model.state_dict_type = torch.nn.Module
            state_dict = self.model.state_dict()
        else:
            state_dict = self.model.state_dict()
            
        if self.rank == 0:
            torch.save({
                'model_state_dict': state_dict,
                'training_state': self.state,
            }, path)

    def train(self):
        """Main training loop"""
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / self.args.gradient_accumulation_steps
        )
        num_training_steps = self.args.num_epochs * num_update_steps_per_epoch
        
        # Create scheduler
        scheduler = self.create_scheduler(num_training_steps)
        
        # Training loop
        for epoch in range(self.state.epoch, self.args.num_epochs):
            self.state.epoch = epoch
            epoch_loss = 0
            step_in_epoch = 0
            
            for step, batch in enumerate(self.train_dataloader):
                # Move batch to device
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                
                loss = self.train_step(batch)
                epoch_loss += loss
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # Clip gradients
                    if self.args.max_grad_norm > 0:
                        if isinstance(self.model, FSDP):
                            self.model.clip_grad_norm_(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm
                            )
                    
                    # Optimizer step
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.state.global_step += 1
                    step_in_epoch += 1
                    
                    # Logging
                    if self.state.global_step % self.args.logging_steps == 0:
                        avg_loss = epoch_loss / step_in_epoch
                        lr = scheduler.get_last_lr()[0]
                        
                        if self.rank == 0:
                            print(f"Epoch {epoch} Step {self.state.global_step}: "
                                  f"loss = {avg_loss:.4f}, lr = {lr:.2e}")
                    
                    # Saving
                    if self.state.global_step % self.args.save_steps == 0:
                        save_path = self.output_dir / f"checkpoint-{self.state.global_step}"
                        self.save_checkpoint(save_path)
                        
                    # Evaluation
                    if self.eval_dataloader is not None and self.state.global_step % self.args.save_steps == 0:
                        eval_loss = self.evaluate()
                        if eval_loss < self.state.best_loss:
                            self.state.best_loss = eval_loss
                            best_path = self.output_dir / "checkpoint-best"
                            self.save_checkpoint(best_path)
            
            # Save at end of epoch
            save_path = self.output_dir / f"checkpoint-epoch-{epoch}"
            self.save_checkpoint(save_path)
    
    def evaluate(self) -> float:
        """Evaluates the model on the evaluation dataset"""
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataloader provided")
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                loss = self.evaluation_step(batch)
                total_loss += loss
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if self.rank == 0:
            print(f"Evaluation loss: {avg_loss:.4f}")
            
        return avg_loss

    def reset_memory_states(self):
        """Resets memory states in all model layers"""
        for layer in self.model.layers:
            if hasattr(layer, 'memory'):
                layer.memory.reset_memory()
            if hasattr(layer, 'cache'):
                layer.cache.reset_cache()
