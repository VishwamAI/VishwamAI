#!/usr/bin/env python3
"""Main training script for Vishwamai model."""

import argparse
import logging
import os
from pathlib import Path
import yaml
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
from dataclasses import dataclass

from vishwamai.data.tokenization import SPTokenizer
from vishwamai.data.dataloader import create_dataloaders
from vishwamai.model.transformer.model import VishwamaiModel
from vishwamai.training.optimizer import AdamWOptimizer, ShardedOptimizer
from vishwamai.training.scheduling import CosineAnnealingWarmupScheduler
from vishwamai.training.distributed import TPUManager, setup_tpu
from vishwamai.training.callbacks import (
    ModelCheckpoint, EarlyStopping, LRSchedulerCallback
)
from vishwamai.utils.logging import setup_logging
from vishwamai.utils.profiling import MemoryTracker, PerformanceProfiler

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration."""
    model_config_path: str
    data_config_path: str
    tokenizer_path: str
    train_data_dir: str
    val_data_dir: str
    output_dir: str
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    warmup_steps: int
    max_grad_norm: float
    mixed_precision: bool
    seed: int

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train Vishwamai model")
    parser.add_argument(
        "--model-config",
        type=str,
        required=True,
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=True,
        help="Path to data config YAML"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer model"
    )
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Directory containing training data"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Directory containing validation data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save model outputs"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of steps for gradient accumulation"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Peak learning rate"
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=2000,
        help="Number of warmup steps"
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()

def train(config: TrainingConfig):
    """Main training function.
    
    Args:
        config (TrainingConfig): Training configuration
    """
    # Setup TPU
    setup_tpu()
    tpu_manager = TPUManager(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    
    # Load configurations
    with open(config.model_config_path) as f:
        model_config = yaml.safe_load(f)
    with open(config.data_config_path) as f:
        data_config = yaml.safe_load(f)
        
    # Create tokenizer and dataloaders
    tokenizer = SPTokenizer.from_pretrained(config.tokenizer_path)
    train_loader, val_loader = create_dataloaders(
        train_dir=config.train_data_dir,
        val_dir=config.val_data_dir,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=4,
        **data_config["dataloader"]
    )
    
    # Create model
    model = VishwamaiModel(model_config)
    model = model.to(tpu_manager.device)
    
    # Setup optimizer and scheduler
    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
        max_grad_norm=config.max_grad_norm
    )
    optimizer = ShardedOptimizer(
        optimizer_cls=type(optimizer),
        model_params=model.parameters(),
        optimizer_kwargs={
            "lr": config.learning_rate,
            "weight_decay": 0.01,
            "max_grad_norm": config.max_grad_norm
        },
        grad_accum_steps=config.gradient_accumulation_steps
    )
    
    total_steps = len(train_loader) * config.num_epochs
    scheduler = CosineAnnealingWarmupScheduler(
        optimizer,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        max_lr=config.learning_rate
    )
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            CheckpointConfig(
                dirpath=config.output_dir,
                monitor="val_loss",
                save_top_k=2
            )
        ),
        EarlyStopping(
            EarlyStoppingConfig(
                monitor="val_loss",
                patience=3
            )
        ),
        LRSchedulerCallback(
            scheduler,
            LRSchedulerConfig(
                interval="step"
            )
        )
    ]
    
    # Setup profilers
    memory_tracker = MemoryTracker()
    performance_profiler = PerformanceProfiler()
    
    # Training loop
    global_step = 0
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        
        train_loader_tpu = tpu_manager.get_parallel_loader(train_loader)
        
        for batch_idx, batch in enumerate(train_loader_tpu):
            with performance_profiler.profile_step():
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Step optimizer
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    tpu_manager.step_optimizer(optimizer, scheduler)
                    optimizer.zero_grad()
                    global_step += 1
                    
                train_loss += loss.item()
                
                # Log metrics
                if (batch_idx + 1) % 100 == 0:
                    mem_stats = memory_tracker.get_memory_stats()
                    perf_stats = performance_profiler.get_performance_stats()
                    logger.info(
                        f"Epoch {epoch}, Step {global_step}, "
                        f"Loss: {loss.item():.4f}, "
                        f"Memory Used: {mem_stats['used_gb']:.2f}GB, "
                        f"Step Time: {perf_stats['avg_step_time']:.2f}s"
                    )
                    
        # Validation
        model.eval()
        val_loss = 0.0
        val_loader_tpu = tpu_manager.get_parallel_loader(val_loader)
        
        with torch.no_grad():
            for batch in val_loader_tpu:
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                
        val_loss /= len(val_loader)
        
        # Callback handling
        metrics = {
            "train_loss": train_loss / len(train_loader),
            "val_loss": val_loss
        }
        
        stop_training = False
        for callback in callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback.on_validation_end(
                    model, optimizer, scheduler, metrics, epoch, global_step
                )
            elif isinstance(callback, EarlyStopping):
                if callback.on_validation_end(metrics, epoch):
                    stop_training = True
                    break
                    
        if stop_training:
            break
            
    logger.info("Training completed")
    
if __name__ == "__main__":
    args = parse_args()
    setup_logging()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    
    config = TrainingConfig(
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        tokenizer_path=args.tokenizer_path,
        train_data_dir=args.train_data,
        val_data_dir=args.val_data,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        seed=args.seed
    )
    
    train(config)
