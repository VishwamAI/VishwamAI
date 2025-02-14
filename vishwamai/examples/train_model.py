#!/usr/bin/env python3
"""
Example script for training VishwamAI model with optimized settings.
"""

import os
import argparse
import json
from typing import Optional

import torch
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from vishwamai.model import Transformer
from vishwamai.model_utils import load_model, get_gpu_memory

def get_gpu_config(gpu_type: Optional[str] = None):
    """Get configuration based on GPU type."""
    config_path = os.path.join(os.path.dirname(__file__), 
                              "../configs/config_optimized.json")
    with open(config_path) as f:
        config = json.load(f)
    
    if not gpu_type:
        # Auto-detect GPU type
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0).lower()
            if 'a100' in gpu_name:
                gpu_type = 'A100_optimized'
            elif 'v100' in gpu_name:
                gpu_type = 'V100_optimized'
            else:
                gpu_type = 'T4_optimized'
    
    # Get GPU-specific config
    gpu_config = config['colab_specific'][gpu_type]
    config['model_config'].update(gpu_config)
    return config

def get_training_config(config: dict, gpu_memory: float):
    """Generate training configuration based on available GPU memory."""
    total_params = config['model_config'].get('dim', 2048) * config['model_config'].get('n_layers', 27)
    memory_per_batch = total_params * 4 / 1e9  # Rough estimate in GB
    
    # Scale batch size based on available memory
    batch_size = max(1, int(gpu_memory * 0.4 / memory_per_batch))
    grad_accum = max(1, 32 // batch_size)  # Target effective batch size of 32
    
    return {
        'batch_size': batch_size,
        'gradient_accumulation_steps': grad_accum,
        'learning_rate': 1e-4,
        'weight_decay': 0.1,
        'warmup_steps': 100,
        'eval_steps': 100,
        'save_steps': 500,
        'logging_steps': 10
    }

def setup_training(args):
    """Setup training environment and configuration."""    
    # Get GPU-optimized config
    config = get_gpu_config(args.gpu_type)
    
    # Load model with optimized settings
    model = load_model(
        config_path=args.config_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        pretrained_path=args.pretrained_path,
        use_cache=not args.disable_cache
    )
    
    # Load datasets
    datasets = {
        "train": load_dataset(args.train_dataset, split="train"),
        "validation": load_dataset(args.eval_dataset, split="validation")
    }
    
    # Get training configuration
    gpu_memory = get_gpu_memory()
    training_config = get_training_config(config, gpu_memory)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=training_config['eval_steps'],
        save_steps=training_config['save_steps'],
        logging_steps=training_config['logging_steps'],
        fp16=False,  # Use bf16 instead since model uses bfloat16
        bf16=True,
        gradient_checkpointing=True,
        report_to="none"  # Disable wandb reporting
    )
    
    return model, datasets, training_args

def main():
    parser = argparse.ArgumentParser(description="Train VishwamAI model")
    parser.add_argument("--config_path", type=str, default="configs/config_optimized.json")
    parser.add_argument("--train_dataset", type=str, default="gsm8k")
    parser.add_argument("--eval_dataset", type=str, default="cais/mmlu")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--gpu_type", type=str, choices=["T4_optimized", "V100_optimized", "A100_optimized"])
    parser.add_argument("--pretrained_path", type=str)
    parser.add_argument("--disable_cache", action="store_true")
    args = parser.parse_args()
    
    # Setup training
    model, datasets, training_args = setup_training(args)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"]
    )
    
    # Start training
    trainer.train()
    
    # Save final model
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
