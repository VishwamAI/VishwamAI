#!/usr/bin/env python3
"""
Script to distill knowledge from Qwen-32B to VishwamAI-7B using safetensors format.
Optimized for TPU execution with JAX/Flax.
"""

import os
import argparse
import logging
from dataclasses import asdict
from typing import Dict, Any, Optional
import json
from functools import partial
import time
from tqdm import tqdm
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import flax
import optax
from flax.training import train_state, checkpoints
from safetensors.flax import load_file, save_file

from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.distillation import VishwamaiShaalaTrainer, VishwamaiGuruKnowledge
from vishwamai.data_utils import create_train_dataloader, create_val_dataloader
from vishwamai.convert import SafeModelConverter
from vishwamai.loss_functions import cross_entropy_loss, kl_divergence_loss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation from Qwen-32B to VishwamAI-7B using safetensors"
    )
    parser.add_argument(
        "--teacher_model_path", type=str, required=True, help="Path to Qwen-32B model in safetensors format"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the distilled model"
    )
    parser.add_argument(
        "--train_data_path", type=str, required=True, help="Path to training dataset"
    )
    parser.add_argument(
        "--val_data_path", type=str, default=None, help="Path to validation dataset (optional)"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to tokenizer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Training batch size per device"
    )
    parser.add_argument(
        "--grad_accum_steps", type=int, default=16, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--num_train_steps", type=int, default=10000, help="Total training steps"
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=1000, help="Warmup steps"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Peak learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--temperature", type=float, default=2.0, help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha_kd", type=float, default=0.8, help="KL loss weight"
    )
    parser.add_argument(
        "--alpha_ce", type=float, default=0.2, help="CE loss weight"
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save checkpoint every X steps"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=500, help="Evaluate every X steps"
    )
    parser.add_argument(
        "--log_steps", type=int, default=50, help="Log metrics every X steps"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="vishwamai-distillation", help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="W&B run name"
    )
    parser.add_argument(
        "--use_tot", action="store_true", help="Use Tree of Thoughts for enhanced distillation"
    )
    parser.add_argument(
        "--use_error_correction", action="store_true", help="Use error correction during distillation"
    )
    parser.add_argument(
        "--use_eplb", action="store_true", help="Use EPLB for better expert balancing"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 precision (default is bfloat16 for TPU)"
    )
    return parser.parse_args()

def create_distillation_config(args):
    """Creates config for distillation from command-line arguments."""
    # Teacher model config (QwQ-32B)
    teacher_config = ModelConfig(
        vocab_size=151936,     # QwQ-32B vocab size
        hidden_size=7168,      # 32B model size
        num_layers=60,         # QwQ architecture
        num_attention_heads=56,
        intermediate_size=28672,
        max_position_embeddings=32768,
        use_flash_attention=True,
        use_rope=True,
        use_gqa=True,
        num_key_value_heads=8,  # GQA for memory efficiency
        dtype="bfloat16" if not args.fp16 else "float16",
        use_dualpipe=True,     # Enable dualpipe for efficient processing
        use_eplb=args.use_eplb,
        gradient_checkpointing=True
    )
    
    # Student model config (7B)
    student_config = ModelConfig(
        vocab_size=32000,
        hidden_size=4096,      # 7B model size
        num_layers=32,
        num_attention_heads=32,
        intermediate_size=14336,
        max_position_embeddings=32768,
        use_flash_attention=True,
        use_rope=True,
        use_gqa=True,
        num_key_value_heads=4,  # GQA for memory efficiency
        dtype="bfloat16" if not args.fp16 else "float16",
        use_dualpipe=True,     # Enable dualpipe for efficient processing
        use_eplb=args.use_eplb,
        gradient_checkpointing=True
    )
    
    # Overall training configuration
    config = {
        'training': {
            'learning_rate': args.learning_rate,
            'warmup_steps': args.num_warmup_steps,
            'max_steps': args.num_train_steps,
            'max_grad_norm': 1.0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'weight_decay': args.weight_decay,
            'use_tot': args.use_tot,
            'tot_search_strategy': 'beam',
            'use_error_correction': args.use_error_correction,
            'device_batch_size': args.batch_size,
            'gradient_accumulation_steps': args.grad_accum_steps,
            'save_steps': args.save_steps,
            'eval_steps': args.eval_steps,
            'log_steps': args.log_steps,
            'seed': args.seed,
        },
        'distillation': {
            'kd_temperature': args.temperature,
            'alpha_kd': args.alpha_kd,
            'alpha_ce': args.alpha_ce,
            'error_threshold': 0.1,
            'eplb_window_size': 100,
            'eplb_threshold': 0.8
        },
        'model': {
            'teacher': asdict(teacher_config),
            'student': asdict(student_config),
            'use_bfloat16': not args.fp16,
            'use_fp16': args.fp16
        },
        'data': {
            'path': args.train_data_path,
            'val_path': args.val_data_path,
            'num_workers': 4,
            'prefetch_factor': 2
        }
    }
    
    return config, teacher_config, student_config

def create_mesh():
    """Creates a device mesh for training."""
    devices = jax.devices()
    n_devices = len(devices)
    logger.info(f"Number of devices: {n_devices}")
    
    if n_devices <= 8:
        # Single host
        mesh_shape = (n_devices,)
        mesh_axes = ('data_parallel',)
    else:
        # Multi-host: assume 8 devices per host
        n_data_parallel = n_devices // 8
        mesh_shape = (n_data_parallel, 8)
        mesh_axes = ('data_parallel', 'model_parallel')
    
    device_mesh = np.array(devices).reshape(mesh_shape)
    mesh = jax.sharding.Mesh(device_mesh, mesh_axes)
    logger.info(f"Device mesh shape: {mesh_shape}, axes: {mesh_axes}")
    
    return mesh

def load_teacher_model_from_safetensors(model_path, config):
    """
    Loads the teacher model from safetensors format.
    """
    logger.info(f"Loading teacher model from {model_path}")
    
    # Create model instance
    model = VishwamAIModel(config)
    
    # Load weights using the optimized safetensors loader
    try:
        model.load_weights(model_path)
        logger.info(f"Teacher model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading teacher model: {str(e)}")
        raise

def save_model_to_safetensors(model, config, output_dir, step=None):
    """
    Saves the model to safetensors format.
    """
    if step is not None:
        output_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model parameters
    params = model.params
    
    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Use SafeModelConverter for optimized safetensors saving
    converter = SafeModelConverter(config)
    
    # Convert params to tensors and save
    tensors = {}
    metadata = {
        "framework_version": "1.0.0",
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": "VishwamAI-7B",
        "distilled_from": "Qwen/QwQ-32B",
        "jax_version": jax.__version__,
        "tpu_compatible": True
    }
    
    # Flatten params and prepare for safetensors
    flat_params = flax.traverse_util.flatten_dict(params, sep='.')
    jax_weights = {}
    
    for key, tensor in flat_params.items():
        # Convert to JAX array
        jax_weights[key] = jnp.asarray(tensor)
    
    # Save in SafeTensors format
    save_file(jax_weights, os.path.join(output_dir, "model.safetensors"), metadata=metadata)
    logger.info(f"Model saved to {output_dir}")

def setup_wandb(args, config):
    """Set up Weights & Biases logging."""
    if args.wandb_run_name is None:
        args.wandb_run_name = f"vishwamai-distillation-{time.strftime('%Y%m%d-%H%M%S')}"
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=config
    )
    return wandb

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    
    # Create configs
    config, teacher_config, student_config = create_distillation_config(args)
    
    # Set up W&B logging
    if args.wandb_project:
        setup_wandb(args, config)
    
    # Create device mesh for training
    mesh = create_mesh()
    
    # Initialize tokenizer
    tokenizer = VishwamAITokenizer.from_pretrained(args.tokenizer_path)
    config["training"]["tokenizer"] = tokenizer
    
    # Load models - wrap with device context to ensure proper device placement
    with jax.default_device(jax.devices("tpu")[0] if jax.devices("tpu") else jax.devices("cpu")[0]):
        # Load teacher model from safetensors
        teacher_model = load_teacher_model_from_safetensors(args.teacher_model_path, teacher_config)
        
        # Initialize student model
        student_model = VishwamAIModel(student_config)
        
        # Initialize trainer
        trainer = VishwamaiShaalaTrainer(teacher_model, student_model, config)
        
        # Create training state
        rng, init_rng = jax.random.split(rng)
        train_state = trainer.create_train_state(init_rng)
    
    # Create data loaders
    train_loader = create_train_dataloader(config)
    val_loader = create_val_dataloader(config) if args.val_data_path else None
    
    # Training loop
    logger.info("Starting distillation training")
    step = 0
    best_val_loss = float('inf')
    
    progress_bar = tqdm(total=args.num_train_steps, desc="Training")
    
    try:
        while step < args.num_train_steps:
            batch = next(train_loader)
            
            # Training step
            rng, train_rng = jax.random.split(rng)
            outputs, train_state = trainer.train_step(train_state, batch, train_rng)
            
            # Log metrics
            if step % args.log_steps == 0:
                metrics = {
                    "train/loss": float(outputs["loss"]),
                    "train/kl_loss": float(outputs["metrics"].get("kd_loss", 0)),
                    "train/ce_loss": float(outputs["metrics"].get("ce_loss", 0)),
                    "train/temperature": float(outputs["metrics"].get("temperature", args.temperature)),
                    "train/learning_rate": float(train_state.opt_state.hyperparams["learning_rate"]),
                    "train/step": step
                }
                
                # Add error correction metrics if enabled
                if args.use_error_correction:
                    metrics.update({
                        "train/error_detection_rate": float(outputs["metrics"].get("error_correction_rate", 0)),
                        "train/improvement": float(outputs["metrics"].get("improvement", 0))
                    })
                
                if args.wandb_project:
                    wandb.log(metrics)
                logger.info(f"Step {step}: loss={metrics['train/loss']:.4f}")
            
            # Evaluation
            if val_loader is not None and step % args.eval_steps == 0:
                eval_metrics = {}
                for _ in range(10):  # Evaluate on 10 batches
                    val_batch = next(val_loader)
                    val_outputs = trainer.eval_step(train_state, val_batch)
                    for k, v in val_outputs["metrics"].items():
                        if k not in eval_metrics:
                            eval_metrics[k] = []
                        eval_metrics[k].append(float(v))
                
                # Average metrics
                avg_metrics = {f"val/{k}": sum(v)/len(v) for k, v in eval_metrics.items()}
                avg_metrics["val/loss"] = float(val_outputs["loss"])
                avg_metrics["val/step"] = step
                
                if args.wandb_project:
                    wandb.log(avg_metrics)
                logger.info(f"Validation step {step}: loss={avg_metrics['val/loss']:.4f}")
                
                # Save best model
                if avg_metrics["val/loss"] < best_val_loss:
                    best_val_loss = avg_metrics["val/loss"]
                    save_model_to_safetensors(student_model, student_config, os.path.join(args.output_dir, "best"))
            
            # Save checkpoint
            if step % args.save_steps == 0:
                save_model_to_safetensors(student_model, student_config, args.output_dir, step)
            
            step += 1
            progress_bar.update(1)
        
        # Save final model
        save_model_to_safetensors(student_model, student_config, args.output_dir)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted")
    finally:
        progress_bar.close()
        if args.wandb_project:
            wandb.finish()
    
    logger.info("Distillation training completed")

if __name__ == "__main__":
    main()
