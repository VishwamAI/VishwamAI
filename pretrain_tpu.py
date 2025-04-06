"""Pre-training script optimized for TPU execution with Gemma weights loading."""

import jax
import jax.numpy as jnp
import optax
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from functools import partial

from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.transformer import TPUTrainingState 
from vishwamai.pipeline import TPUDataPipeline, DistillationDataPipeline
from vishwamai.device_mesh import TPUMeshContext
from vishwamai.distill import DistillationTrainer, create_student_model
from vishwamai.thoughts import TreeOfThoughts, ChainOfThoughtPrompting
from vishwamai.configs.tpu_v3_config import TPUV3Config
from vishwamai.profiler import TPUProfiler
from vishwamai.logger import setup_logging
from vishwamai import init_thinking_components
import time
from tqdm.auto import tqdm

# Setup logging
logger = setup_logging(__name__)

# Constants
CHECKPOINT_DIR = Path("checkpoints")
LOGS_DIR = Path("logs")
GEMMA_MODEL_ID = "google/gemma-3-27b-pt"
CACHE_DIR = Path("model_cache")

def get_pretrain_config() -> Dict[str, Any]:
    """Get pretraining configuration."""
    # Start with TPU v3 optimized base config
    tpu_config = TPUV3Config()
    
    return {
        "model": {
            **tpu_config.model_config,
            "use_flash_attn": True,
            "vocab_size": 256000,  # Gemma tokenizer vocab size
            "hidden_dim": 8192,    # Scaled for 27B model
            "num_layers": 80,      # Gemma architecture
            "num_heads": 64,
            "head_dim": 128,
            "mlp_dim": 28672,
            "max_seq_len": 8192,
            "dropout_rate": 0.0,
            "attention_dropout": 0.0
        },
        "training": {
            **tpu_config.training_config,
            "batch_size": 32,      # Per TPU core
            "grad_accum_steps": 8,
            "learning_rate": 1e-4,
            "warmup_steps": 2000,
            "max_steps": 100000,
            "weight_decay": 0.1,
            "max_grad_norm": 1.0,
            "checkpoint_steps": 1000
        },
        "optimization": {
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "mixed_precision": True
        },
        "thinking": {
            "max_branches": 4,
            "max_depth": 3,
            "beam_width": 4,
            "temperature": 0.7,
            "max_length": 512,
            "num_samples": 3
        },
        "distillation": {
            "teacher_model": GEMMA_MODEL_ID,
            "temperature": 2.0,
            "alpha": 0.5,
            "layer_mapping_strategy": "uniform"
        }
    }

def main():
    # Load configuration
    config = get_pretrain_config()
    
    # Create cache directory
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize teacher (Gemma) model
    logger.info("Loading Gemma teacher model...")
    teacher_model = VishwamAI(config=VishwamAIConfig(**config["model"]))
    
    # Create and initialize student model
    logger.info("Creating student model...")
    student_model, student_vars, student_config = create_student_model(
        config["model"],
        teacher_model,
        reduction_factor=0.3,  # Start with 30% of teacher size
        rng=jax.random.PRNGKey(42)
    )

    # Initialize TPU mesh
    devices = jax.devices()
    logger.info(f"Available TPU devices: {devices}")
    mesh_context = TPUMeshContext(config, data_parallel=True)

    # Initialize thinking components for both teacher and student
    logger.info("Initializing thinking components...")
    teacher_tot, teacher_cot = init_thinking_components(
        model=teacher_model,
        params=teacher_model.params,
        tokenizer=teacher_model.tokenizer,
        thinking_config=config["thinking"]
    )
    
    student_tot, student_cot = init_thinking_components(
        model=student_model,
        params=student_vars,
        tokenizer=student_model.tokenizer,
        thinking_config=config["thinking"]
    )

    # Create distillation trainer with thinking components
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_config=student_config,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        use_flash_attn=config["model"]["use_flash_attn"],
        profiler=TPUProfiler(config),
        teacher_tot=teacher_tot,
        teacher_cot=teacher_cot,
        student_tot=student_tot,
        student_cot=student_cot
    )

    # Create data pipeline
    data_pipeline = DistillationDataPipeline(
        config=config,
        teacher_model=teacher_model,
        devices=devices,
        enable_thinking=True
    )

    train_loader = data_pipeline.create_distillation_dataset(
        "train-*.parquet",
        is_training=True,
        cache_teacher_outputs=True
    )

    # Training loop with TPU optimizations
    with mesh_context:
        step = 0
        start_time = time.time()
        
        with tqdm(total=config["training"]["max_steps"]) as pbar:
            for batch in train_loader:
                if step >= config["training"]["max_steps"]:
                    break
                
                # Training step with distillation and thinking
                state, metrics = trainer.train_step(
                    batch,
                    enable_thinking=True,
                    thinking_weight=0.3  # Weight for thinking loss
                )
                
                # Update progress
                step += 1
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'distill_loss': f"{metrics.get('distill_loss', 0.0):.4f}",
                    'thinking_loss': f"{metrics.get('thinking_loss', 0.0):.4f}",
                    'lr': f"{metrics.get('learning_rate', 0.0):.6f}"
                })
                
                # Save checkpoints periodically
                if step % config["training"]["checkpoint_steps"] == 0:
                    checkpoint_path = CHECKPOINT_DIR / f"checkpoint-{step}"
                    save_checkpoint(
                        checkpoint_path, 
                        state, 
                        teacher_tot, 
                        teacher_cot,
                        student_tot,
                        student_cot,
                        metrics
                    )
                    
                # Log metrics
                if step % 100 == 0:
                    current_time = time.time()
                    steps_per_second = 100 / (current_time - start_time)
                    logger.info(
                        f"Step {step}: loss = {metrics['loss']:.4f}, "
                        f"thinking_quality = {metrics.get('thinking_quality', 0.0):.4f}, "
                        f"steps/second = {steps_per_second:.2f}"
                    )
                    start_time = current_time

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")

def save_checkpoint(
    path: Path,
    state: TPUTrainingState,
    teacher_tot: TreeOfThoughts,
    teacher_cot: ChainOfThoughtPrompting,
    student_tot: TreeOfThoughts,
    student_cot: ChainOfThoughtPrompting,
    metrics: Dict[str, Any]
):
    """Save training checkpoint with model and thinking component states."""
    path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dict = {
        'step': state.step,
        'state': state.params,
        'optimizer_state': state.opt_state,
        'teacher_tot_state': teacher_tot.get_state(),
        'teacher_cot_state': teacher_cot.get_state(),
        'student_tot_state': student_tot.get_state(),
        'student_cot_state': student_cot.get_state(),
        'metrics': metrics
    }
    
    with (path / "checkpoint.safetensors").open('wb') as f:
        jax.tree_util.tree_map(
            lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
            checkpoint_dict
        ).save(f)
    
    logger.info(f"Saved checkpoint to {path}")

if __name__ == "__main__":
    main()