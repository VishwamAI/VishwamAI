"""Pre-training script optimized for TPU execution with Gemma weights loading."""

import jax
import jax.numpy as jnp
import optax
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, Union
from functools import partial

from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.transformer import EnhancedTransformerModel, TPUTrainingState 
from vishwamai.pipeline import TPUDataPipeline, DistillationDataPipeline
from vishwamai.device_mesh import TPUMeshContext
from vishwamai.distill import create_student_model
from vishwamai.thoughts import TreeOfThoughts
from vishwamai.configs.tpu_v3_config import TPUV3Config
from vishwamai.utils.model_loading import load_pretrained_weights
from vishwamai.profiler import TPUProfiler
from vishwamai.logger import setup_logging
import time
from tqdm.auto import tqdm

# Setup logging
logger = setup_logging(__name__)

# Constants
CHECKPOINT_DIR = Path("checkpoints")
LOGS_DIR = Path("logs")
GEMMA_MODEL_ID = "google/gemma-3-27b-pt"
CACHE_DIR = Path("model_cache")

def train_step(state: TPUTrainingState, batch: Dict[str, Any]) -> Tuple[TPUTrainingState, Dict[str, Any]]:
    """Perform a single training step."""
    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            deterministic=False
        )['logits']
        
        # Calculate loss with label smoothing
        labels = batch['labels']
        padding_mask = (labels != 0)
        label_smoothing = 0.1
        
        # Convert labels to one-hot with label smoothing
        num_classes = logits.shape[-1]
        labels_onehot = jax.nn.one_hot(labels, num_classes)
        labels_smooth = ((1.0 - label_smoothing) * labels_onehot +
                        label_smoothing / num_classes)
        
        # Compute cross entropy loss
        logits_shifted = logits[..., :-1, :]
        labels_shifted = labels_smooth[..., 1:, :]
        padding_mask_shifted = padding_mask[..., 1:]
        
        loss = -jnp.sum(
            labels_shifted * jax.nn.log_softmax(logits_shifted, axis=-1),
            axis=-1
        )
        loss = jnp.sum(loss * padding_mask_shifted) / jnp.sum(padding_mask_shifted)
        
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Update state
    state = state.apply_gradients(grads=grads)
    
    # Compute metrics
    metrics = {
        'loss': loss,
        'learning_rate': state.opt_state[1].learning_rate,
        'gradient_norm': optax.global_norm(grads)
    }
    
    return state, metrics

def get_pretrain_config() -> Dict[str, Any]:
    """Get pretraining configuration."""
    # Start with TPU v3 optimized base config
    config = {
        "model": {
            "vocab_size": 256000,  # Gemma vocabulary size
            "hidden_dim": 5120,    # Matches Gemma-3b architecture
            "num_layers": 28,
            "num_heads": 32,
            "head_dim": 160,
            "mlp_dim": 20480,
            "max_seq_len": 8192,
            "dropout_rate": 0.0,
            "attention_dropout": 0.0,
            "use_flash_attn": True
        },
        "training": {
            "batch_size": 32,      # Reduced for GPU memory
            "grad_accum_steps": 4,
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
            "num_steps": 3,
            "max_branches": 4,
            "max_depth": 3,
            "beam_width": 4,
            "temperature": 0.7
        },
        "hardware": {
            "device": "gpu",  # Will be auto-detected
            "mesh_shape": [1, 1],  # Single device for now
            "mesh_order": ['data', 'model']
        }
    }
    
    # Auto-detect hardware and adjust configuration
    try:
        import jax
        if len(jax.devices()) > 1:
            config["hardware"]["mesh_shape"] = [len(jax.devices()), 1]
        if any(d.platform == "tpu" for d in jax.devices()):
            config["hardware"]["device"] = "tpu"
            config["training"]["batch_size"] = 512  # Increase for TPU
    except:
        pass
        
    return config

def create_vishwamai_transformer(
    config: VishwamAIConfig,
    pretrained_weights_path: Optional[str] = None,
    dtype: str = "bfloat16"
) -> Tuple[VishwamAI, Dict[str, Any]]:
    """Create a VishwamAI transformer model and initialize with Gemma weights."""
    model = VishwamAI(config=config)
    
    if pretrained_weights_path:
        # Load pretrained Gemma weights
        pretrained_weights = load_pretrained_weights(
            model_id=GEMMA_MODEL_ID,
            cache_dir=str(CACHE_DIR),
            dtype=getattr(jnp, dtype)
        )
        
        # Initialize with dummy input to get param structure
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
        variables = model.init(rng, dummy_input)
        
        # Map pretrained weights to model structure
        mapped_weights = {}
        for name, param in variables['params'].items():
            if name in pretrained_weights:
                mapped_weights[name] = pretrained_weights[name]
            else:
                logger.warning(f"Parameter {name} not found in pretrained weights")
                mapped_weights[name] = param
        
        variables['params'] = mapped_weights
    else:
        # Initialize from scratch
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
        variables = model.init(rng, dummy_input)
    
    return model, variables

def main():
    # Load configuration
    config = get_pretrain_config()
    
    # Create cache directory if it doesn't exist
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize model with pretrained Gemma weights
    model, variables = create_vishwamai_transformer(
        config=VishwamAIConfig(**config["model"]),
        pretrained_weights_path=GEMMA_MODEL_ID,
        dtype=config["optimization"]["dtype"]
    )

    # Initialize TPU mesh
    devices = jax.devices()
    logger.info(f"Available TPU devices: {devices}")
    mesh_context = TPUMeshContext(config, data_parallel=True)

    # Create Tree of Thoughts for reasoning capabilities
    tot = TreeOfThoughts(
        model=model,
        params=variables["params"],
        tokenizer=model.tokenizer,
        max_branches=config["thinking"]["max_branches"],
        max_depth=config["thinking"]["max_depth"],
        beam_width=config["thinking"]["beam_width"],
        temperature=config["thinking"]["temperature"]
    )

    # Setup training state and optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config["training"]["max_grad_norm"]),
        optax.adamw(
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"]
        )
    )

    state = TPUTrainingState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
        dynamic_scale=None  # We're using bfloat16 which doesn't need dynamic scaling
    )

    # Create data pipeline
    data_pipeline = TPUDataPipeline(
        config=config,
        devices=devices,
        enable_thinking=True
    )

    train_loader = data_pipeline.create_training_dataset(
        "train-*.parquet",
        is_training=True
    )

    # Training loop with TPU optimizations
    with mesh_context:
        step = 0
        start_time = time.time()
        
        with tqdm(total=config["training"]["max_steps"]) as pbar:
            for batch in train_loader:
                if step >= config["training"]["max_steps"]:
                    break
                
                # Training step
                state, metrics = train_step(state, batch)
                
                # Update progress
                step += 1
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics.get('learning_rate', 0.0):.6f}",
                })
                
                # Save checkpoints periodically
                if step % config["training"]["checkpoint_steps"] == 0:
                    checkpoint_path = CHECKPOINT_DIR / f"checkpoint-{step}"
                    save_checkpoint(checkpoint_path, state, tot, metrics)
                    
                # Log metrics
                if step % 100 == 0:
                    current_time = time.time()
                    steps_per_second = 100 / (current_time - start_time)
                    logger.info(
                        f"Step {step}: loss = {metrics['loss']:.4f}, "
                        f"steps/second = {steps_per_second:.2f}"
                    )
                    start_time = current_time

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final metrics: {metrics}")

        # Save final checkpoint
        final_checkpoint_path = CHECKPOINT_DIR / "checkpoint-final"
        save_checkpoint(final_checkpoint_path, state, tot, metrics)

def save_checkpoint(path: Path, state: TPUTrainingState, tot: TreeOfThoughts, metrics: Dict[str, Any]):
    """Save training checkpoint with model state and metrics."""
    path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dict = {
        'step': state.step,
        'state': state.params,
        'optimizer_state': state.opt_state,
        'tot_state': tot.get_state(),
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