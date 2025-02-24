#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints
from flax.jax_utils import replicate, unreplicate
import optax
from tqdm import tqdm

from .model import VishwamAIModel, ModelConfig, create_optimizer
from .tokenizer import VishwamAITokenizer
from .error_correction import ErrorCorrectionModule, ModelIntegrator
from .tot import TreeOfThoughts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_train_state(
    model: VishwamAIModel,
    config: ModelConfig,
    learning_rate: float,
    rng: jax.random.PRNGKey
) -> train_state.TrainState:
    """Initialize training state."""
    params_rng, dropout_rng = jax.random.split(rng)
    
    # Initialize model parameters with a dummy input
    sample_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    initial_params = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        input_ids=sample_input,
        train=True
    )['params']

    # Create optimizer
    tx = create_optimizer(learning_rate)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=initial_params,
        tx=tx
    )

def compute_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray
) -> jnp.ndarray:
    """Compute masked cross-entropy loss."""
    vocab_size = logits.shape[-1]
    label_smoothing = 0.1  # Small amount of label smoothing for regularization
    
    # Create smoothed labels
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = label_smoothing / (vocab_size - 1)
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)
    smooth_labels = one_hot_labels * smooth_positives + \
                   (1 - one_hot_labels) * smooth_negatives

    # Compute cross entropy loss
    loss = -jnp.sum(smooth_labels * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply mask and compute mean
    masked_loss = loss * mask
    return jnp.sum(masked_loss) / jnp.sum(mask)

def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
    intermediate_outputs: List = None,
    error_gates: jnp.ndarray = None,
    load_balance_loss: jnp.ndarray = None
) -> Dict[str, jnp.ndarray]:
    """Compute comprehensive training metrics including MoE and error correction."""
    # Base metrics
    loss = compute_loss(logits, labels, mask)
    predictions = jnp.argmax(logits, axis=-1)
    correct_predictions = (predictions == labels) * mask
    accuracy = jnp.sum(correct_predictions) / jnp.sum(mask)
    perplexity = jnp.exp(loss)
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }
    
    # Error correction metrics
    if error_gates is not None:
        error_rate = jnp.mean(error_gates)
        metrics.update({
            'error_rate': error_rate,
            'error_gate_std': jnp.std(error_gates)
        })
    
    # MoE metrics
    if load_balance_loss is not None:
        metrics['moe_loss'] = load_balance_loss
    
    # Layer-wise metrics if intermediate outputs available
    if intermediate_outputs:
        for i, layer_output in enumerate(intermediate_outputs):
            layer_loss = compute_loss(layer_output, labels, mask)
            metrics[f'layer_{i}_loss'] = layer_loss
    
    return metrics

def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    error_correction: ModelIntegrator,
) -> Tuple[train_state.TrainState, Dict[str, jnp.ndarray], jax.random.PRNGKey]:
    """Perform a single training step with MoE and error correction."""
    rng, new_rng = jax.random.split(rng)
    
    def loss_fn(params):
        outputs = state.apply_fn(
            {'params': params},
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            train=True,
            rngs={'dropout': rng}
        )
        
        logits = outputs['logits']
        intermediate_outputs = outputs.get('intermediate_outputs', [])
        error_gates = outputs.get('error_gates', None)
        load_balance_loss = outputs.get('load_balance_loss', None)
        
        # Base loss computation
        loss = compute_loss(logits, batch['labels'], batch['attention_mask'])
        
        # Add MoE load balancing loss if available
        if load_balance_loss is not None:
            loss = loss + 0.01 * load_balance_loss  # Scale factor for load balancing
            
        # Apply error correction if enabled
        if error_correction is not None:
            logits = error_correction.process_logits(logits)
            
        return loss, (logits, intermediate_outputs, error_gates, load_balance_loss)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    
    # Update state
    state = state.apply_gradients(grads=grads)
    
    # Compute metrics
    metrics = compute_metrics(logits, batch['labels'], batch['attention_mask'])
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    return state, metrics, new_rng

# Parallel training step
p_train_step = jax.pmap(
    train_step,
    axis_name='batch',
    donate_argnums=(0,),  # Donate state buffers
)

def create_train_dataset(
    data_path: str,
    tokenizer: Any,
    max_seq_len: int,
    batch_size: int
):
    """Create training dataset from text files."""
    from datasets import load_dataset
    
    # Load raw text files
    dataset = load_dataset('text', data_files=data_path)['train']
    
    # Tokenization function
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=max_seq_len,
            return_tensors='np'
        )
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids']  # For language modeling
        }
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Create data loader
    dataloader = tokenized_dataset.with_format('numpy').iter(
        batch_size=batch_size,
        drop_last=True
    )
    
    return dataloader

def train_epoch(
    state: train_state.TrainState,
    train_loader: Any,
    rng: jax.random.PRNGKey,
    error_correction: ModelIntegrator,
    epoch: int,
    tot_controller=None
) -> Tuple[train_state.TrainState, Dict[str, float]]:
    """Train for one epoch with MoE and ToT support."""
    batch_metrics = []
    
    # Training loop with progress bar
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch in pbar:
            # Shard batch across devices
            batch = jax.device_put_sharded(
                jax.tree_util.tree_map(
                    lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]),
                    batch
                ),
                jax.devices()
            )
            
            # Perform training step
            state, metrics, rng = p_train_step(state, batch, rng, error_correction)
            batch_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': float(metrics['loss'].mean()),
                'acc': float(metrics['accuracy'].mean()),
                'ppl': float(metrics['perplexity'].mean())
            })
    
    # Compute epoch metrics
    epoch_metrics = {
        k: float(jnp.mean(jnp.stack([m[k].mean() for m in batch_metrics])))
        for k in batch_metrics[0].keys()
    }
    
    return state, epoch_metrics

def save_checkpoint(
    state: train_state.TrainState,
    save_dir: str,
    step: int
):
    """Save a checkpoint."""
    state = unreplicate(state)
    checkpoints.save_checkpoint(
        ckpt_dir=save_dir,
        target=state,
        step=step,
        overwrite=True
    )

def main():
    parser = argparse.ArgumentParser(description='Train VishwamAI Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to model configuration file')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size per device')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Save checkpoint every N steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--use_error_correction', action='store_true',
                       help='Enable error correction module')
    parser.add_argument('--use_tot', action='store_true',
                       help='Enable Tree of Thoughts')
    parser.add_argument('--moe_loss_weight', type=float, default=0.01,
                       help='Weight for MoE load balancing loss')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Enable gradient checkpointing for memory efficiency')
    
    args = parser.parse_args()
    
    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    
    # Load configuration
    with open(args.config) as f:
        config = ModelConfig(**json.load(f))
    
    # Initialize model and tokenizer
    model = VishwamAIModel(config)
    tokenizer = VishwamAITokenizer.from_pretrained(args.tokenizer)
    
    # Initialize error correction if enabled
    error_correction = None
    if args.use_error_correction:
        error_module = ErrorCorrectionModule(config)
        error_correction = ModelIntegrator(error_module)
    
    # Initialize ToT if enabled
    tot = None
    if args.use_tot:
        tot = TreeOfThoughts(config)
    
    # Create training state
    state = create_train_state(model, config, args.learning_rate, rng)
    
    # Replicate state across devices
    state = replicate(state)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create training dataset
    train_loader = create_train_dataset(
        args.train_data,
        tokenizer,
        config.max_seq_len,
        args.batch_size
    )

    # Training loop with improved logging
    logger.info("Starting training with configuration:")
    logger.info(f"Model config: {config}")
    logger.info(f"Using error correction: {args.use_error_correction}")
    logger.info(f"Using ToT: {args.use_tot}")
    logger.info(f"MoE loss weight: {args.moe_loss_weight}")
    logger.info(f"Gradient checkpointing: {args.gradient_checkpointing}")
    
    for epoch in range(args.num_epochs):
        rng, epoch_rng = jax.random.split(rng)
        
        # Train for one epoch with all components
        state, metrics = train_epoch(
            state=state,
            train_loader=train_loader,
            rng=epoch_rng,
            error_correction=error_correction,
            epoch=epoch + 1,
            tot_controller=tot if args.use_tot else None
        )
        
        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Loss: {metrics['loss']:.4f}, "
            f"Accuracy: {metrics['accuracy']:.4f}, "
            f"Perplexity: {metrics['perplexity']:.4f}"
        )
        
        # Save checkpoint
        if (epoch + 1) % args.save_steps == 0:
            save_checkpoint(state, args.output_dir, epoch + 1)
    
    # Save final checkpoint
    save_checkpoint(state, args.output_dir, args.num_epochs)
    logger.info("Training complete!")

if __name__ == '__main__':
    main()
