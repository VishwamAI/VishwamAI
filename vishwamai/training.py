#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, List, NamedTuple, Optional
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state, checkpoints, dynamic_scale
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

@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    learning_rate: float
    batch_size: int
    grad_accum_steps: int
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    use_amp: bool = True
    dynamic_batch_size: bool = True
    min_batch_size: int = 4
    target_batch_size: int = 32

class MetricsState(NamedTuple):
    """State for tracking training metrics."""
    loss_scale: float
    grad_norm: float
    param_norm: float
    learning_rate: float

class TrainState(train_state.TrainState):
    """Custom train state with dynamic scaling for mixed precision."""
    dynamic_scale: Optional[dynamic_scale.DynamicScale]
    metrics: Optional[MetricsState] = None
    grad_accum_count: int = 0

def create_train_state(
    model: VishwamAIModel,
    config: ModelConfig,
    training_config: TrainingConfig,
    rng: jax.random.PRNGKey
) -> TrainState:
    """Initialize training state with dynamic scaling."""
    params_rng, dropout_rng = jax.random.split(rng)
    
    # Initialize model parameters
    sample_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    initial_params = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        input_ids=sample_input,
        train=True
    )['params']
    
    # Create learning rate schedule with warmup
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=training_config.learning_rate,
        warmup_steps=training_config.warmup_steps,
        decay_steps=10000,
        end_value=training_config.learning_rate * 0.1
    )
    
    # Create optimizer with gradient clipping and weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=training_config.weight_decay
        )
    )
    
    # Create dynamic scale for mixed precision if enabled
    dynamic_scale = None
    if training_config.use_amp:
        dynamic_scale = dynamic_scale.DynamicScale()
    
    return TrainState.create(
        apply_fn=model.apply,
        params=initial_params,
        tx=optimizer,
        dynamic_scale=dynamic_scale,
        metrics=MetricsState(0.0, 0.0, 0.0, training_config.learning_rate)
    )

def compute_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray
) -> jnp.ndarray:
    """Compute masked cross-entropy loss with label smoothing."""
    vocab_size = logits.shape[-1]
    label_smoothing = 0.1
    
    smooth_positives = 1.0 - label_smoothing
    smooth_negatives = label_smoothing / (vocab_size - 1)
    one_hot_labels = jax.nn.one_hot(labels, vocab_size)
    smooth_labels = one_hot_labels * smooth_positives + \
                   (1 - one_hot_labels) * smooth_negatives
    
    loss = -jnp.sum(smooth_labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    loss = loss * mask
    return jnp.sum(loss) / jnp.sum(mask)

def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: jnp.ndarray,
    intermediate_outputs: Optional[List] = None,
    error_gates: Optional[jnp.ndarray] = None,
    load_balance_loss: Optional[jnp.ndarray] = None
) -> Dict[str, jnp.ndarray]:
    """Compute comprehensive training metrics."""
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
    
    if error_gates is not None:
        error_rate = jnp.mean(error_gates)
        metrics.update({
            'error_rate': error_rate,
            'error_gate_std': jnp.std(error_gates)
        })
    
    if load_balance_loss is not None:
        metrics['moe_loss'] = load_balance_loss
    
    if intermediate_outputs:
        for i, layer_output in enumerate(intermediate_outputs):
            layer_loss = compute_loss(layer_output, labels, mask)
            metrics[f'layer_{i}_loss'] = layer_loss
    
    return metrics

def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    error_correction: Optional[ModelIntegrator],
    accum_steps: int,
) -> Tuple[TrainState, Dict[str, jnp.ndarray], jax.random.PRNGKey]:
    """Perform a training step with gradient accumulation and mixed precision."""
    
    # Helper function for forward pass with mixed precision
    def forward_fn(params):
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
        
        loss = compute_loss(logits, batch['labels'], batch['attention_mask'])
        if load_balance_loss is not None:
            loss = loss + 0.01 * load_balance_loss
            
        if error_correction is not None:
            logits = error_correction.process_logits(logits)
            
        return loss, (logits, intermediate_outputs, error_gates, load_balance_loss)
    
    # Handle mixed precision training
    if state.dynamic_scale:
        # Run forward pass with dynamic scaling
        dynamic_scale, is_finite, aux = state.dynamic_scale.value_and_grad(
            forward_fn, has_aux=True)(state.params)
        
        # Extract gradients and values
        grad, (loss, (logits, intermediate_outputs, error_gates, load_balance_loss)) = aux
        
        # Update dynamic scale
        dynamic_scale = dynamic_scale.update(is_finite)
        
        # Skip step if gradients contain Inf/NaN
        grad = jax.tree_map(
            lambda x: jnp.where(is_finite, x, jnp.zeros_like(x)),
            grad
        )
    else:
        # Regular forward pass without mixed precision
        (loss, (logits, intermediate_outputs, error_gates, load_balance_loss)), grad = jax.value_and_grad(
            forward_fn, has_aux=True)(state.params)
        is_finite = True
    
    # Accumulate gradients
    grad = jax.tree_map(lambda x: x / accum_steps, grad)
    
    # Average gradients across devices
    grad = jax.lax.pmean(grad, axis_name='batch')
    
    # Update state if we have accumulated enough gradients
    if state.grad_accum_count == accum_steps - 1:
        # Compute gradient and parameter norms
        grad_norm = optax.global_norm(grad)
        param_norm = optax.global_norm(state.params)
        
        state = state.apply_gradients(
            grads=grad,
            dynamic_scale=state.dynamic_scale if state.dynamic_scale else None,
            metrics=MetricsState(
                loss_scale=state.dynamic_scale.scale if state.dynamic_scale else 1.0,
                grad_norm=grad_norm,
                param_norm=param_norm,
                learning_rate=state.tx.learning_rate(state.step)
            )
        )
        grad_accum_count = 0
    else:
        state = state.replace(grad_accum_count=state.grad_accum_count + 1)
    
    # Compute metrics
    metrics = compute_metrics(logits, batch['labels'], batch['attention_mask'])
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    rng, new_rng = jax.random.split(rng)
    return state, metrics, new_rng

# Parallel training step
p_train_step = jax.pmap(
    train_step,
    axis_name='batch',
    donate_argnums=(0,),
)

def adjust_batch_size(metrics: Dict[str, float], config: TrainingConfig) -> int:
    """Dynamically adjust batch size based on training metrics."""
    if not config.dynamic_batch_size:
        return config.batch_size
    
    # Check if we have out of memory or gradient overflow issues
    has_issues = (metrics.get('grad_norm', 0) > config.max_grad_norm * 2 or
                 not metrics.get('is_finite', True))
    
    if has_issues:
        # Reduce batch size
        new_batch_size = max(config.batch_size // 2, config.min_batch_size)
        logger.info(f"Reducing batch size to {new_batch_size}")
        return new_batch_size
    elif config.batch_size < config.target_batch_size:
        # Try to increase batch size
        new_batch_size = min(config.batch_size * 2, config.target_batch_size)
        logger.info(f"Increasing batch size to {new_batch_size}")
        return new_batch_size
    
    return config.batch_size

def create_train_dataset(
    data_path: str,
    tokenizer: VishwamAITokenizer,
    config: TrainingConfig
):
    """Create training dataset with dynamic batching."""
    from datasets import load_dataset
    
    dataset = load_dataset('text', data_files=data_path)['train']
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=config.max_seq_len,
            return_tensors='np'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    dataloader = tokenized_dataset.with_format('numpy').iter(
        batch_size=config.batch_size,
        drop_last=True
    )
    
    return dataloader

def train_epoch(
    state: TrainState,
    train_loader: Any,
    rng: jax.random.PRNGKey,
    config: TrainingConfig,
    error_correction: Optional[ModelIntegrator],
    epoch: int,
    tot_controller=None
) -> Tuple[TrainState, Dict[str, float]]:
    """Train for one epoch with gradient accumulation and dynamic batch sizing."""
    batch_metrics = []
    
    with tqdm(train_loader, desc=f'Epoch {epoch}') as pbar:
        for batch in pbar:
            # Adjust batch size if needed
            if len(batch_metrics) > 0:
                config.batch_size = adjust_batch_size(batch_metrics[-1], config)
            
            # Shard batch across devices
            batch = jax.device_put_sharded(
                jax.tree_util.tree_map(
                    lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]),
                    batch
                ),
                jax.devices()
            )
            
            # Perform training step with gradient accumulation
            state, metrics, rng = p_train_step(
                state, batch, rng, error_correction, config.grad_accum_steps
            )
            batch_metrics.append(metrics)
            
            # Update progress bar with current metrics and training state
            pbar.set_postfix({
                'loss': float(metrics['loss'].mean()),
                'acc': float(metrics['accuracy'].mean()),
                'ppl': float(metrics['perplexity'].mean()),
                'grad_norm': float(state.metrics.grad_norm if state.metrics else 0),
                'lr': float(state.metrics.learning_rate if state.metrics else 0),
                'batch_size': config.batch_size
            })
    
    # Compute epoch metrics
    epoch_metrics = {
        k: float(jnp.mean(jnp.stack([m[k].mean() for m in batch_metrics])))
        for k in batch_metrics[0].keys()
    }
    
    return state, epoch_metrics

def save_checkpoint(
    state: TrainState,
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
    """CLI for training VishwamAI Model."""
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
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--use_amp', action='store_true',
                       help='Enable automatic mixed precision training')
    parser.add_argument('--dynamic_batch_size', action='store_true',
                       help='Enable dynamic batch sizing')
    
    args = parser.parse_args()
    
    # Set random seed
    rng = jax.random.PRNGKey(args.seed)
    
    # Load configuration
    with open(args.config) as f:
        config = ModelConfig(**json.load(f))
    
    # Create training config
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        use_amp=args.use_amp,
        dynamic_batch_size=args.dynamic_batch_size
    )
    
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
    state = create_train_state(model, config, training_config, rng)
    
    # Replicate state across devices
    state = replicate(state)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create training dataset
    train_loader = create_train_dataset(
        args.train_data,
        tokenizer,
        training_config
    )

    # Training loop with improved logging
    logger.info("Starting training with configuration:")
    logger.info(f"Model config: {config}")
    logger.info(f"Training config: {training_config}")
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
            config=training_config,
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
