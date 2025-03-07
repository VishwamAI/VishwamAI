import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
import os
from typing import Dict, Any, Optional, Tuple, Iterator
import logging
from functools import partial
from omegaconf import DictConfig

from .model import VishwamAIModel, ModelConfig
from .distillation import VishwamaiShaalaTrainer, VishwamaiGuruKnowledge
from .data_utils import create_train_dataloader, create_val_dataloader

logger = logging.getLogger(__name__)

class TrainState(train_state.TrainState):
    """Extended train state with error correction and distillation metrics."""
    error_metrics: Optional[Dict] = None
    distillation_metrics: Optional[Dict] = None

def create_train_state(
    model: VishwamAIModel,
    config: DictConfig,
    rng: jax.random.PRNGKey
) -> TrainState:
    """Create training state optimized for TPU."""
    dtype = jnp.bfloat16 if config.model.use_bfloat16 else jnp.float32
    
    # Initialize learning rate schedule
    learning_rate_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.max_steps,
        end_value=0.0
    )

    # Create optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=config.training.adam_beta1,
            b2=config.training.adam_beta2,
            weight_decay=config.training.weight_decay
        )
    )

    # Initialize model
    dummy_input = jnp.ones((1, config.model.max_position_embeddings), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)

    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
        error_metrics={},
        distillation_metrics={}
    )

@partial(jax.pmap, axis_name='batch')
def train_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    teacher_outputs: Optional[Dict[str, jnp.ndarray]] = None,
    dropout_rng: Optional[jax.random.PRNGKey] = None
) -> Tuple[TrainState, Dict[str, Any]]:
    """TPU-parallel training step with distillation."""
    dropout_rng = jax.random.fold_in(dropout_rng, state.step)

    def loss_fn(params):
        outputs = state.apply_fn(
            {'params': params},
            batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            deterministic=False,
            rngs={'dropout': dropout_rng}
        )
        
        logits = outputs['logits'][:, :-1]
        labels = batch['labels'][:, 1:]
        
        # Standard cross-entropy loss
        ce_loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        ce_loss = ce_loss.mean()
        
        # Distillation loss if teacher outputs provided
        if teacher_outputs is not None:
            teacher_logits = teacher_outputs['logits'][:, :-1]
            kl_loss = optax.softmax_cross_entropy(
                logits / 2.0,  # temperature=2.0
                jax.nn.softmax(teacher_logits / 2.0)
            )
            kl_loss = kl_loss.mean() * (2.0 ** 2)  # Scale by temperature^2
            loss = 0.7 * kl_loss + 0.3 * ce_loss  # alpha=0.7
        else:
            loss = ce_loss
            kl_loss = jnp.array(0.0)

        metrics = {
            'loss': loss,
            'ce_loss': ce_loss,
            'kl_loss': kl_loss if teacher_outputs is not None else 0.0,
        }
        return loss, metrics

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params)
    
    # All-reduce gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    state = state.apply_gradients(grads=grads)
    return state, metrics

@partial(jax.pmap, axis_name='batch')
def eval_step(
    state: TrainState,
    batch: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """TPU-parallel evaluation step."""
    outputs = state.apply_fn(
        {'params': state.params},
        batch['input_ids'],
        attention_mask=batch.get('attention_mask'),
        deterministic=True
    )
    
    logits = outputs['logits'][:, :-1]
    labels = batch['labels'][:, 1:]
    
    # Loss and metrics
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    loss = loss.mean()
    
    # Calculate accuracy
    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == labels).astype(jnp.float32)
    accuracy = jnp.mean(correct)
    
    # Calculate perplexity
    perplexity = jnp.exp(jnp.minimum(loss, 100.0))
    
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }
    
    return jax.lax.pmean(metrics, axis_name='batch')

def evaluate(
    state: TrainState, 
    val_loader: Iterator[Dict[str, jnp.ndarray]], 
    config: DictConfig,
    num_batches: int = 10
) -> Dict[str, float]:
    """Evaluate the model on validation data with TPU parallelization."""
    metrics_list = []
    
    for _ in range(num_batches):
        try:
            batch = next(val_loader)
            batch_metrics = eval_step(state, batch)
            
            # Get metrics from first device
            device_metrics = jax.tree_map(lambda x: x[0], batch_metrics)
            metrics_list.append(device_metrics)
        except StopIteration:
            break
    
    # Average metrics across batches
    avg_metrics = {}
    if metrics_list:
        for k in metrics_list[0].keys():
            avg_metrics[k] = float(sum(m[k] for m in metrics_list)) / len(metrics_list)
    
    return avg_metrics

def save_checkpoint(
    state: TrainState,
    config: DictConfig,
    step: int
) -> None:
    """Save training checkpoint with TPU compatibility."""
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(
        config.training.output_dir, 
        config.training.checkpoint_dir
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save only unreplicated params to avoid duplication
    unreplicated_state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    
    checkpoints.save_checkpoint(
        ckpt_dir=checkpoint_dir,
        target=unreplicated_state,
        step=step,
        overwrite=True,
        keep=config.training.keep_checkpoints
    )
    
    logger.info(f"Checkpoint saved at step {step}")

def train(
    config: DictConfig,
    model: VishwamAIModel,
    teacher_model: Optional[VishwamAIModel] = None,
    rng: Optional[jax.random.PRNGKey] = None
) -> TrainState:
    """Main training loop with TPU and distillation support."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    # Create training state
    state = create_train_state(model, config, rng)
    
    # Initialize data loaders
    train_loader = create_train_dataloader(config)
    val_loader = create_val_dataloader(config)
    
    # Initialize distillation if teacher provided
    if teacher_model is not None:
        distiller = VishwamaiShaalaTrainer(
            teacher_model=teacher_model,
            student_model=model,
            cfg=config
        )
    
    # Training loop
    for step in range(config.training.max_steps):
        rng, dropout_rng = jax.random.split(rng)
        
        batch = next(train_loader)
        if teacher_model is not None:
            with jax.disable_jit():  # Prevent teacher computation from being jitted
                teacher_outputs = teacher_model(
                    batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    deterministic=True
                )
        else:
            teacher_outputs = None
            
        state, metrics = train_step(
            state,
            batch,
            teacher_outputs,
            dropout_rng
        )
        
        # Logging and validation
        if (step + 1) % config.training.log_every == 0:
            logger.info(f"Step {step+1}: {metrics}")
            
        if (step + 1) % config.training.eval_every == 0:
            val_metrics = evaluate(state, val_loader, config)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            if (step + 1) % config.training.save_every == 0:
                save_checkpoint(state, config, step + 1)
    
    return state

# Import this module in __init__.py
__all__ = ['train', 'create_train_state', 'TrainState', 'evaluate', 'save_checkpoint']
