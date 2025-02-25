import jax
import jax.numpy as jnp
import flax
import optax
from flax.training import train_state
from typing import Dict, List, Tuple, Callable, Any, Iterator, Optional
import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf, DictConfig
import random
import logging
from functools import partial
import os
from .tot import TreeOfThoughts
from .integration import ToTIntegrationLayer, MixtureDensityNetwork, MultiLevelToTAttention
from .transformer import VishwamAIModel
# Import from loss_functions module
from .loss_functions import cross_entropy_loss, tot_guided_loss, compute_metrics

# Import from error_correction_trainer to avoid circular imports
from .error_correction_trainer import (
    create_error_corrected_train_step,
    create_error_corrected_eval_step,
    evaluate_error_correction,
    ErrorCorrectionTrainer
)

logger = logging.getLogger(__name__)

class TrainingState(train_state.TrainState):
    """Extended train state with EMA and other training metrics."""
    ema_params: Dict = None
    step: int = 0
    best_metrics: Dict = None
    tot_state: Dict = None  # New field for Tree of Thoughts state

def create_learning_rate_scheduler(
    factors="constant * linear_warmup * cosine_decay",
    base_learning_rate=0.0001,
    warmup_steps=1000,
    decay_steps=100000,
):
    """Creates a learning rate schedule.
    
    Args:
        factors: Factors separated by '*' for the learning rate formula.
        base_learning_rate: Base learning rate.
        warmup_steps: Number of warmup steps.
        decay_steps: Number of decay steps.
        
    Returns:
        A function that takes step and returns learning rate.
    """
    factors = [f.strip() for f in factors.split('*')]
    
    def schedule(step):
        """Calculate learning rate based on step."""
        rate = 1.0
        for factor in factors:
            if factor == 'constant':
                rate *= base_learning_rate
            elif factor == 'linear_warmup':
                rate *= jnp.minimum(1.0, step / warmup_steps)
            elif factor == 'cosine_decay':
                rate *= 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step, decay_steps) / decay_steps))
            else:
                raise ValueError(f"Unknown factor: {factor}")
        return rate
    
    return schedule

def create_optimizer(config):
    """Create optimizer with learning rate schedule."""
    lr_schedule = create_learning_rate_scheduler(
        base_learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.max_steps,
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.training.adam_beta1,
            b2=config.training.adam_beta2,
            weight_decay=config.training.weight_decay
        )
    )
    
    return tx, lr_schedule

def create_train_state(model, config):
    """Create initial training state."""
    tx, _ = create_optimizer(config)
    
    state = TrainingState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=tx,
        ema_params=None,
        best_metrics={
            'loss': float('inf'),
            'accuracy': 0.0,
            'ec_improvement': 0.0,  # Track error correction improvement
        },
        tot_state={
            'enabled': config.training.get('use_tot', False),
            'search_strategy': config.training.get('tot_search_strategy', 'beam'),
            'thoughts_per_batch': 0,
            'best_thought_score': 0.0
        }
    )
    
    return state

class DataProcessor:
    """Processor for tokenizing and batching data."""
    
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.data.max_seq_length
    
    def tokenize_function(self, examples):
        """Tokenize a batch of text examples."""
        texts = examples[self.config.data.text_column]
        
        tokenized = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        
        # Create labels for autoregressive training (shift input_ids right)
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset):
        """Prepare dataset by tokenizing and formatting."""
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def collate_fn(self, examples):
        """Collate examples into a batch."""
        batch = {
            "input_ids": np.array([example["input_ids"] for example in examples]),
            "attention_mask": np.array([example["attention_mask"] for example in examples]),
            "labels": np.array([example["labels"] for example in examples]),
        }
        
        return batch

def create_train_dataloader(config):
    """Create data loader for training."""
    # Load the dataset
    dataset_name = config.data.dataset_name
    dataset = load_dataset(dataset_name, split=config.data.train_split)
    
    if config.data.max_train_samples is not None:
        dataset = dataset.select(range(config.data.max_train_samples))
    
    # Create tokenizer instance (assumes tokenizer is already provided)
    from vishwamai.tokenizer import VishwamAITokenizer
    tokenizer = VishwamAITokenizer(
        vocab_size=config.model.vocab_size, 
        model_prefix=config.model.name
    )
    
    # Process dataset
    data_processor = DataProcessor(tokenizer, config)
    processed_dataset = data_processor.prepare_dataset(dataset)
    
    # Create data loader
    def data_iterator():
        """Iterator that yields batches indefinitely."""
        epoch = 0
        while True:
            # Shuffle at the beginning of each epoch
            indices = list(range(len(processed_dataset)))
            random.shuffle(indices)
            
            # Create batches
            for i in range(0, len(indices), config.training.batch_size):
                batch_indices = indices[i:i + config.training.batch_size]
                examples = [processed_dataset[idx] for idx in batch_indices]
                yield data_processor.collate_fn(examples)
            
            epoch += 1
            print(f"Finished epoch {epoch}")
    
    return data_iterator()

def create_val_dataloader(config):
    """Create data loader for validation."""
    # Load the dataset
    dataset_name = config.data.dataset_name
    dataset = load_dataset(dataset_name, split=config.data.val_split)
    
    if config.data.max_val_samples is not None:
        dataset = dataset.select(range(config.data.max_val_samples))
    
    # Create tokenizer instance (assumes tokenizer is already provided)
    from vishwamai.tokenizer import VishwamAITokenizer
    tokenizer = VishwamAITokenizer(
        vocab_size=config.model.vocab_size, 
        model_prefix=config.model.name
    )
    
    # Process dataset
    data_processor = DataProcessor(tokenizer, config)
    processed_dataset = data_processor.prepare_dataset(dataset)
    
    # Create data loader that iterates once
    def data_iterator():
        """Iterator that yields batches once."""
        indices = list(range(len(processed_dataset)))
        
        # Create batches
        for i in range(0, len(indices), config.training.eval_batch_size):
            batch_indices = indices[i:i + config.training.eval_batch_size]
            examples = [processed_dataset[idx] for idx in batch_indices]
            yield data_processor.collate_fn(examples)
    
    return data_iterator()

def train_step(state, batch, model_config, z_loss=0.0, rng_key=None):
    """Perform a single training step with ToT integration."""
    use_tot = state.tot_state['enabled'] if state.tot_state else False
    
    def loss_fn(params):
        # Split PRNG key for different operations
        step_key, dropout_key, tot_key = jax.random.split(rng_key, 3) if rng_key else (None, None, None)
        
        # Forward pass with optional ToT integration
        outputs = state.apply_fn(
            {'params': params}, 
            batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            deterministic=False,
            rngs={'dropout': dropout_key},
            use_tot=use_tot,
            tot_rng_key=tot_key if use_tot else None
        )
        
        logits = outputs.get('last_hidden_state', None)
        if logits is None:
            raise ValueError("Model output doesn't contain 'last_hidden_state'")
        
        # Apply linear head to get vocabulary distribution
        head_params = params.get('lm_head', None)
        if head_params is not None:
            logits = state.apply_fn({'params': head_params}, logits)
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = batch['labels'][:, 1:]
        
        # Compute appropriate loss based on ToT availability
        if use_tot and 'tot_outputs' in outputs:
            loss = tot_guided_loss(
                shift_logits, 
                shift_labels, 
                outputs['tot_outputs'],
                alpha=model_config.get('tot_guidance_alpha', 0.1)
            )
        else:
            loss = cross_entropy_loss(shift_logits, shift_labels, z_loss=z_loss)
        
        # Add MoE load balancing loss if available
        if 'mod_weights' in outputs:
            mod_weights = outputs['mod_weights']
            # Encourage balanced expert usage with entropy maximization
            entropy = -jnp.mean(jnp.sum(mod_weights * jnp.log(mod_weights + 1e-8), axis=-1))
            balance_weight = model_config.get('mod_balance_weight', 0.01)
            loss = loss - balance_weight * entropy
        
        # Update ToT metrics if available
        if use_tot and 'tot_outputs' in outputs:
            thought = outputs['tot_outputs'].get('thought', None)
            if thought is not None:
                # Count valid thoughts and track best score
                thought_score = thought.score
                state.tot_state['thoughts_per_batch'] += 1
                state.tot_state['best_thought_score'] = max(
                    state.tot_state['best_thought_score'], 
                    thought_score
                )
        
        return loss
    
    # Compute loss and gradients
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    
    # Update parameters
    new_state = state.apply_gradients(grads=grads)
    
    # Update EMA parameters if enabled
    if state.ema_params is not None:
        new_ema_params = jax.tree_map(
            lambda ema, param: ema * 0.999 + param * 0.001,
            state.ema_params, new_state.params
        )
        new_state = new_state.replace(ema_params=new_ema_params)
    
    metrics = {'loss': loss}
    
    return new_state, metrics

def eval_step(state, batch, use_tot=False, rng_key=None):
    """Perform a single evaluation step with optional ToT integration."""
    # Split PRNG key for different operations
    dropout_key, tot_key = jax.random.split(rng_key, 2) if rng_key else (None, None)
    
    # Forward pass with optional ToT
    outputs = state.apply_fn(
        {'params': state.params}, 
        batch['input_ids'], 
        attention_mask=batch['attention_mask'],
        deterministic=True,
        rngs={'dropout': dropout_key} if dropout_key else None,
        use_tot=use_tot,
        tot_rng_key=tot_key if use_tot else None
    )
    
    logits = outputs.get('last_hidden_state', None)
    if logits is None:
        raise ValueError("Model output doesn't contain 'last_hidden_state'")
    
    # Apply linear head to get vocabulary distribution
    head_params = state.params.get('lm_head', None)
    if head_params is not None:
        logits = state.apply_fn({'params': head_params}, logits)
    
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = batch['labels'][:, 1:]
    
    # Compute metrics
    metrics = compute_metrics(shift_logits, shift_labels)
    
    # Add ToT metrics if available
    if use_tot and 'tot_outputs' in outputs:
        tot_outputs = outputs['tot_outputs']
        if 'thought' in tot_outputs and tot_outputs['thought'] is not None:
            metrics['tot_score'] = tot_outputs['thought'].score
        if 'integration_info' in tot_outputs:
            tot_weight, model_weight = tot_outputs['integration_info'][:2]
            metrics['tot_weight'] = jnp.mean(tot_weight)
            metrics['model_weight'] = jnp.mean(model_weight)
    
    return metrics

def initialize_tot_components(model, config):
    """Initialize Tree of Thoughts components for training."""
    if not config.training.get('use_tot', False):
        return model
    
    logger.info("Initializing Tree of Thoughts components for training")
    
    # Create dummy transformer for ToT
    vision_transformer = VishwamAIModel(model.config)
    
    # Create ToT model
    tot_model = TreeOfThoughts(
        transformer=vision_transformer,
        max_thoughts=config.training.get('tot_max_thoughts', 5),
        max_depth=config.training.get('tot_max_depth', 3),
        beam_width=config.training.get('tot_beam_width', 8),
        pruning_threshold=config.training.get('tot_pruning_threshold', 0.3),
        exploration_factor=config.training.get('tot_exploration_factor', 1.0)
    )
    
    # Create integration components
    integration_layer = ToTIntegrationLayer(model.config)
    mla = MultiLevelToTAttention(
        hidden_size=model.config.hidden_size,
        num_heads=min(8, model.config.num_attention_heads)
    )
    
    # Create MoD component if enabled
    if config.model.get('use_mod', False):
        mod_layer = MixtureDensityNetwork(
            hidden_size=model.config.hidden_size,
            num_mixtures=config.model.get('mod_num_mixtures', 5)
        )
        model.mod_layer = mod_layer
    
    # Add components to model
    model.tot_model = tot_model
    model.tot_integration = integration_layer
    model.tot_mla = mla
    
    # Set flag to use ToT
    model.use_tot = True
    
    return model

def train(
    model,
    config,
    train_dataloader,
    val_dataloader=None,
    num_steps=10000,
    log_every=100,
    eval_every=1000,
    checkpoint_dir=None,
):
    """
    Enhanced training loop with ToT integration and error correction.
    
    Args:
        model: The model to train
        config: Configuration
        train_dataloader: Iterator yielding training batches
        val_dataloader: Iterator yielding validation batches
        num_steps: Total number of training steps
        log_every: Log metrics every N steps
        eval_every: Evaluate model every N steps
        checkpoint_dir: Directory to save checkpoints
    """
    logger.info("Starting training with enhanced ToT integration and error correction")
    
    # Initialize ToT components if enabled
    model = initialize_tot_components(model, config)
    
    # Initialize error correction trainer if enabled
    use_error_correction = config.training.get('use_error_correction', True)
    if use_error_correction:
        logger.info("Initializing error correction system")
        error_trainer = ErrorCorrectionTrainer(
            config=config,
            use_tot=config.training.get('use_tot', False),
            use_mod=config.model.get('use_mod', False),
            history_size=config.training.get('error_history_size', 100),
            threshold_percentile=config.training.get('error_threshold_percentile', 85.0)
        )
        
        # Create enhanced train and eval step functions
        ec_train_step = create_error_corrected_train_step(
            train_step,
            error_trainer,
            alpha=config.training.get('ec_loss_weight', 0.2)
        )
        
        ec_eval_step = create_error_corrected_eval_step(
            eval_step,
            error_trainer
        )
    else:
        # Use standard train and eval steps
        ec_train_step = train_step
        ec_eval_step = eval_step
        error_trainer = None
    
    # Create training state
    state = create_train_state(model, config)
    
    # Set up PRNG key for reproducibility
    rng_key = jax.random.PRNGKey(config.training.seed)
    
    # Training loop
    for step in range(num_steps):
        # Get next batch
        batch = next(train_dataloader)
        
        # Step PRNG key
        rng_key, step_key = jax.random.split(rng_key)
        
        # Training step with ToT integration and error correction
        state, metrics = ec_train_step(
            state, batch, config.model,
            z_loss=config.training.get('z_loss', 0.0),
            rng_key=step_key
        )
        
        # Log metrics
        if step % log_every == 0:
            logger.info(f"Step {step}/{num_steps}: loss = {metrics['loss']:.4f}")
            
            # Log ToT metrics if available
            if state.tot_state and state.tot_state['enabled']:
                logger.info(f"ToT metrics - thoughts/batch: {state.tot_state['thoughts_per_batch']}, "
                          f"best score: {state.tot_state['best_thought_score']:.4f}")
                
            # Log error correction metrics if available
            if use_error_correction and 'ec_loss' in metrics:
                logger.info(f"Error correction - base loss: {metrics.get('base_loss', 0.0):.4f}, "
                          f"ec loss: {metrics['ec_loss']:.4f}, "
                          f"threshold: {metrics.get('error_threshold', 0.0):.4f}, "
                          f"impact: {metrics.get('correction_impact', 0.0):.4f}")
        
        # Evaluate model
        if val_dataloader is not None and step % eval_every == 0:
            logger.info(f"Evaluating at step {step}/{num_steps}")
            
            # Run evaluation with and without ToT to compare
            eval_metrics = {}
            eval_metrics_tot = {}
            
            # Generate new evaluation key
            rng_key, eval_key = jax.random.split(rng_key)
            
            # Evaluate without ToT
            for eval_batch in val_dataloader():
                batch_metrics = ec_eval_step(state, eval_batch, use_tot=False, rng_key=eval_key)
                for k, v in batch_metrics.items():
                    eval_metrics[k] = eval_metrics.get(k, 0.0) + v
            
            # Evaluate with ToT if enabled
            if state.tot_state and state.tot_state['enabled']:
                rng_key, tot_eval_key = jax.random.split(rng_key)
                for eval_batch in val_dataloader():
                    batch_metrics = ec_eval_step(state, eval_batch, use_tot=True, rng_key=tot_eval_key)
                    for k, v in batch_metrics.items():
                        eval_metrics_tot[k] = eval_metrics_tot.get(k, 0.0) + v
            
            # Log evaluation metrics
            logger.info(f"Evaluation metrics at step {step}:")
            for k, v in eval_metrics.items():
                logger.info(f"  {k}: {v:.4f}")
                
            if eval_metrics_tot:
                logger.info(f"Evaluation metrics with ToT at step {step}:")
                for k, v in eval_metrics_tot.items():
                    logger.info(f"  {k}: {v:.4f}")
                
                # Compute improvement from ToT
                if 'loss' in eval_metrics and 'loss' in eval_metrics_tot:
                    improvement = eval_metrics['loss'] - eval_metrics_tot['loss']
                    logger.info(f"  ToT improvement: {improvement:.4f} ({improvement / eval_metrics['loss'] * 100:.2f}%)")
            
            # Run dedicated error correction evaluation if enabled
            if use_error_correction and error_trainer and step > 0 and step % (eval_every * 5) == 0:
                rng_key, ec_eval_key = jax.random.split(rng_key)
                ec_metrics = evaluate_error_correction(
                    model=model.bind({'params': state.params}),
                    dataloader=val_dataloader,
                    error_trainer=error_trainer,
                    num_batches=10,
                    rng_key=ec_eval_key
                )
                
                logger.info(f"Error correction evaluation at step {step}:")
                for k, v in ec_metrics.items():
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.4f}")
                    else:
                        logger.info(f"  {k}: {v}")
                
                # Add to eval metrics
                eval_metrics.update({f"ec_{k}": v for k, v in ec_metrics.items()})
            
            # Update best metrics
            if eval_metrics['loss'] < state.best_metrics['loss']:
                state.best_metrics['loss'] = eval_metrics['loss']
                state.best_metrics['accuracy'] = eval_metrics['accuracy']
                
                # Track error correction improvement if available
                if 'ec_improvement' in eval_metrics:
                    state.best_metrics['ec_improvement'] = eval_metrics['ec_improvement']
                
                # Save best checkpoint
                if checkpoint_dir:
                    save_checkpoint(state, os.path.join(checkpoint_dir, "best_model"))
        
        # Save checkpoint
        if checkpoint_dir and step % config.training.save_every == 0:
            save_checkpoint(state, os.path.join(checkpoint_dir, f"checkpoint_{step}"))
    
    # Save final model
    if checkpoint_dir:
        save_checkpoint(state, os.path.join(checkpoint_dir, "final_model"))
    
    return state

def save_checkpoint(state, path):
    """Save a checkpoint of the training state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}.msgpack", "wb") as f:
        f.write(flax.serialization.msgpack_serialize(state))
    logger.info(f"Saved checkpoint to {path}.msgpack")

def load_checkpoint(path, state):
    """Load a checkpoint into the training state."""
    with open(f"{path}.msgpack", "rb") as f:
        loaded_state = flax.serialization.msgpack_restore(f.read())
    return loaded_state

def main(config_path):
    """Main training function with config from file."""
    # Load config
    config = OmegaConf.load(config_path)
    
    # Initialize model
    from vishwamai.model import VishwamAIModel, ModelConfig
    model_config = ModelConfig(**config.model)
    model = VishwamAIModel(model_config)
    
    # Create data loaders
    train_dataloader = create_train_dataloader(config)
    val_dataloader = create_val_dataloader(config)
    
    # Create checkpoints directory
    checkpoint_dir = config.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train model
    final_state = train(
        model,
        config,
        train_dataloader,
        val_dataloader=val_dataloader,
        num_steps=config.training.max_steps,
        log_every=config.training.log_every,
        eval_every=config.training.eval_every,
        checkpoint_dir=checkpoint_dir
    )
    
    logger.info("Training completed!")
    logger.info(f"Best loss: {final_state.best_metrics['loss']:.4f}")
    logger.info(f"Best accuracy: {final_state.best_metrics['accuracy']:.4f}")
    
    return final_state
