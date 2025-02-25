"""
Training step functions to avoid circular imports between modules.
"""
import jax
import jax.numpy as jnp
from typing import Dict, Any, Callable, Tuple
import logging

# Add import for loss functions
from .loss_functions import cross_entropy_loss, tot_guided_loss, compute_metrics

logger = logging.getLogger(__name__)

def create_error_corrected_train_step(
    base_train_step_fn: Callable,
    error_trainer,
    alpha: float = 0.2
):
    """
    Create a training step that incorporates error correction.
    
    Args:
        base_train_step_fn: Base training step function
        error_trainer: ErrorCorrectionTrainer instance
        alpha: Weight of error correction loss in total loss
        
    Returns:
        Enhanced training step function with error correction
    """
    def train_step_with_error_correction(state, batch, model_config, z_loss=0.0, rng_key=None):
        """Train step with integrated error correction."""
        use_tot = state.tot_state['enabled'] if hasattr(state, 'tot_state') else False
        
        def loss_fn(params):
            # Split PRNG key for different operations
            if rng_key is not None:
                step_key, dropout_key, tot_key, ec_key = jax.random.split(rng_key, 4)
            else:
                step_key = dropout_key = tot_key = ec_key = None
            
            # Forward pass with optional ToT integration
            outputs = state.apply_fn(
                {'params': params}, 
                batch['input_ids'], 
                attention_mask=batch['attention_mask'],
                deterministic=False,
                rngs={'dropout': dropout_key} if dropout_key else None,
                use_tot=use_tot,
                tot_rng_key=tot_key if use_tot and tot_key else None
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
            
            # Apply error correction
            corrected_logits, ec_state = error_trainer.apply_error_correction(
                shift_logits, 
                logits[:, :-1, :],  # Use full features for correction
                shift_labels,
                training=True,
                rng_key=ec_key
            )
            
            # Compute base loss
            if use_tot and 'tot_outputs' in outputs:
                base_loss = tot_guided_loss(
                    shift_logits, 
                    shift_labels, 
                    outputs['tot_outputs'],
                    alpha=model_config.get('tot_guidance_alpha', 0.1)
                )
            else:
                base_loss = cross_entropy_loss(shift_logits, shift_labels, z_loss=z_loss)
            
            # Compute error correction loss
            ec_loss = cross_entropy_loss(corrected_logits, shift_labels)
            
            # Combined loss with weighting
            loss = (1 - alpha) * base_loss + alpha * ec_loss
            
            # Add MoE/MoD balance loss if available
            if 'mod_weights' in outputs:
                mod_weights = outputs['mod_weights']
                entropy = -jnp.mean(jnp.sum(mod_weights * jnp.log(mod_weights + 1e-8), axis=-1))
                balance_weight = model_config.get('mod_balance_weight', 0.01)
                loss = loss - balance_weight * entropy
            
            # Log metrics for monitoring
            metrics = {
                'base_loss': base_loss,
                'ec_loss': ec_loss,
                'error_threshold': ec_state.error_threshold,
                'correction_strength': ec_state.correction_strength,
                'tot_triggered': ec_state.tot_triggered,
                'correction_impact': ec_state.avg_correction_impact
            }
            
            return loss, metrics
        
        # Compute loss, metrics and gradients
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # Update parameters
        new_state = state.apply_gradients(grads=grads)
        
        # Update EMA parameters if enabled
        if state.ema_params is not None:
            new_ema_params = jax.tree_map(
                lambda ema, param: ema * 0.999 + param * 0.001,
                state.ema_params, new_state.params
            )
            new_state = new_state.replace(ema_params=new_ema_params)
        
        # Add loss to metrics
        metrics['loss'] = loss
        
        return new_state, metrics
    
    return train_step_with_error_correction

def create_error_corrected_eval_step(
    base_eval_step_fn: Callable,
    error_trainer
):
    """
    Create an evaluation step that incorporates error correction.
    
    Args:
        base_eval_step_fn: Base evaluation step function
        error_trainer: ErrorCorrectionTrainer instance
        
    Returns:
        Enhanced evaluation step function with error correction
    """
    def eval_step_with_error_correction(state, batch, use_tot=False, rng_key=None):
        """Evaluation step with integrated error correction."""
        # Split PRNG key for different operations
        if rng_key is not None:
            dropout_key, tot_key, ec_key = jax.random.split(rng_key, 3)
        else:
            dropout_key = tot_key = ec_key = None
        
        # Forward pass with optional ToT
        outputs = state.apply_fn(
            {'params': state.params}, 
            batch['input_ids'], 
            attention_mask=batch['attention_mask'],
            deterministic=True,
            rngs={'dropout': dropout_key} if dropout_key else None,
            use_tot=use_tot,
            tot_rng_key=tot_key if use_tot and tot_key else None
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
        
        # Apply error correction (only for metrics, not actual training)
        corrected_logits, _ = error_trainer.apply_error_correction(
            shift_logits,
            logits[:, :-1, :],
            shift_labels,
            training=False,
            rng_key=ec_key
        )
        
        # Compute regular metrics
        base_metrics = compute_metrics(shift_logits, shift_labels)
        
        # Compute metrics with error correction
        ec_metrics = compute_metrics(corrected_logits, shift_labels)
        for k, v in ec_metrics.items():
            base_metrics[f'ec_{k}'] = v
        
        # Add improvement metrics
        base_loss = base_metrics['loss']
        ec_loss = base_metrics['ec_loss']
        if base_loss > 0:
            base_metrics['ec_improvement'] = (base_loss - ec_loss) / base_loss
        
        # Add ToT metrics if available
        if use_tot and 'tot_outputs' in outputs:
            tot_outputs = outputs['tot_outputs']
            if 'thought' in tot_outputs and tot_outputs['thought'] is not None:
                base_metrics['tot_score'] = tot_outputs['thought'].score
            if 'integration_info' in tot_outputs:
                tot_weight, model_weight = tot_outputs['integration_info'][:2]
                base_metrics['tot_weight'] = jnp.mean(tot_weight)
                base_metrics['model_weight'] = jnp.mean(model_weight)
        
        return base_metrics
    
    return eval_step_with_error_correction

def evaluate_error_correction(
    model,
    dataloader,
    error_trainer,
    num_batches: int = 10,
    rng_key = None
) -> Dict[str, float]:
    """
    Evaluate the error correction system on validation data.
    
    Args:
        model: The model to evaluate
        dataloader: Data loader for validation data
        error_trainer: ErrorCorrectionTrainer instance
        num_batches: Number of batches to evaluate
        rng_key: Random key
    
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating error correction performance")
    
    metrics = {
        'base_loss': 0.0,
        'corrected_loss': 0.0,
        'improvement': 0.0,
        'tot_triggered': 0,
        'base_perplexity': 0.0,
        'corrected_perplexity': 0.0
    }
    
    data_iterator = dataloader()
    for i in range(num_batches):
        try:
            batch = next(data_iterator)
        except StopIteration:
            # Restart iterator if we run out of data
            data_iterator = dataloader()
            batch = next(data_iterator)
        
        # Split PRNG key
        if rng_key is not None:
            rng_key, step_key = jax.random.split(rng_key)
        else:
            step_key = None
            
        # Forward pass
        outputs = model(
            batch['input_ids'],
            attention_mask=batch['attention_mask'],
            deterministic=True
        )
        
        logits = outputs.get('last_hidden_state', None)
        if logits is None:
            continue
        
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = batch['labels'][:, 1:]
        
        # Apply error correction
        corrected_logits, ec_state = error_trainer.apply_error_correction(
            shift_logits,
            logits[:, :-1, :],
            shift_labels,
            training=False,
            rng_key=step_key
        )
        
        # Compute losses
        base_loss = cross_entropy_loss(shift_logits, shift_labels)
        corrected_loss = cross_entropy_loss(corrected_logits, shift_labels)
        
        # Update metrics
        metrics['base_loss'] += float(base_loss)
        metrics['corrected_loss'] += float(corrected_loss)
        metrics['tot_triggered'] += ec_state.tot_triggered
        metrics['base_perplexity'] += float(jnp.exp(base_loss))
        metrics['corrected_perplexity'] += float(jnp.exp(corrected_loss))
    
    # Compute averages
    for key in ['base_loss', 'corrected_loss', 'base_perplexity', 'corrected_perplexity']:
        metrics[key] /= num_batches
    
    # Compute improvement percentage
    if metrics['base_loss'] > 0:
        metrics['improvement'] = (metrics['base_loss'] - metrics['corrected_loss']) / metrics['base_loss'] * 100
    
    logger.info(f"Error correction evaluation complete")
    logger.info(f"Base loss: {metrics['base_loss']:.4f}, Corrected loss: {metrics['corrected_loss']:.4f}")
    logger.info(f"Improvement: {metrics['improvement']:.2f}%")
    logger.info(f"ToT triggered: {metrics['tot_triggered']} times")
    
    return metrics
