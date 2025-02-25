import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, List, Tuple, Any, Optional, NamedTuple, Callable
import logging
from functools import partial
import numpy as np
from tqdm import tqdm

from .error_correction import ErrorCorrectionModule, ErrorMetrics, compute_error_metrics
from .tot import TreeOfThoughts, Thought, SearchState
from .integration import ToTIntegrationLayer, MixtureDensityNetwork, MultiLevelToTAttention
from .training import tot_guided_loss, cross_entropy_loss

logger = logging.getLogger(__name__)

class ErrorCorrectionState(NamedTuple):
    """State for tracking error correction during training."""
    error_threshold: float = 0.1
    error_history: jnp.ndarray = None
    correction_strength: float = 1.0
    history_idx: int = 0
    tot_triggered: int = 0  # Count of ToT activations due to errors
    avg_correction_impact: float = 0.0  # Average impact of corrections

def create_error_correction_state(history_size: int = 100):
    """Create initial error correction state."""
    return ErrorCorrectionState(
        error_history=jnp.zeros(history_size),
    )

class ErrorCorrectionTrainer:
    """
    Enhanced error correction system that integrates with training
    and utilizes Tree of Thoughts for complex error resolution.
    """
    def __init__(
        self, 
        config: Dict[str, Any],
        use_tot: bool = True,
        use_mod: bool = True,
        history_size: int = 100,
        threshold_percentile: float = 85.0
    ):
        self.config = config
        self.use_tot = use_tot
        self.use_mod = use_mod
        self.history_size = history_size
        self.threshold_percentile = threshold_percentile
        
        # Create error correction module
        hidden_size = config.model.hidden_size
        self.error_module = ErrorCorrectionModule(
            hidden_size=hidden_size,
            num_heads=min(8, config.model.num_attention_heads),
            memory_size=config.get('error_memory_size', 1024)
        )
        
        # Initialize state
        self.state = create_error_correction_state(history_size)
        
        # Create integration components if tot is enabled
        if self.use_tot:
            self.tot_integration = ToTIntegrationLayer(config.model)
        
        # Create MoD component if enabled
        if self.use_mod:
            self.mod_layer = MixtureDensityNetwork(
                hidden_size=hidden_size,
                num_mixtures=config.get('mod_num_mixtures', 5)
            )
            
    def update_error_history(self, error: float) -> ErrorCorrectionState:
        """Update error history and dynamically adjust threshold."""
        # Circular buffer update
        state = self.state
        error_history = state.error_history.at[state.history_idx].set(error)
        history_idx = (state.history_idx + 1) % self.history_size
        
        # Compute error statistics
        if jnp.count_nonzero(error_history) > 10:  # Ensure enough data points
            # Use percentile for robust thresholding
            error_threshold = jnp.percentile(
                error_history[error_history > 0],  # Only consider non-zero errors
                self.threshold_percentile
            )
            
            # Adapt correction strength based on error distribution
            mean_error = jnp.mean(error_history[error_history > 0])
            std_error = jnp.std(error_history[error_history > 0])
            
            # Stronger correction for larger errors relative to distribution
            correction_strength = jax.nn.sigmoid((error - mean_error) / (std_error + 1e-6) * 2)
        else:
            # Default values until we have enough history
            error_threshold = state.error_threshold
            correction_strength = state.correction_strength
            
        return state._replace(
            error_history=error_history,
            history_idx=history_idx,
            error_threshold=error_threshold,
            correction_strength=correction_strength
        )
    
    def apply_error_correction(
        self,
        logits: jnp.ndarray,
        features: jnp.ndarray,
        labels: Optional[jnp.ndarray] = None,
        training: bool = True,
        rng_key: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, ErrorCorrectionState]:
        """Apply error correction to model outputs."""
        # Apply base error correction
        outputs = self.error_module(features, training=training)
        corrected_logits = outputs.corrected
        error_gate = outputs.error_gate
        
        # Compute error metrics if we have labels
        if labels is not None:
            metrics = compute_error_metrics(logits, labels)
            error = metrics.mae
            
            # Update error history
            self.state = self.update_error_history(error)
            
            # Check if error exceeds threshold
            if error > self.state.error_threshold and self.use_tot and rng_key is not None:
                # Error is significant, trigger Tree of Thoughts
                logger.info(f"Error {error:.4f} exceeds threshold {self.state.error_threshold:.4f}, triggering ToT")
                
                # Create corrected features
                corrected_features = features * (1 - error_gate) + corrected_logits * error_gate
                
                # Generate thoughts about the error
                tot_outputs = self._generate_error_thoughts(
                    features=corrected_features,
                    error=error,
                    rng_key=rng_key
                )
                
                if tot_outputs is not None:
                    # Apply MoD if enabled
                    if self.use_mod:
                        mod_features, mixture_weights = self.mod_layer(corrected_features)
                        tot_features = tot_outputs['thought_features']
                        
                        # Integrate ToT and MoD features
                        integrated_features, integration_info = self.tot_integration(
                            mod_features, tot_features
                        )
                        
                        # Apply stronger correction for more serious errors
                        strength = self.state.correction_strength
                        corrected_logits = logits * (1 - strength) + integrated_features * strength
                        
                        # Update state with ToT trigger
                        self.state = self.state._replace(
                            tot_triggered=self.state.tot_triggered + 1
                        )
                        
                        # Measure correction impact
                        if labels is not None:
                            before_loss = cross_entropy_loss(logits, labels)
                            after_loss = cross_entropy_loss(corrected_logits, labels)
                            impact = (before_loss - after_loss) / before_loss
                            
                            # Update rolling average of correction impact
                            avg_impact = self.state.avg_correction_impact
                            self.state = self.state._replace(
                                avg_correction_impact=(avg_impact * 0.95 + impact * 0.05)
                            )
                            
        return corrected_logits, self.state
    
    def _generate_error_thoughts(
        self,
        features: jnp.ndarray,
        error: float,
        rng_key: jnp.ndarray
    ) -> Optional[Dict[str, Any]]:
        """Generate thoughts to analyze and correct errors."""
        # Import here to avoid circular imports
        from .tot import TreeOfThoughts
        from .transformer import VishwamAIModel
        
        try:
            # Create a minimal transformer for ToT
            transformer = VishwamAIModel(self.config.model)
            
            # Create ToT model
            tot = TreeOfThoughts(
                transformer=transformer,
                max_thoughts=self.config.get('tot_max_thoughts', 5),
                max_depth=self.config.get('tot_max_depth', 3),
                beam_width=self.config.get('tot_beam_width', 8),
                pruning_threshold=0.2  # Lower threshold to generate more thoughts
            )
            
            # Generate error analysis thoughts
            thought = tot(features, rng_key)
            
            if thought is None:
                return None
            
            # Collect thought features
            thought_features = []
            current_thought = thought
            while current_thought:
                thought_features.append(current_thought.embeddings)
                if current_thought.children:
                    current_thought = max(current_thought.children, key=lambda t: t.score)
                else:
                    break
            
            # Return thought features for integration
            if thought_features:
                return {
                    'thought': thought,
                    'thought_features': jnp.stack(thought_features),
                    'error': error
                }
            return None
        
        except Exception as e:
            logger.warning(f"Error in thought generation: {str(e)}")
            return None

def create_error_corrected_train_step(
    base_train_step_fn: Callable,
    error_trainer: ErrorCorrectionTrainer,
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

def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    perplexity = jnp.exp(loss)
    return {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity
    }

def create_error_corrected_eval_step(
    base_eval_step_fn: Callable,
    error_trainer: ErrorCorrectionTrainer
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
    error_trainer: ErrorCorrectionTrainer,
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
    for i in tqdm(range(num_batches)):
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
