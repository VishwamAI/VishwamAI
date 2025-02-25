from typing import Optional, Dict, Any, NamedTuple
from dataclasses import dataclass
from functools import partial
import logging
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
# Re-export ErrorCorrectionTrainer from error_correction_trainer to maintain compatibility
from .error_correction_trainer import ErrorCorrectionTrainer

logger = logging.getLogger(__name__)

class ErrorCorrectionOutput(NamedTuple):
    """Output structure for error correction module."""
    corrected: jnp.ndarray
    error_gate: jnp.ndarray

@dataclass
class ErrorMetrics:
    """Metrics for error tracking and thresholding."""
    mse: float
    mae: float
    threshold: float
    correction_strength: float

class ErrorCorrectionModule(nn.Module):
    """Module for error detection and correction in model outputs."""
    hidden_dim: int = 1024
    num_correction_layers: int = 2
    correction_threshold: float = 0.7
    
    @nn.compact
    def __call__(self, hidden_states, labels=None, deterministic=True):
        # Error detection network
        detection_logits = nn.Dense(features=1, name="error_detector")(hidden_states)
        detection_probs = jax.nn.sigmoid(detection_logits)
        
        # Error correction network
        correction_states = hidden_states
        for i in range(self.num_correction_layers):
            correction_states = nn.Dense(features=self.hidden_dim)(correction_states)
            correction_states = nn.gelu(correction_states)
            correction_states = nn.Dropout(rate=0.1, deterministic=deterministic)(correction_states)
        
        corrected_states = nn.Dense(features=hidden_states.shape[-1])(correction_states)
        
        # Apply correction based on detection
        error_mask = detection_probs > self.correction_threshold
        final_states = jnp.where(error_mask, corrected_states, hidden_states)
        
        result = {
            "corrected_states": final_states,
            "error_probs": detection_probs,
            "correction_mask": error_mask
        }
        
        if labels is not None:
            # Calculate error detection metrics when labels are provided
            has_error = labels != jnp.argmax(hidden_states @ hidden_states.T, axis=-1)
            detection_loss = optax.sigmoid_binary_cross_entropy(
                detection_logits, has_error.astype(jnp.float32)
            ).mean()
            result["detection_loss"] = detection_loss
            
        return result

def compute_error_metrics(logits, corrected_logits, labels):
    """Compute metrics to evaluate error correction performance."""
    original_predictions = jnp.argmax(logits, axis=-1)
    corrected_predictions = jnp.argmax(corrected_logits, axis=-1)
    
    original_accuracy = jnp.mean((original_predictions == labels).astype(jnp.float32))
    corrected_accuracy = jnp.mean((corrected_predictions == labels).astype(jnp.float32))
    
    # Calculate where corrections helped vs. where they made things worse
    original_correct = original_predictions == labels
    corrected_correct = corrected_predictions == labels
    
    helped = jnp.logical_and(~original_correct, corrected_correct)
    hurt = jnp.logical_and(original_correct, ~corrected_correct)
    
    helped_ratio = jnp.sum(helped) / jnp.sum(~original_correct) if jnp.sum(~original_correct) > 0 else 0.0
    hurt_ratio = jnp.sum(hurt) / jnp.sum(original_correct) if jnp.sum(original_correct) > 0 else 0.0
    
    return {
        "original_accuracy": original_accuracy,
        "corrected_accuracy": corrected_accuracy,
        "helped_ratio": helped_ratio,
        "hurt_ratio": hurt_ratio,
        "net_improvement": corrected_accuracy - original_accuracy,
    }

def create_error_corrected_train_step(base_train_step, error_correction_weight=0.5):
    """Wraps a training step with error correction capabilities."""
    def error_corrected_train_step(state, batch, dropout_rng):
        base_outputs, updated_state = base_train_step(state, batch, dropout_rng)
        
        # Apply error correction module
        error_module = ErrorCorrectionModule()
        error_outputs = error_module(
            hidden_states=base_outputs["hidden_states"], 
            labels=batch.get("labels"),
            deterministic=False
        )
        
        # Combine losses
        total_loss = base_outputs["loss"]
        if "detection_loss" in error_outputs:
            total_loss = total_loss + error_correction_weight * error_outputs["detection_loss"]
        
        # Update metrics
        metrics = base_outputs.get("metrics", {})
        metrics.update({
            "error_detection_loss": error_outputs.get("detection_loss", 0.0),
            "error_correction_rate": jnp.mean(error_outputs["correction_mask"].astype(jnp.float32))
        })
        
        return {
            **base_outputs,
            "corrected_hidden_states": error_outputs["corrected_states"],
            "error_probs": error_outputs["error_probs"],
            "metrics": metrics,
            "loss": total_loss
        }, updated_state
    
    return error_corrected_train_step

def create_error_corrected_eval_step(base_eval_step):
    """Wraps an evaluation step with error correction capabilities."""
    def error_corrected_eval_step(state, batch):
        base_outputs = base_eval_step(state, batch)
        
        # Apply error correction module in eval mode
        error_module = ErrorCorrectionModule()
        error_outputs = error_module(
            hidden_states=base_outputs["hidden_states"], 
            labels=batch.get("labels"),
            deterministic=True
        )
        
        # Calculate error correction metrics
        if "logits" in base_outputs and "labels" in batch:
            corrected_logits = error_outputs["corrected_states"] @ state.params["output_projection"]["kernel"].T
            error_metrics = compute_error_metrics(
                base_outputs["logits"], 
                corrected_logits,
                batch["labels"]
            )
            
            metrics = base_outputs.get("metrics", {})
            metrics.update(error_metrics)
            base_outputs["metrics"] = metrics
        
        base_outputs["corrected_hidden_states"] = error_outputs["corrected_states"]
        base_outputs["error_probs"] = error_outputs["error_probs"]
        
        return base_outputs
    
    return error_corrected_eval_step

class AdaptiveErrorCorrection(nn.Module):
    """
    Enhanced error correction that adapts during training with ToT integration.
    """
    hidden_size: int
    num_heads: int = 4
    dropout_rate: float = 0.1
    qkv_features: Optional[int] = None
    use_tot_features: bool = True
    
    @nn.compact
    def __call__(self, 
                features: jnp.ndarray):
        pass

@partial(jax.jit, static_argnums=(2,))
def compute_error_metrics(predictions: jnp.ndarray, 
                        targets: jnp.ndarray,
                        prev_threshold: float = 0.1) -> ErrorMetrics:
    """
    JIT-compiled error metrics computation with enhanced stability.
    """
    logger.info("Computing error metrics with parallel processing")
    
    # Parallel computation of error metrics
    def compute_batch_metrics(p, t):
        squared_error = (p - t) ** 2
        abs_error = jnp.abs(p - t)
        return squared_error, abs_error
    
    squared_errors, abs_errors = jax.vmap(compute_batch_metrics)(predictions, targets)
    
    # Compute robust statistics
    mse = jnp.mean(squared_errors)
    mae = jnp.mean(abs_errors)
    
    # Compute error percentiles for robust thresholding
    error_percentiles = jnp.percentile(abs_errors, jnp.array([25, 50, 75]))
    iqr = error_percentiles[2] - error_percentiles[0]
    
    # Robust threshold adjustment using IQR
    threshold_adjustment = jnp.clip(
        (error_percentiles[1] - prev_threshold) / (iqr + 1e-6),
        -0.1, 0.1
    )
    
    new_threshold = jnp.clip(
        prev_threshold * (1.0 + threshold_adjustment),
        0.05,  # Minimum threshold
        0.2    # Maximum threshold
    )
    
    # Adaptive correction strength using robust statistics
    correction_strength = jax.nn.sigmoid(
        (mae - error_percentiles[1]) / (iqr + 1e-6)
    )
    
    return ErrorMetrics(
        mse=float(mse),
        mae=float(mae),
        threshold=float(new_threshold),
        correction_strength=float(correction_strength)
    )

class ModelIntegrator:
    """Integrates DeepSeek, ToT and Transformer components with adaptive error correction."""
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ModelIntegrator with the given configuration.

        Args:
            config (Dict[str, Any]): Model configuration.
        """
        self.config = config
        self.error_threshold = 0.1
        self.error_history = jnp.zeros(10)  # Keep track of recent errors
        self.history_idx = 0
        
    def update_error_history(self, error: float) -> None:
        """
        Update the rolling error history for adaptive thresholding.

        Args:
            error (float): New error value to add to history.
        """
        self.error_history = self.error_history.at[self.history_idx].set(error)
        self.history_idx = (self.history_idx + 1) % 10
        
        # Adapt threshold based on recent error statistics
        mean_error = jnp.mean(self.error_history)
        std_error = jnp.std(self.error_history)
        self.error_threshold = jnp.clip(
            mean_error + std_error,
            0.05,  # Minimum threshold
            0.2    # Maximum threshold
        )
    
    def validate_outputs(self, 
                        outputs: jnp.ndarray, 
                        expected: Optional[jnp.ndarray] = None) -> bool:
        """
        Validate model outputs and detect anomalies.

        Args:
            outputs (jnp.ndarray): Model outputs.
            expected (Optional[jnp.ndarray]): Expected outputs.

        Returns:
            bool: True if outputs are valid, False otherwise.
        """
        logger.info("Validating model outputs")
        if expected is not None:
            error_rate = jnp.mean(jnp.abs(outputs - expected))
            self.update_error_history(error_rate)
            return error_rate < self.error_threshold
        return True
    
    def error_correction_strategy(self, 
                                outputs: jnp.ndarray, 
                                errors: float) -> jnp.ndarray:
        """
        Apply adaptive error correction based on detected errors.

        Args:
            outputs (jnp.ndarray): Model outputs.
            errors (float): Detected error rate.

        Returns:
            jnp.ndarray: Corrected outputs.
        """
        logger.info("Applying error correction strategy")
        
        if errors > self.error_threshold:
            # Compute smooth correction factor
            correction = jnp.tanh(errors / self.error_threshold) * 0.5
            
            # Apply adaptive correction
            corrected = outputs - correction * (outputs - jnp.mean(outputs, axis=-1, keepdims=True))
            
            # Update error history for future adaptations
            self.update_error_history(errors)
            
            return corrected
        return outputs
