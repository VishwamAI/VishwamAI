"""
Loss functions and evaluation metrics for VishwamAI models.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional


def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    """
    Compute cross entropy loss with optional masking.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
        mask: Mask where 1 indicates valid positions [batch, seq_len]
        
    Returns:
        Loss value
    """
    # Shape checks
    if logits.ndim == 3 and labels.ndim == 2:
        vocab_size = logits.shape[-1]
        labels_onehot = jax.nn.one_hot(labels, vocab_size)
    else:
        raise ValueError(f"Incompatible shapes: logits {logits.shape}, labels {labels.shape}")
        
    # Compute cross entropy
    loss = -jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply mask if provided, otherwise create from non-zero labels
    if mask is None:
        mask = (labels != 0).astype(jnp.float32)
    
    # Compute masked mean
    loss = jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    return loss


def kl_divergence_loss(student_logits: jnp.ndarray, teacher_logits: jnp.ndarray, 
                        temperature: float = 1.0) -> jnp.ndarray:
    """
    Compute KL divergence loss for knowledge distillation.
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        temperature: Temperature for softening the distributions
        
    Returns:
        KL divergence loss
    """
    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    
    # Compute softmax distributions
    student_probs = jax.nn.softmax(student_logits, axis=-1)
    teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
    
    # KL divergence: p * log(p/q)
    kl_div = teacher_probs * (jnp.log(teacher_probs + 1e-10) - jnp.log(student_probs + 1e-10))
    
    # Sum over the vocabulary dimension and take mean over batch and sequence
    loss = jnp.mean(jnp.sum(kl_div, axis=-1))
    
    # Scale by temperature^2 as in the original distillation paper
    return loss * (temperature ** 2)


def tot_guided_loss(logits: jnp.ndarray, guided_logits: jnp.ndarray, 
                   labels: jnp.ndarray, alpha: float = 0.5) -> jnp.ndarray:
    """
    Tree of Thoughts guided loss - combines standard CE with guided logits.
    
    Args:
        logits: Model output logits
        guided_logits: Logits from ToT guidance
        labels: Target token IDs
        alpha: Weight for combining the losses
        
    Returns:
        Combined loss
    """
    # Standard cross entropy with original logits
    standard_loss = cross_entropy_loss(logits, labels)
    
    # Loss with guided logits
    guided_loss = cross_entropy_loss(guided_logits, labels)
    
    # Combine with weighting
    return alpha * standard_loss + (1.0 - alpha) * guided_loss


def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
        
    Returns:
        Dictionary of metrics
    """
    # Create mask for valid positions (non-padding)
    mask = (labels != 0).astype(jnp.float32)
    
    # Get predictions
    predictions = jnp.argmax(logits, axis=-1)
    
    # Calculate accuracy
    correct = (predictions == labels) * mask
    accuracy = jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)
    
    # Calculate perplexity
    vocab_size = logits.shape[-1]
    labels_onehot = jax.nn.one_hot(labels, vocab_size)
    log_probs = jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1)
    masked_log_probs = log_probs * mask
    perplexity = jnp.exp(-jnp.sum(masked_log_probs) / jnp.maximum(jnp.sum(mask), 1.0))
    
    # Calculate loss
    loss = cross_entropy_loss(logits, labels, mask)
    
    return {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "loss": loss
    }
