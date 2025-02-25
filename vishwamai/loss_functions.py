"""
Loss functions and evaluation metrics for VishwamAI models.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional, Union
import optax


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
        mask = (labels > 0).astype(jnp.float32)
    
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
    mask = (labels > 0).astype(jnp.float32)
    
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


def compute_loss(outputs: Dict, batch: Dict) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute standard cross-entropy loss for language modeling.
    
    Args:
        outputs: Model output dictionary with logits
        batch: Input batch with labels
    
    Returns:
        loss: The computed loss value
        metrics: Dictionary with additional metrics
    """
    logits = outputs.get("logits")
    labels = batch.get("labels")
    
    if logits is None or labels is None:
        return jnp.array(0.0), {"accuracy": 0.0}
    
    # Apply label smoothing if specified in batch
    label_smoothing = batch.get("label_smoothing", 0.0)
    
    # Compute cross-entropy loss
    if label_smoothing > 0:
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, labels, label_smoothing=label_smoothing
        )
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    
    # Optional masking
    if "loss_mask" in batch:
        loss = loss * batch["loss_mask"]
        normalizer = batch["loss_mask"].sum()
    else:
        normalizer = jnp.prod(jnp.array(labels.shape))
    
    loss = loss.sum() / (normalizer + 1e-8)
    
    # Compute accuracy
    accuracy = jnp.mean(
        (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
    )
    
    return loss, {"accuracy": accuracy}


def compute_weighted_cross_entropy(
    logits: jnp.ndarray, 
    targets: jnp.ndarray, 
    weights: Optional[jnp.ndarray] = None,
    temperature: float = 1.0
) -> jnp.ndarray:
    """
    Compute weighted cross-entropy loss, useful for distillation and other tasks.
    
    Args:
        logits: Predicted logits
        targets: Target probabilities (already softmaxed)
        weights: Optional sample weights
        temperature: Temperature for softmax scaling
    
    Returns:
        loss: The computed weighted cross-entropy loss
    """
    logits = logits / temperature
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(targets * log_probs, axis=-1)
    
    if weights is not None:
        loss = loss * weights
        normalizer = weights.sum()
    else:
        normalizer = jnp.prod(jnp.array(targets.shape[:-1]))
    
    return loss.sum() / (normalizer + 1e-8)


def compute_contrastive_loss(
    embeddings: jnp.ndarray, 
    labels: jnp.ndarray, 
    temperature: float = 0.07
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute contrastive loss (InfoNCE) for representation learning.
    
    Args:
        embeddings: Normalized embeddings
        labels: Integer labels for determining positive pairs
        temperature: Temperature parameter
    
    Returns:
        loss: Contrastive loss value
        metrics: Dictionary with additional metrics
    """
    # Compute similarity matrix
    similarity = jnp.matmul(embeddings, embeddings.transpose())
    similarity = similarity / temperature
    
    # Create label mask (1 for same-class pairs)
    label_mask = jnp.equal(labels[:, None], labels[None, :])
    
    # Remove self-contrast
    identity_mask = jnp.eye(embeddings.shape[0], dtype=jnp.bool_)
    label_mask = label_mask & ~identity_mask
    
    # For each row, compute the contrastive loss
    row_norm = jax.nn.log_softmax(similarity, axis=1)
    col_norm = jax.nn.log_softmax(similarity, axis=0)
    
    # Compute the positive loss term
    pos_label_mask = label_mask.astype(jnp.float32)
    num_positives = pos_label_mask.sum(axis=1)
    
    # Average over positive examples when they exist
    row_loss = -jnp.sum(pos_label_mask * row_norm, axis=1) / jnp.maximum(num_positives, 1.0)
    col_loss = -jnp.sum(pos_label_mask * col_norm, axis=0) / jnp.maximum(num_positives, 1.0)
    
    # Average loss
    loss = (row_loss.mean() + col_loss.mean()) / 2.0
    
    # Compute metrics
    metrics = {
        "contrastive_loss": loss,
        "avg_similarity": jnp.mean(similarity),
        "avg_positive_similarity": (jnp.sum(similarity * label_mask.astype(jnp.float32)) / 
                                   jnp.maximum(jnp.sum(label_mask), 1.0))
    }
    
    return loss, metrics


def compute_moe_load_balancing_loss(
    dispatch_weights: jnp.ndarray,
    expert_indices: jnp.ndarray,
    num_experts: int
) -> jnp.ndarray:
    """
    Compute load balancing loss for Mixture of Experts models.
    
    Args:
        dispatch_weights: The weights used to dispatch to experts [batch, num_selected]
        expert_indices: The indices of selected experts [batch, num_selected]
        num_experts: Total number of experts
        
    Returns:
        load_balancing_loss: The computed load balancing loss
    """
    # Compute fraction of tokens routed to each expert
    expert_mask = jax.nn.one_hot(expert_indices, num_experts)  # [batch, num_selected, num_experts]
    expert_weights = dispatch_weights[..., None] * expert_mask  # Apply weights to mask
    expert_prop = expert_weights.sum(axis=(0, 1)) / expert_weights.sum()  # Normalize
    
    # Compute load balancing loss
    target_prop = jnp.ones_like(expert_prop) / num_experts
    load_balancing_loss = jnp.sum((expert_prop - target_prop) ** 2) * num_experts
    
    return load_balancing_loss


def compute_tot_loss(
    outputs: Dict,
    batch: Dict,
    base_loss_weight: float = 0.8,
    tot_weight: float = 0.2
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute combined loss for Tree of Thoughts training.
    
    Args:
        outputs: Model outputs including ToT-related outputs
        batch: Input batch
        base_loss_weight: Weight for the base model loss
        tot_weight: Weight for the ToT-specific loss
        
    Returns:
        combined_loss: The weighted combination of losses
        metrics: Dictionary with detailed loss metrics
    """
    # Compute standard loss
    base_loss, base_metrics = compute_loss(outputs, batch)
    
    # Get ToT-specific loss if available
    tot_loss = outputs.get("tot_loss", jnp.array(0.0))
    tot_accuracy = outputs.get("tot_accuracy", jnp.array(0.0))
    
    # Combine losses
    combined_loss = base_loss_weight * base_loss + tot_weight * tot_loss
    
    # Combine metrics
    metrics = {
        **base_metrics,
        "tot_loss": tot_loss,
        "tot_accuracy": tot_accuracy,
        "combined_loss": combined_loss
    }
    
    return combined_loss, metrics
