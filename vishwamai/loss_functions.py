"""
Loss functions and evaluation metrics for VishwamAI models.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional, Union
import optax

def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    error_weights: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute cross-entropy loss with dynamic label smoothing and error weighting.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
        mask: Mask where 1 indicates valid positions [batch, seq_len]
        label_smoothing: Smoothing factor (0 to 1)
        error_weights: Optional weights to emphasize high-error regions [batch, seq_len]
        
    Returns:
        Loss value
    """
    if logits.ndim == 3 and labels.ndim == 2:
        vocab_size = logits.shape[-1]
        labels_onehot = jax.nn.one_hot(labels, vocab_size)
    else:
        raise ValueError(f"Incompatible shapes: logits {logits.shape}, labels {labels.shape}")
    
    # Apply label smoothing
    if label_smoothing > 0:
        smoothed_labels = (1 - label_smoothing) * labels_onehot + label_smoothing / vocab_size
        loss = -jnp.sum(smoothed_labels * jax.nn.log_softmax(logits), axis=-1)
    else:
        loss = -jnp.sum(labels_onehot * jax.nn.log_softmax(logits), axis=-1)
    
    # Apply mask
    if mask is None:
        mask = (labels > 0).astype(jnp.float32)
    
    # Apply error weights if provided
    if error_weights is not None:
        loss = loss * error_weights
    
    # Compute masked mean
    normalizer = jnp.maximum(jnp.sum(mask), 1.0)
    loss = jnp.sum(loss * mask) / normalizer
    return loss

def kl_divergence_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    temperature: float = 1.0,
    step: Optional[int] = None,
    max_steps: Optional[int] = None,
    error_rate: Optional[float] = None
) -> jnp.ndarray:
    """
    Compute KL divergence loss with adaptive temperature scaling.
    
    Args:
        student_logits: Student model logits [batch, seq_len, vocab_size]
        teacher_logits: Teacher model logits [batch, seq_len, vocab_size]
        temperature: Base temperature for softening
        step: Current training step (for adaptive scaling)
        max_steps: Total training steps (for adaptive scaling)
        error_rate: Error rate to adjust temperature dynamically
        
    Returns:
        KL divergence loss
    """
    # Adaptive temperature scaling
    effective_temperature = temperature
    if step is not None and max_steps is not None:
        progress = step / max_steps
        effective_temperature *= (1 - 0.5 * progress)  # Decrease temperature over time
    if error_rate is not None:
        effective_temperature *= (1 + jnp.tanh(error_rate))  # Increase with high errors
    
    student_logits = student_logits / effective_temperature
    teacher_logits = teacher_logits / effective_temperature
    
    student_probs = jax.nn.softmax(student_logits, axis=-1)
    teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
    
    # KL divergence with numerical stability
    kl_div = teacher_probs * (jnp.log(teacher_probs + 1e-10) - jnp.log(student_probs + 1e-10))
    loss = jnp.mean(jnp.sum(kl_div, axis=-1)) * (effective_temperature ** 2)
    
    return loss

def tot_guided_loss(
    logits: jnp.ndarray,
    guided_logits: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 0.5,
    context_weight: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Enhanced Tree of Thoughts guided loss with contextual blending.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        guided_logits: ToT-guided logits [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
        alpha: Base weight for combining losses
        context_weight: Optional context-aware weights [batch, seq_len]
        
    Returns:
        Combined loss
    """
    standard_loss = cross_entropy_loss(logits, labels)
    guided_loss = cross_entropy_loss(guided_logits, labels)
    
    # Dynamic alpha based on context (if provided)
    effective_alpha = alpha
    if context_weight is not None:
        effective_alpha = alpha * context_weight + (1 - alpha) * (1 - context_weight)
        combined_loss = effective_alpha * standard_loss + (1 - effective_alpha) * guided_loss
        return jnp.mean(combined_loss)
    
    return alpha * standard_loss + (1.0 - alpha) * guided_loss

def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    corrected_logits: Optional[jnp.ndarray] = None
) -> Dict[str, float]:
    """
    Compute advanced evaluation metrics including correction impact.
    
    Args:
        logits: Model output logits [batch, seq_len, vocab_size]
        labels: Target token IDs [batch, seq_len]
        corrected_logits: Optional corrected logits for comparison
        
    Returns:
        Dictionary of metrics
    """
    mask = (labels > 0).astype(jnp.float32)
    predictions = jnp.argmax(logits, axis=-1)
    
    # Accuracy
    correct = (predictions == labels) * mask
    accuracy = float(jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0))
    
    # Perplexity
    vocab_size = logits.shape[-1]
    log_probs = jnp.sum(jax.nn.one_hot(labels, vocab_size) * jax.nn.log_softmax(logits), axis=-1)
    perplexity = float(jnp.exp(-jnp.sum(log_probs * mask) / jnp.maximum(jnp.sum(mask), 1.0)))
    
    # Loss
    loss = float(cross_entropy_loss(logits, labels, mask))
    
    metrics = {
        "accuracy": accuracy,
        "perplexity": perplexity,
        "loss": loss
    }
    
    # Correction metrics if provided
    if corrected_logits is not None:
        corrected_preds = jnp.argmax(corrected_logits, axis=-1)
        corrected_correct = (corrected_preds == labels) * mask
        corrected_accuracy = float(jnp.sum(corrected_correct) / jnp.maximum(jnp.sum(mask), 1.0))
        improvement = corrected_accuracy - accuracy
        metrics.update({
            "corrected_accuracy": corrected_accuracy,
            "improvement": improvement
        })
    
    return metrics

def compute_loss(
    outputs: Dict,
    batch: Dict,
    error_weights: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute standard cross-entropy loss with error weighting support.
    
    Args:
        outputs: Model output dictionary with logits
        batch: Input batch with labels
        error_weights: Optional weights from error correction
        
    Returns:
        loss: The computed loss value
        metrics: Dictionary with additional metrics
    """
    logits = outputs.get("logits")
    labels = batch.get("labels")
    
    if logits is None or labels is None:
        return jnp.array(0.0), {"accuracy": 0.0}
    
    label_smoothing = batch.get("label_smoothing", 0.0)
    loss = cross_entropy_loss(logits, labels, batch.get("loss_mask"), label_smoothing, error_weights)
    
    accuracy = float(jnp.mean((jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)))
    metrics = {"accuracy": accuracy}
    
    return loss, metrics

def compute_weighted_cross_entropy(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    temperature: float = 1.0
) -> jnp.ndarray:
    """
    Compute weighted cross-entropy loss with temperature scaling.
    
    Args:
        logits: Predicted logits [batch, seq_len, vocab_size]
        targets: Target probabilities [batch, seq_len, vocab_size]
        weights: Optional sample weights [batch, seq_len]
        temperature: Temperature for softmax scaling
    
    Returns:
        Loss value
    """
    logits = logits / temperature
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(targets * log_probs, axis=-1)
    
    if weights is not None:
        loss = loss * weights
    normalizer = weights.sum() if weights is not None else jnp.prod(jnp.array(targets.shape[:-1]))
    
    return loss.sum() / (normalizer + 1e-8)

def compute_contrastive_loss(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 0.07,
    num_negatives: int = 128
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute contrastive loss with multi-positive pairs and negative sampling.
    
    Args:
        embeddings: Normalized embeddings [batch, embedding_size]
        labels: Integer labels [batch]
        temperature: Temperature parameter
        num_negatives: Number of negative samples to consider
        
    Returns:
        loss: Contrastive loss value
        metrics: Dictionary with additional metrics
    """
    batch_size = embeddings.shape[0]
    similarity = jnp.matmul(embeddings, embeddings.T) / temperature
    
    # Positive pairs mask
    label_mask = jnp.equal(labels[:, None], labels[None, :]) & ~jnp.eye(batch_size, dtype=jnp.bool_)
    
    # Negative sampling (randomly select num_negatives)
    rng = jax.random.PRNGKey(0)  # For reproducibility in example; use proper RNG in practice
    neg_indices = jax.random.choice(rng, batch_size, (batch_size, num_negatives), replace=False)
    neg_mask = jnp.zeros((batch_size, batch_size), dtype=jnp.bool_)
    neg_mask = neg_mask.at[jnp.arange(batch_size)[:, None], neg_indices].set(True)
    
    # Compute loss
    exp_sim = jnp.exp(similarity)
    pos_sum = jnp.sum(exp_sim * label_mask.astype(jnp.float32), axis=-1)
    neg_sum = jnp.sum(exp_sim * neg_mask.astype(jnp.float32), axis=-1)
    loss = -jnp.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))
    loss = jnp.mean(loss)
    
    # Metrics
    metrics = {
        "contrastive_loss": float(loss),
        "avg_positive_similarity": float(jnp.mean(similarity * label_mask.astype(jnp.float32))),
        "avg_negative_similarity": float(jnp.mean(similarity * neg_mask.astype(jnp.float32)))
    }
    
    return loss, metrics

def compute_moe_load_balancing_loss(
    dispatch_weights: jnp.ndarray,
    expert_indices: jnp.ndarray,
    num_experts: int,
    fairness_alpha: float = 0.1
) -> jnp.ndarray:
    """
    Compute enhanced MoE load balancing loss with fairness regularization.
    
    Args:
        dispatch_weights: Weights used to dispatch to experts [batch, num_selected]
        expert_indices: Indices of selected experts [batch, num_selected]
        num_experts: Total number of experts
        fairness_alpha: Weight for fairness term
        
    Returns:
        Load balancing loss
    """
    expert_mask = jax.nn.one_hot(expert_indices, num_experts)
    expert_weights = dispatch_weights[..., None] * expert_mask
    expert_prop = expert_weights.sum(axis=(0, 1)) / expert_weights.sum()
    
    # Standard load balancing
    target_prop = jnp.ones_like(expert_prop) / num_experts
    balance_loss = jnp.sum((expert_prop - target_prop) ** 2) * num_experts
    
    # Fairness regularization (minimize variance in expert usage)
    fairness_loss = jnp.var(expert_prop)
    
    return balance_loss + fairness_alpha * fairness_loss

def compute_tot_loss(
    outputs: Dict,
    batch: Dict,
    base_loss_weight: float = 0.8,
    tot_weight: float = 0.2,
    adaptive_weighting: bool = True
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute advanced ToT loss with adaptive weighting.
    
    Args:
        outputs: Model outputs including ToT-related outputs
        batch: Input batch
        base_loss_weight: Base weight for standard loss
        tot_weight: Base weight for ToT-specific loss
        adaptive_weighting: Adjust weights based on ToT confidence
        
    Returns:
        combined_loss: Weighted combination of losses
        metrics: Dictionary with detailed metrics
    """
    base_loss, base_metrics = compute_loss(outputs, batch)
    tot_loss = outputs.get("tot_loss", jnp.array(0.0))
    tot_accuracy = outputs.get("tot_accuracy", jnp.array(0.0))
    tot_score = outputs.get("tot_outputs", {}).get("score", 0.0) if "tot_outputs" in outputs else 0.0
    
    # Adaptive weighting based on ToT confidence
    if adaptive_weighting and tot_score > 0:
        confidence = jnp.tanh(tot_score)  # Normalize ToT score to [0, 1]
        effective_tot_weight = tot_weight * confidence
        effective_base_weight = base_loss_weight * (1 - confidence)
    else:
        effective_tot_weight = tot_weight
        effective_base_weight = base_loss_weight
    
    combined_loss = effective_base_weight * base_loss + effective_tot_weight * tot_loss
    
    metrics = {
        **base_metrics,
        "tot_loss": float(tot_loss),
        "tot_accuracy": float(tot_accuracy),
        "combined_loss": float(combined_loss),
        "tot_weight": float(effective_tot_weight)
    }
    
    return combined_loss, metrics

def compute_composite_loss(
    outputs: Dict,
    batch: Dict,
    teacher_logits: Optional[jnp.ndarray] = None,
    weights: Dict[str, float] = None
) -> Tuple[jnp.ndarray, Dict]:
    """
    Compute composite loss combining CE, KD, ToT, and MoE objectives.
    
    Args:
        outputs: Model outputs with logits, tot_outputs, etc.
        batch: Input batch with labels
        teacher_logits: Optional teacher logits for KD
        weights: Dictionary of weights for each loss component (ce, kd, tot, moe)
        
    Returns:
        combined_loss: Total composite loss
        metrics: Detailed metrics for each component
    """
    weights = weights or {'ce': 0.4, 'kd': 0.3, 'tot': 0.2, 'moe': 0.1}
    
    # CE Loss
    ce_loss, ce_metrics = compute_loss(outputs, batch)
    
    # KD Loss
    kd_loss = 0.0
    if teacher_logits is not None and 'logits' in outputs:
        kd_loss = kl_divergence_loss(outputs['logits'], teacher_logits)
    
    # ToT Loss
    tot_loss, tot_metrics = compute_tot_loss(outputs, batch, base_loss_weight=0.0, tot_weight=1.0)
    
    # MoE Load Balancing Loss
    moe_loss = 0.0
    if 'dispatch_weights' in outputs and 'expert_indices' in outputs:
        moe_loss = compute_moe_load_balancing_loss(
            outputs['dispatch_weights'],
            outputs['expert_indices'],
            num_experts=outputs.get('num_experts', 8)
        )
    
    # Combine Losses
    combined_loss = (
        weights['ce'] * ce_loss +
        weights['kd'] * kd_loss +
        weights['tot'] * tot_loss +
        weights['moe'] * moe_loss
    )
    
    # Metrics
    metrics = {
        **ce_metrics,
        "kd_loss": float(kd_loss),
        **tot_metrics,
        "moe_loss": float(moe_loss),
        "combined_loss": float(combined_loss)
    }
    
    return combined_loss, metrics