"""
Loss functions and evaluation metrics optimized for TPU with advanced batching and sharding.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, Optional, Union,Callable
import optax
import logging
from functools import partial

logger = logging.getLogger(__name__)

@partial(jax.jit, static_argnums=(3, 4))
def cross_entropy_loss(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    label_smoothing: float = 0.0,
    use_bfloat16: bool = True,
    error_weights: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """TPU-optimized cross-entropy loss."""
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    logits = logits.astype(dtype)
    
    if logits.ndim == 3 and labels.ndim == 2:
        vocab_size = logits.shape[-1]
        labels_onehot = jax.nn.one_hot(labels, vocab_size, dtype=dtype)
    else:
        raise ValueError(f"Incompatible shapes: logits {logits.shape}, labels {labels.shape}")
    
    if label_smoothing > 0:
        smoothed_labels = ((1 - label_smoothing) * labels_onehot + 
                         label_smoothing / vocab_size).astype(dtype)
        loss = -jnp.sum(smoothed_labels * jax.nn.log_softmax(logits, dtype=dtype), axis=-1)
    else:
        loss = -jnp.sum(labels_onehot * jax.nn.log_softmax(logits, dtype=dtype), axis=-1)
    
    if mask is None:
        mask = (labels > 0).astype(dtype)
    
    if error_weights is not None:
        loss = loss * error_weights.astype(dtype)
    
    normalizer = jnp.maximum(jnp.sum(mask), 1e-8)
    return jnp.sum(loss * mask) / normalizer

@partial(jax.jit, static_argnums=(2, 3, 4))
def kl_divergence_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    temperature: float = 1.0,
    use_bfloat16: bool = True,
    use_pmap: bool = True
) -> jnp.ndarray:
    """TPU-optimized KL divergence loss."""
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    student_logits = student_logits.astype(dtype)
    teacher_logits = teacher_logits.astype(dtype)
    
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    
    student_probs = jax.nn.softmax(student_logits, dtype=dtype)
    teacher_probs = jax.nn.softmax(teacher_logits, dtype=dtype)
    
    kl_div = teacher_probs * (
        jnp.log(teacher_probs + 1e-10) - 
        jnp.log(student_probs + 1e-10)
    )
    
    if use_pmap:
        kl_div = jax.lax.pmean(kl_div, axis_name='batch')
    
    loss = jnp.mean(jnp.sum(kl_div, axis=-1)) * (temperature ** 2)
    return loss

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3, 4, 5))
def tot_guided_loss(
    logits: jnp.ndarray,
    guided_logits: jnp.ndarray,
    labels: jnp.ndarray,
    alpha: float = 0.5,
    use_bfloat16: bool = True,
    use_context_weights: bool = True,
    context_weight: Optional[jnp.ndarray] = None,
    tot_score: Optional[float] = None
) -> jnp.ndarray:
    """TPU-parallel Tree of Thoughts guided loss."""
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    logits = logits.astype(dtype)
    guided_logits = guided_logits.astype(dtype)
    
    standard_loss = cross_entropy_loss(logits, labels, use_bfloat16=use_bfloat16)
    guided_loss = cross_entropy_loss(guided_logits, labels, use_bfloat16=use_bfloat16)
    
    effective_alpha = jnp.array(alpha, dtype=dtype)
    if tot_score is not None:
        confidence = jnp.tanh(tot_score)
        effective_alpha = alpha + (1 - alpha) * confidence
    
    if use_context_weights and context_weight is not None:
        context_weight = context_weight.astype(dtype)
        effective_alpha = (effective_alpha * context_weight + 
                         (1 - effective_alpha) * (1 - context_weight))
        combined_loss = effective_alpha * standard_loss + (1 - effective_alpha) * guided_loss
        return jax.lax.pmean(jnp.mean(combined_loss), axis_name='batch')
    
    return jax.lax.pmean(
        effective_alpha * standard_loss + (1.0 - effective_alpha) * guided_loss,
        axis_name='batch'
    )

@partial(jax.jit, static_argnums=(4,))
def compute_metrics(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    corrected_logits: Optional[jnp.ndarray] = None,
    decoded_logits: Optional[str] = None,
    use_bfloat16: bool = True,
    decoded_labels: Optional[str] = None
) -> Dict[str, float]:
    """TPU-optimized metrics computation."""
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    logits = logits.astype(dtype)
    mask = (labels > 0).astype(dtype)
    
    predictions = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean((predictions == labels) * mask) / jnp.maximum(jnp.sum(mask), 1e-8)
    
    vocab_size = logits.shape[-1]
    labels_onehot = jax.nn.one_hot(labels, vocab_size, dtype=dtype)
    log_probs = jnp.sum(labels_onehot * jax.nn.log_softmax(logits, dtype=dtype), axis=-1)
    perplexity = jnp.exp(-jnp.sum(log_probs * mask) / jnp.maximum(jnp.sum(mask), 1e-8))
    
    metrics = {
        "accuracy": float(accuracy),
        "perplexity": float(perplexity),
        "loss": float(cross_entropy_loss(logits, labels, mask, use_bfloat16=use_bfloat16))
    }
    
    if corrected_logits is not None:
        corrected_logits = corrected_logits.astype(dtype)
        corrected_preds = jnp.argmax(corrected_logits, axis=-1)
        corrected_accuracy = float(
            jnp.mean((corrected_preds == labels) * mask) / 
            jnp.maximum(jnp.sum(mask), 1e-8)
        )
        metrics.update({
            "corrected_accuracy": corrected_accuracy,
            "improvement": corrected_accuracy - float(accuracy)
        })
    
    # GSM8K-specific evaluation
    if decoded_logits and decoded_labels:
        metrics.update(_compute_gsm8k_metrics(decoded_logits, decoded_labels))
    
    return metrics

@jax.jit
def _compute_gsm8k_metrics(decoded_logits: str, decoded_labels: str) -> Dict[str, float]:
    """TPU-compatible GSM8K metrics computation."""
    pred_steps = [s.strip() for s in decoded_logits.split('\n') if s.strip().startswith('Step:')]
    target_steps = [s.strip() for s in decoded_labels.split('\n') if s.strip().startswith('Step:')]
    
    correct_steps = sum(1 for p, t in zip(pred_steps, target_steps) if p == t)
    total_steps = max(len(pred_steps), len(target_steps), 1)
    
    pred_answer = (decoded_logits.split('####')[-1].strip() if '####' in decoded_logits 
                  else pred_steps[-1].split()[-1] if pred_steps else "")
    target_answer = (decoded_labels.split('####')[-1].strip() if '####' in decoded_labels 
                    else target_steps[-1].split()[-1] if target_steps else "")
    
    exact_match = (pred_answer == target_answer and 
                  len(pred_steps) == len(target_steps) and 
                  all(p == t for p, t in zip(pred_steps, target_steps)))
    
    return {
        "step_accuracy": float(correct_steps / total_steps),
        "exact_match": float(exact_match),
        "answer_match": float(pred_answer == target_answer)
    }

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(2, 3))
def compute_contrastive_loss(
    embeddings: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 0.07,
    use_bfloat16: bool = True,
    num_negatives: int = 128
) -> Tuple[jnp.ndarray, Dict]:
    """TPU-parallel contrastive loss computation."""
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    embeddings = embeddings.astype(dtype)
    
    batch_size = embeddings.shape[0]
    similarity = jnp.matmul(embeddings, embeddings.T) / temperature
    
    # Create label mask for positives
    label_mask = jnp.equal(labels[:, None], labels[None, :])
    label_mask = label_mask & ~jnp.eye(batch_size, dtype=bool)
    
    # Generate negative mask with TPU optimization
    rng = jax.random.PRNGKey(0)  # Replace with proper RNG in practice
    neg_indices = jax.random.choice(
        rng, batch_size, 
        shape=(batch_size, num_negatives),
        replace=False
    )
    neg_mask = jnp.zeros((batch_size, batch_size), dtype=bool)
    neg_mask = neg_mask.at[jnp.arange(batch_size)[:, None], neg_indices].set(True)
    
    # Compute loss with TPU optimization
    exp_sim = jnp.exp(similarity)
    pos_sum = jnp.sum(exp_sim * label_mask.astype(dtype), axis=-1)
    neg_sum = jnp.sum(exp_sim * neg_mask.astype(dtype), axis=-1)
    
    loss = -jnp.log((pos_sum + 1e-8) / (pos_sum + neg_sum + 1e-8))
    loss = jax.lax.pmean(jnp.mean(loss), axis_name='batch')
    
    metrics = {
        "contrastive_loss": float(loss),
        "avg_positive_similarity": float(
            jnp.mean(similarity * label_mask.astype(dtype))
        ),
        "avg_negative_similarity": float(
            jnp.mean(similarity * neg_mask.astype(dtype))
        )
    }
    
    return loss, metrics

@partial(jax.jit, static_argnums=(2, 3, 4))
def compute_moe_load_balancing_loss(
    dispatch_weights: jnp.ndarray,
    expert_indices: jnp.ndarray,
    num_experts: int,
    fairness_alpha: float = 0.1,
    use_bfloat16: bool = True
) -> jnp.ndarray:
    """TPU-optimized MoE load balancing loss."""
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    dispatch_weights = dispatch_weights.astype(dtype)
    
    expert_mask = jax.nn.one_hot(expert_indices, num_experts, dtype=dtype)
    expert_weights = dispatch_weights[..., None] * expert_mask
    expert_prop = expert_weights.sum(axis=(0, 1)) / (expert_weights.sum() + 1e-8)
    
    target_prop = jnp.ones_like(expert_prop, dtype=dtype) / num_experts
    balance_loss = jnp.sum((expert_prop - target_prop) ** 2) * num_experts
    fairness_loss = jnp.var(expert_prop)
    
    return balance_loss + fairness_alpha * fairness_loss

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3, 4, 5))
def compute_composite_loss(
    outputs: Dict,
    batch: Dict,
    teacher_logits: Optional[jnp.ndarray] = None,
    weights: Optional[Dict[str, float]] = None,
    use_bfloat16: bool = True,
    use_tpu_optimizations: bool = True
) -> Tuple[jnp.ndarray, Dict]:
    """TPU-parallel composite loss computation."""
    weights = weights or {'ce': 0.4, 'kd': 0.3, 'tot': 0.2, 'moe': 0.1}
    dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
    
    # Cross Entropy Loss
    ce_loss, ce_metrics = cross_entropy_loss(
        outputs['logits'],
        batch['labels'],
        batch.get('loss_mask'),
        use_bfloat16=use_bfloat16
    )
    
    # Knowledge Distillation Loss
    kd_loss = jnp.array(0.0, dtype=dtype)
    if teacher_logits is not None and 'logits' in outputs:
        kd_loss = kl_divergence_loss(
            outputs['logits'],
            teacher_logits,
            use_bfloat16=use_bfloat16,
            use_pmap=use_tpu_optimizations
        )
    
    # Tree of Thoughts Loss
    tot_loss = jnp.array(0.0, dtype=dtype)
    if 'tot_outputs' in outputs:
        tot_loss = tot_guided_loss(
            outputs['logits'],
            outputs['tot_outputs'].get('guided_logits', outputs['logits']),
            batch['labels'],
            use_bfloat16=use_bfloat16
        )
    
    # MoE Load Balancing Loss
    moe_loss = jnp.array(0.0, dtype=dtype)
    if 'dispatch_weights' in outputs and 'expert_indices' in outputs:
        moe_loss = compute_moe_load_balancing_loss(
            outputs['dispatch_weights'],
            outputs['expert_indices'],
            outputs.get('num_experts', 8),
            use_bfloat16=use_bfloat16
        )
    
    # Combine losses with TPU optimization
    combined_loss = jax.lax.pmean(
        (weights['ce'] * ce_loss +
         weights['kd'] * kd_loss +
         weights['tot'] * tot_loss +
         weights['moe'] * moe_loss),
        axis_name='batch'
    )
    
    metrics = {
        **ce_metrics,
        "kd_loss": float(kd_loss),
        "tot_loss": float(tot_loss),
        "moe_loss": float(moe_loss),
        "combined_loss": float(combined_loss)
    }
    
    return combined_loss, metrics

def create_loss_fn(
    model_fn: Callable,
    weights: Optional[Dict[str, float]] = None,
    use_bfloat16: bool = True,
    use_tpu_optimizations: bool = True
) -> Callable:
    """Create TPU-optimized training loss function."""
    @partial(jax.jit, static_argnums=(3, 4))
    def loss_fn(params, batch, rng, is_training=True):
        outputs = model_fn({'params': params}, batch, rngs={'dropout': rng})
        loss, metrics = compute_composite_loss(
            outputs,
            batch,
            use_bfloat16=use_bfloat16,
            use_tpu_optimizations=use_tpu_optimizations,
            weights=weights
        )
        return loss, (metrics, outputs)
    return loss_fn
