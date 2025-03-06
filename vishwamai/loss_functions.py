"""Loss functions for model training and distillation with TPU optimizations."""
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2,))
def kl_divergence_loss(student_logits: jnp.ndarray, teacher_logits: jnp.ndarray, temperature: float = 1.0) -> jnp.ndarray:
    """Compute KL divergence loss between student and teacher logits."""
    scaled_student = student_logits / temperature
    scaled_teacher = teacher_logits / temperature
    
    student_log_probs = jax.nn.log_softmax(scaled_student)
    teacher_probs = jax.nn.softmax(scaled_teacher)
    
    return jnp.mean(jnp.sum(teacher_probs * (jax.nn.log_softmax(scaled_teacher) - student_log_probs), axis=-1))

@jax.jit
def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray, ignore_index: int = -100) -> jnp.ndarray:
    """Compute cross entropy loss with TPU optimization."""
    valid_mask = (labels != ignore_index).astype(jnp.float32)
    num_valid = jnp.maximum(jnp.sum(valid_mask), 1.0)
    
    log_probs = jax.nn.log_softmax(logits)
    labels_onehot = jax.nn.one_hot(labels, num_classes=logits.shape[-1])
    per_example_loss = -jnp.sum(labels_onehot * log_probs, axis=-1)
    
    return jnp.sum(per_example_loss * valid_mask) / num_valid

@jax.jit
def feature_matching_loss(student_features: jnp.ndarray, teacher_features: jnp.ndarray) -> jnp.ndarray:
    """Compute feature matching loss between student and teacher intermediate layers."""
    # Normalize features for better matching
    student_norm = student_features / (jnp.linalg.norm(student_features, axis=-1, keepdims=True) + 1e-6)
    teacher_norm = teacher_features / (jnp.linalg.norm(teacher_features, axis=-1, keepdims=True) + 1e-6)
    
    return jnp.mean(jnp.square(student_norm - teacher_norm))

@jax.jit
def attention_matching_loss(student_attention: jnp.ndarray, teacher_attention: jnp.ndarray) -> jnp.ndarray:
    """Compute attention matching loss for knowledge transfer."""
    # Normalize attention matrices
    student_norm = student_attention / (jnp.sum(student_attention, axis=-1, keepdims=True) + 1e-6)
    teacher_norm = teacher_attention / (jnp.sum(teacher_attention, axis=-1, keepdims=True) + 1e-6)
    
    return jnp.mean(jnp.square(student_norm - teacher_norm))

@partial(jax.jit, static_argnums=(2,))
def compute_distillation_loss(
    student_outputs: dict,
    teacher_outputs: dict,
    layer_mapping: dict,
    temperature: float = 1.0,
    alpha_ce: float = 0.1,
    alpha_kd: float = 0.9,
    alpha_features: float = 0.1,
    alpha_attention: float = 0.1,
):
    """Compute combined distillation loss with feature and attention matching for Qwen models."""
    # KL divergence on logits
    kd_loss = kl_divergence_loss(
        student_outputs['logits'],
        teacher_outputs['logits'],
        temperature
    )
    
    # Cross entropy with labels if provided
    ce_loss = 0.0
    if 'labels' in student_outputs:
        ce_loss = cross_entropy_loss(student_outputs['logits'], student_outputs['labels'])
    
    # Feature matching loss across mapped layers
    feature_losses = []
    for student_layer, teacher_layer in layer_mapping.items():
        if f'layer_{student_layer}' in student_outputs and f'layer_{teacher_layer}' in teacher_outputs:
            student_feat = student_outputs[f'layer_{student_layer}']
            teacher_feat = teacher_outputs[f'layer_{teacher_layer}']
            feature_losses.append(feature_matching_loss(student_feat, teacher_feat))
    
    feature_loss = jnp.mean(jnp.array(feature_losses)) if feature_losses else 0.0
    
    # Attention matching loss if available
    attention_loss = 0.0
    if 'attentions' in student_outputs and 'attentions' in teacher_outputs:
        attention_loss = attention_matching_loss(
            student_outputs['attentions'],
            teacher_outputs['attentions']
        )
    
    # Combine all losses
    total_loss = (
        alpha_kd * kd_loss +
        alpha_ce * ce_loss +
        alpha_features * feature_loss +
        alpha_attention * attention_loss
    )
    
    return total_loss, {
        'kd_loss': kd_loss,
        'ce_loss': ce_loss,
        'feature_loss': feature_loss,
        'attention_loss': attention_loss,
        'total_loss': total_loss
    }
