"""
Knowledge distillation implementation for VishwamAI transformer.
"""

import jax
import jax.numpy as jnp
from typing import Any, Dict, Tuple, Optional
from functools import partial
from .transformer import EnhancedTransformerModel, create_vishwamai_transformer, create_train_state

def compute_distillation_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    labels: jnp.ndarray,
    temperature: float = 2.0,
    alpha: float = 0.5,
    label_smoothing: float = 0.0
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute the distillation loss combining soft and hard targets.
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        labels: Ground truth labels
        temperature: Temperature for softening probability distributions
        alpha: Weight for balancing soft and hard losses
        label_smoothing: Label smoothing factor
    """
    # Compute soft targets from teacher
    soft_targets = jax.nn.softmax(teacher_logits / temperature)
    
    # Compute soft loss
    student_log_probs = jax.nn.log_softmax(student_logits / temperature)
    soft_loss = -jnp.sum(soft_targets * student_log_probs, axis=-1)
    soft_loss = soft_loss * (temperature ** 2)
    
    # Compute hard loss with label smoothing
    vocab_size = student_logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = label_smoothing / (vocab_size - 1)
    
    # Create smoothed targets
    hard_targets = jax.nn.one_hot(labels, vocab_size)
    hard_targets = hard_targets * confidence + low_confidence * (1.0 - hard_targets)
    
    hard_loss = -jnp.sum(hard_targets * jax.nn.log_softmax(student_logits), axis=-1)
    
    # Combine losses
    total_loss = (alpha * soft_loss) + ((1 - alpha) * hard_loss)
    
    metrics = {
        'soft_loss': jnp.mean(soft_loss),
        'hard_loss': jnp.mean(hard_loss),
        'total_loss': jnp.mean(total_loss)
    }
    
    return total_loss, metrics

@partial(jax.jit, static_argnums=(4,))
def distillation_train_step(
    student_state: Any,
    teacher_params: Any,
    batch: Dict[str, jnp.ndarray],
    dropout_rng: Any,
    temperature: float = 2.0,
    alpha: float = 0.5,
    label_smoothing: float = 0.0
) -> Tuple[Any, Dict[str, jnp.ndarray]]:
    """
    Single training step for knowledge distillation.
    
    Args:
        student_state: Training state of student model
        teacher_params: Parameters of teacher model
        batch: Batch of training data
        dropout_rng: PRNG key for dropout
        temperature: Temperature for distillation
        alpha: Weight for balancing soft and hard losses
        label_smoothing: Label smoothing factor
    """
    def loss_fn(params):
        # Get student predictions
        student_logits = student_state.apply_fn(
            {'params': params},
            batch['input_ids'],
            deterministic=False,
            rngs={'dropout': dropout_rng}
        )
        
        # Get teacher predictions (no dropout/training)
        teacher_logits = student_state.apply_fn(
            {'params': teacher_params},
            batch['input_ids'],
            deterministic=True
        )
        
        loss, metrics = compute_distillation_loss(
            student_logits,
            teacher_logits,
            batch['labels'],
            temperature=temperature,
            alpha=alpha,
            label_smoothing=label_smoothing
        )
        
        return loss, (metrics, student_logits)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (metrics, logits)), grads = grad_fn(student_state.params)
    
    # Update student parameters
    new_student_state = student_state.apply_gradients(grads=grads)
    
    metrics['learning_rate'] = new_student_state.opt_state.hyperparams['learning_rate']
    
    return new_student_state, metrics

def create_student_model(
    config: Dict[str, Any],
    teacher_config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Create a student model for distillation.
    
    Args:
        config: Configuration for student model
        teacher_config: Optional configuration of teacher model to derive student config
    """
    if teacher_config is not None:
        # Create smaller student model based on teacher architecture
        student_config = {
            'vocab_size': teacher_config['vocab_size'],
            'num_layers': teacher_config['num_layers'] // 2,  # Reduce layers
            'num_heads': max(teacher_config['num_heads'] // 2, 8),  # Reduce heads
            'head_dim': teacher_config['head_dim'],
            'hidden_dim': teacher_config['hidden_dim'] // 2,  # Reduce width
            'mlp_dim': teacher_config['mlp_dim'] // 2,
            'max_seq_len': teacher_config['max_seq_len'],
            'dropout_rate': config.get('dropout_rate', 0.1),
            'attention_dropout_rate': config.get('attention_dropout_rate', 0.1),
            'use_enhanced': True,
            'use_rotary': True,
            'use_flash_attn': True,
            'use_rms_norm': True,
            'dtype': config.get('dtype', 'bfloat16')
        }
    else:
        student_config = config
        
    return create_vishwamai_transformer(student_config)

def initialize_from_teacher(
    student_state: Any,
    teacher_state: Any,
    method: str = 'layer_random'
) -> Any:
    """
    Initialize student model using teacher model weights.
    
    Args:
        student_state: Training state of student model
        teacher_state: Training state of teacher model
        method: Initialization method ('layer_random' or 'layer_select')
    """
    if method == 'layer_random':
        # Randomly initialize student layers from teacher
        student_layers = len(student_state.params['transformer_blocks'])
        teacher_layers = len(teacher_state.params['transformer_blocks'])
        
        # Randomly select teacher layers to initialize from
        rng = jax.random.PRNGKey(0)
        selected_layers = jax.random.permutation(
            rng, teacher_layers
        )[:student_layers]
        
        # Copy selected layer parameters
        new_params = dict(student_state.params)
        for i, teacher_idx in enumerate(selected_layers):
            new_params[f'transformer_block_{i}'] = teacher_state.params[f'transformer_block_{teacher_idx}']
            
        return student_state.replace(params=new_params)
        
    elif method == 'layer_select':
        # Select evenly spaced layers from teacher
        student_layers = len(student_state.params['transformer_blocks'])
        teacher_layers = len(teacher_state.params['transformer_blocks'])
        
        # Calculate layer spacing
        spacing = teacher_layers // student_layers
        selected_layers = list(range(0, teacher_layers, spacing))[:student_layers]
        
        # Copy selected layer parameters
        new_params = dict(student_state.params)
        for i, teacher_idx in enumerate(selected_layers):
            new_params[f'transformer_block_{i}'] = teacher_state.params[f'transformer_block_{teacher_idx}']
            
        return student_state.replace(params=new_params)
        
    else:
        raise ValueError(f"Unknown initialization method: {method}")

def create_distillation_train_state(
    rng: Any,
    config: Dict[str, Any],
    learning_rate_schedule: Any,
    teacher_state: Optional[Any] = None
) -> Any:
    """
    Create training state for distillation, optionally initializing from teacher.
    
    Args:
        rng: PRNG key
        config: Model configuration
        learning_rate_schedule: Learning rate schedule function
        teacher_state: Optional teacher model state for initialization
    """
    # Create student model
    student_state = create_train_state(rng, config, learning_rate_schedule)
    
    # Initialize from teacher if provided
    if teacher_state is not None:
        student_state = initialize_from_teacher(
            student_state,
            teacher_state,
            method=config.get('init_method', 'layer_random')
        )
    
    return student_state
