"""Knowledge distillation from Gemma to VishwamAI."""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple
import optax
from .kernels.core.kernel import fp8_gemm_optimized

def create_student_config(teacher_config: Dict[str, Any], reduction_factor: float) -> EnhancedTransformerConfig:
    """Create optimal student config for TPU v3-8."""
    return EnhancedTransformerConfig(
        vocab_size=teacher_config["vocab_size"],
        hidden_size=max(512, int(teacher_config["hidden_size"] * reduction_factor)),
        num_attention_heads=max(8, int(teacher_config["num_attention_heads"] * reduction_factor)),
        num_hidden_layers=max(4, int(teacher_config["num_hidden_layers"] * reduction_factor)),
        intermediate_size=max(2048, int(teacher_config["intermediate_size"] * reduction_factor)),
        max_position_embeddings=teacher_config["max_position_embeddings"],
        dropout_rate=0.1,
        attention_dropout=0.1,
        use_flash_attention=True,
        use_fp8=True,
        use_parallel=True,
        block_size=128  # Optimal for TPU v3
    )

def compute_distillation_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    labels: Optional[jnp.ndarray] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    mask: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute distillation loss with TPU optimizations."""
    
    # Use TPU-optimized operations
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    student_logits_temp = student_logits / temperature
    student_log_probs = jax.nn.log_softmax(student_logits_temp, axis=-1)
    
    # Compute soft loss efficiently
    soft_loss = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
    soft_loss = soft_loss * (temperature ** 2)
    
    if mask is not None:
        soft_loss = soft_loss * mask
        
    metrics = {"soft_loss": jnp.mean(soft_loss)}
    
    # Compute hard loss if labels provided
    if labels is not None:
        hard_loss = optax.softmax_cross_entropy_with_integer_labels(
            student_logits, labels
        )
        if mask is not None:
            hard_loss = hard_loss * mask
            
        # Combine losses
        loss = alpha * hard_loss + (1 - alpha) * soft_loss
        metrics["hard_loss"] = jnp.mean(hard_loss)
    else:
        loss = soft_loss
        
    # Compute accuracy metrics
    student_preds = jnp.argmax(student_logits, axis=-1)
    teacher_preds = jnp.argmax(teacher_logits, axis=-1)
    
    if labels is not None:
        student_acc = jnp.mean(student_preds == labels)
        teacher_acc = jnp.mean(teacher_preds == labels)
        metrics.update({
            "student_accuracy": student_acc,
            "teacher_accuracy": teacher_acc
        })
    
    # Agreement between student and teacher
    agreement = jnp.mean(student_preds == teacher_preds)
    metrics["teacher_student_agreement"] = agreement
    
    return loss, metrics

def create_student_model(
    config: Dict[str, Any],
    teacher_model: Any = None,
    reduction_factor: float = 0.5,
    rng: Optional[jnp.ndarray] = None
) -> Tuple[EnhancedTransformerModel, Any, Dict[str, Any]]:
    """Create optimized student model for TPU v3-8."""
    
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    # Create student config
    if teacher_model is not None:
        student_config = create_student_config(teacher_model.config, reduction_factor)
    else:
        student_config = EnhancedTransformerConfig(**config)
    
    # Create TPU-optimized student model
    student_model = EnhancedTransformerModel(config=student_config)
    
    # Initialize with dummy input
    batch_size = 32  # Good TPU batch size
    seq_length = 512  # Start with shorter sequences
    dummy_input = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    variables = student_model.init(rng, dummy_input, deterministic=False)
    
    return student_model, variables, student_config.__dict__

def create_distillation_train_state(
    student_model: EnhancedTransformerModel,
    learning_rate: float = 1e-4,
    warmup_steps: int = 1000,
    rng: Optional[jnp.ndarray] = None
) -> Any:
    """Create training state optimized for TPU v3-8."""
    
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    # TPU-optimized learning rate schedule
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=100_000,
        end_value=learning_rate * 0.1
    )
    
    # Optimizer with gradient clipping and weight decay
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule_fn,
            b1=0.9,
            b2=0.999,
            eps=1e-8,
            weight_decay=0.01
        )
    )
    
    # Create training state
    return train_utils.create_train_state(
        model=student_model,
        optimizer=optimizer,
        rng=rng
    )

def initialize_from_teacher(
    student_state: Any,
    teacher_state: Any,
    method: str = "layer_mapping",
    mapping_strategy: str = "uniform"
) -> Any:
    """Initialize student from teacher with TPU optimizations."""
    
    student_params = student_state.params
    teacher_params = teacher_state.params
    
    # Get layer counts
    student_layers = len([k for k in student_params.keys() if "layer_" in k])
    teacher_layers = len([k for k in teacher_params.keys() if "layer_" in k])
    
    if method == "layer_mapping":
        # Create layer mapping
        if mapping_strategy == "uniform":
            # Uniform spacing of teacher layers
            indices = jnp.linspace(
                0, teacher_layers - 1,
                student_layers,
                dtype=jnp.int32
            )
        else:
            # Take last layers
            indices = jnp.arange(
                teacher_layers - student_layers,
                teacher_layers,
                dtype=jnp.int32
            )
            
        # Map layers
        for i, teacher_idx in enumerate(indices):
            student_key = f"layer_{i}"
            teacher_key = f"layer_{teacher_idx}"
            
            # Copy and reshape weights
            for param in ["attention", "intermediate", "output"]:
                if param in teacher_params[teacher_key]:
                    shape_student = student_params[student_key][param]["kernel"].shape
                    weights = teacher_params[teacher_key][param]["kernel"]
                    
                    # Reshape using TPU-optimized operations
                    if weights.shape != shape_student:
                        weights = jax.image.resize(
                            weights,
                            shape_student,
                            method="linear"
                        )
                    student_params[student_key][param]["kernel"] = weights
                    
    # Copy embeddings with shape adjustment
    if "embeddings" in teacher_params:
        shape_student = student_params["embeddings"]["kernel"].shape
        weights = teacher_params["embeddings"]["kernel"]
        if weights.shape != shape_student:
            weights = jax.image.resize(
                weights,
                shape_student,
                method="linear"
            )
        student_params["embeddings"]["kernel"] = weights
        
    return student_state.replace(params=student_params)