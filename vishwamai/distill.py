"""Knowledge distillation from Gemma to VishwamAI."""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Optional, Tuple, Callable
import optax
from dataclasses import dataclass
from .kernels.core.kernel import fp8_gemm_optimized
from .kernels.core import train_utils
from vishwamai.transformer import EnhancedTransformerConfig, EnhancedTransformerModel, TPUTrainingState
from vishwamai.thoughts.tot import TreeOfThoughts
from vishwamai.profiler import TPUProfiler
from vishwamai.model import VishwamAI, VishwamAIConfig

@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 2.0
    alpha: float = 0.5
    use_flash_attn: bool = True
    use_fp8: bool = True
    block_size: int = 128
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    max_steps: int = 100_000
    gradient_accumulation_steps: int = 4
    enable_thinking: bool = True
    thinking_weight: float = 0.1

class DistillationTrainer:
    """Trainer for knowledge distillation with TPU optimizations."""
    
    def __init__(
        self,
        teacher_model: Any,
        student_config: Dict[str, Any],
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_flash_attn: bool = True,
        profiler: Optional[TPUProfiler] = None
    ):
        """Initialize distillation trainer."""
        self.teacher_model = teacher_model
        self.config = DistillationConfig(
            temperature=temperature,
            alpha=alpha,
            use_flash_attn=use_flash_attn
        )
        self.profiler = profiler
        
        # Initialize with dummy input - removed deterministic parameter
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
        
        # Initialize student model without passing deterministic
        self.student_model, self.variables, self.student_config = create_student_model(
            student_config,
            self.teacher_model,
            reduction_factor=0.5,
            rng=rng
        )
        
        # Initialize Tree of Thoughts if enabled
        self.teacher_tot = None
        self.student_tot = None
        if self.config.enable_thinking:
            self.teacher_tot = TreeOfThoughts(self.teacher_model)
            self.student_tot = TreeOfThoughts(self.student_model)
        
        # Initialize training state
        self.state = self._create_train_state(rng)
        
        # Compile training step
        self.train_step = self._create_train_step()
    
    def _create_train_state(self, rng: jnp.ndarray) -> TPUTrainingState:
        """Create optimized training state."""
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.max_steps,
            end_value=self.config.learning_rate * 0.1
        )
        
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
        
        # Initialize state directly without using train_utils
        return TPUTrainingState.create(
            apply_fn=self.student_model.apply,
            params=self.variables["params"] if self.variables is not None else None,
            tx=optimizer
        )
    
    def _create_train_step(self) -> Callable:
        """Create TPU-optimized training step."""
        
        @jax.jit
        def train_step(
            state: TPUTrainingState,
            batch: Dict[str, jnp.ndarray],
            teacher_logits: jnp.ndarray,
            rng: jnp.ndarray
        ) -> Tuple[TPUTrainingState, Dict[str, Any]]:
            
            def loss_fn(params):
                # Get student outputs
                student_logits = state.apply_fn(
                    {"params": params},
                    batch["input_ids"],
                    deterministic=False,
                    rngs={"dropout": rng}
                )
                
                # Compute distillation loss
                loss, metrics = compute_distillation_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=batch.get("labels"),
                    temperature=self.config.temperature,
                    alpha=self.config.alpha
                )
                
                # Add thinking loss if enabled
                if self.config.enable_thinking and self.teacher_tot and self.student_tot:
                    teacher_thoughts = self.teacher_tot.generate_thoughts(
                        batch["input_ids"]
                    )
                    student_thoughts = self.student_tot.generate_thoughts(
                        batch["input_ids"]
                    )
                    thinking_loss = jnp.mean(
                        jnp.square(teacher_thoughts - student_thoughts)
                    )
                    loss = loss + self.config.thinking_weight * thinking_loss
                    metrics["thinking_loss"] = thinking_loss
                
                return loss, metrics
            
            # Compute gradients
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, metrics), grads = grad_fn(state.params)
            
            # Update state
            new_state = state.apply_gradients(grads=grads)
            
            # Update metrics 
            metrics["loss"] = loss
            metrics["learning_rate"] = self._get_learning_rate(state)
            
            return new_state, metrics
        
        return train_step
    
    def _get_learning_rate(self, state: TPUTrainingState) -> float:
        """Get current learning rate."""
        if hasattr(state.opt_state, "hyperparams") and "learning_rate" in state.opt_state.hyperparams:
            return state.opt_state.hyperparams["learning_rate"]
        # Fallback to accessing through schedule function
        step = state.step
        return self.config.learning_rate  # Fallback to config value
    
    def train(
        self,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        max_steps: Optional[int] = None,
        eval_steps: int = 100,
        save_steps: int = 1000,
        checkpoint_dir: Optional[str] = None
    ):
        """Train student model with knowledge distillation."""
        max_steps = max_steps or self.config.max_steps
        step = 0
        
        for batch in train_dataset:
            if step >= max_steps:
                break
            
            # Get teacher predictions
            teacher_logits = self.teacher_model(
                batch["input_ids"],
                deterministic=True
            )
            
            # Training step
            rng = jax.random.fold_in(jax.random.PRNGKey(42), step)
            self.state, metrics = self.train_step(
                self.state,
                batch,
                teacher_logits,
                rng
            )
            
            # Log metrics
            if self.profiler and step % 100 == 0:
                self.profiler.log_metrics(metrics, step)
            
            # Evaluate
            if eval_dataset and step % eval_steps == 0:
                eval_metrics = self.evaluate(eval_dataset)
                if self.profiler:
                    self.profiler.log_metrics(eval_metrics, step, prefix="eval/")
            
            # Save checkpoint
            if checkpoint_dir and step % save_steps == 0:
                train_utils.save_checkpoint(
                    checkpoint_dir,
                    self.state,
                    step,
                    keep=2
                )
            
            step += 1
    
    def evaluate(self, eval_dataset: Any) -> Dict[str, float]:
        """Evaluate student model."""
        metrics = []
        
        for batch in eval_dataset:
            # Get teacher and student predictions
            teacher_logits = self.teacher_model(
                batch["input_ids"],
                deterministic=True
            )
            
            student_logits = self.student_model.apply(
                {"params": self.state.params},
                batch["input_ids"],
                deterministic=True
            )
            
            # Compute metrics
            _, batch_metrics = compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=batch.get("labels"),
                temperature=self.config.temperature,
                alpha=self.config.alpha
            )
            metrics.append(batch_metrics)
        
        # Average metrics
        avg_metrics = jax.tree_map(
            lambda *x: jnp.mean(jnp.stack(x)),
            *metrics
        )
        
        return avg_metrics

def create_student_config(teacher_config: Dict[str, Any], reduction_factor: float) -> EnhancedTransformerConfig:
    """Create credit-optimized student config for TPU v3-8."""
    # Handle both hidden_size and hidden_dim keys for compatibility
    hidden_size = teacher_config.get("hidden_size", teacher_config.get("hidden_dim"))
    if hidden_size is None:
        raise ValueError("Teacher config must specify either hidden_size or hidden_dim")
    
    # Calculate efficient dimensions that are multiples of 128 for TPU
    hidden_size = max(512, ((int(hidden_size * reduction_factor) + 127) // 128) * 128)
    intermediate_size = max(2048, ((int(hidden_size * 4 * reduction_factor) + 127) // 128) * 128)
    
    # Optimize number of heads for TPU utilization
    num_attention_heads = max(8, ((int(teacher_config.get("num_attention_heads", teacher_config.get("num_heads", 8)) * reduction_factor) + 3) // 4) * 4)
    
    # Use shorter sequences initially and gradually increase
    max_seq_length = min(teacher_config.get("max_position_embeddings", teacher_config.get("max_seq_len", 1024)), 1024)
    
    return EnhancedTransformerConfig(
        vocab_size=teacher_config["vocab_size"],
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=max(4, int(teacher_config.get("num_layers", teacher_config.get("num_hidden_layers", 12)) * reduction_factor)),
        intermediate_size=intermediate_size,
        max_position_embeddings=max_seq_length,
        dropout_rate=0.1,
        attention_dropout=0.1,
        use_flash_attention=True,
        use_fp8=True,
        use_parallel=True,
        block_size=128
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
    
    # Mean over non-masked elements
    if mask is not None:
        # Safe division to avoid dividing by zero
        mask_sum = jnp.sum(mask)
        soft_loss_mean = jnp.sum(soft_loss) / jnp.maximum(mask_sum, 1.0)
    else:
        soft_loss_mean = jnp.mean(soft_loss)
        
    metrics = {"soft_loss": soft_loss_mean}
    
    # Compute hard loss if labels provided
    if labels is not None:
        hard_loss = optax.softmax_cross_entropy_with_integer_labels(
            student_logits, labels
        )
        if mask is not None:
            hard_loss = hard_loss * mask
            # Safe division
            hard_loss_mean = jnp.sum(hard_loss) / jnp.maximum(jnp.sum(mask), 1.0)
        else:
            hard_loss_mean = jnp.mean(hard_loss)
            
        # Combine losses
        loss = alpha * hard_loss_mean + (1 - alpha) * soft_loss_mean
        metrics["hard_loss"] = hard_loss_mean
    else:
        loss = soft_loss_mean
        
    # Compute accuracy metrics
    student_preds = jnp.argmax(student_logits, axis=-1)
    teacher_preds = jnp.argmax(teacher_logits, axis=-1)
    
    if labels is not None:
        if mask is not None:
            # Calculate accuracy only on masked positions
            student_correct = (student_preds == labels) * mask
            teacher_correct = (teacher_preds == labels) * mask
            student_acc = jnp.sum(student_correct) / jnp.maximum(jnp.sum(mask), 1.0)
            teacher_acc = jnp.sum(teacher_correct) / jnp.maximum(jnp.sum(mask), 1.0)
        else:
            student_acc = jnp.mean(student_preds == labels)
            teacher_acc = jnp.mean(teacher_preds == labels)
        
        metrics.update({
            "student_accuracy": student_acc,
            "teacher_accuracy": teacher_acc
        })
    
    # Agreement between student and teacher
    if mask is not None:
        agreement = jnp.sum((student_preds == teacher_preds) * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    else:
        agreement = jnp.mean(student_preds == teacher_preds)
    
    metrics["teacher_student_agreement"] = agreement
    
    return loss, metrics

def create_student_model(student_config, teacher_model=None, reduction_factor=0.5, rng=None):
    """Create a student model for distillation."""
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    # Create dummy input for initialization
    dummy_input = jnp.ones((1, 32), dtype=jnp.int32)
    
    # Create student model with reduced size
    student_model = VishwamAI(
        config=VishwamAIConfig(
            vocab_size=student_config["vocab_size"],
            hidden_dim=int(student_config["hidden_dim"] * reduction_factor),
            num_layers=max(1, int(student_config["num_layers"] * reduction_factor)),
            num_heads=max(1, int(student_config["num_heads"] * reduction_factor)),
            head_dim=student_config["head_dim"],
            mlp_dim=int(student_config["mlp_dim"] * reduction_factor),
            dropout_rate=student_config["dropout_rate"],
            attention_dropout=student_config["attention_dropout"]
        )
    )
    
    # Initialize model
    variables = student_model.init(rng, dummy_input)
    return student_model, variables, student_config

def initialize_from_teacher(
    student_state: Any,
    teacher_state: Any,
    method: str = "layer_mapping",
    mapping_strategy: str = "uniform"
) -> Any:
    """Initialize student from teacher with TPU optimizations."""
    
    student_params = student_state.params
    teacher_params = teacher_state.params
    
    # Handle potential param structures
    if not isinstance(student_params, dict):
        student_params = student_params.unfreeze() if hasattr(student_params, 'unfreeze') else dict(student_params)
    if not isinstance(teacher_params, dict):
        teacher_params = teacher_params.unfreeze() if hasattr(teacher_params, 'unfreeze') else dict(teacher_params)
    
    # Get layer counts (handle nested structure)
    def count_layers(params, prefix="layer_"):
        if isinstance(params, dict):
            return len([k for k in params.keys() if prefix in k])
        return 0
    
    student_layers = count_layers(student_params)
    teacher_layers = count_layers(teacher_params)
    
    # If no layers found with direct naming, try to find nested structure
    if student_layers == 0 or teacher_layers == 0:
        # Try to find layers in nested structure
        for key in student_params:
            if isinstance(student_params[key], dict):
                student_layers = count_layers(student_params[key])
                if student_layers > 0:
                    break
        
        for key in teacher_params:
            if isinstance(teacher_params[key], dict):
                teacher_layers = count_layers(teacher_params[key])
                if teacher_layers > 0:
                    break
    
    if method == "layer_mapping" and student_layers > 0 and teacher_layers > 0:
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
            
            # Handle potential nested structure
            def get_nested_dict(params, key):
                if key in params:
                    return params[key]
                for k, v in params.items():
                    if isinstance(v, dict) and key in v:
                        return v[key]
                return None
            
            student_layer = get_nested_dict(student_params, student_key)
            teacher_layer = get_nested_dict(teacher_params, teacher_key)
            
            if student_layer is not None and teacher_layer is not None:
                # Copy and reshape weights
                for param in ["attention", "intermediate", "output"]:
                    if param in teacher_layer and param in student_layer:
                        if "kernel" in teacher_layer[param] and "kernel" in student_layer[param]:
                            shape_student = student_layer[param]["kernel"].shape
                            weights = teacher_layer[param]["kernel"]
                            
                            # Reshape using TPU-optimized operations
                            if weights.shape != shape_student:
                                # Use a safe reshape that works for different dimensions
                                if len(weights.shape) == len(shape_student):
                                    try:
                                        weights = jax.image.resize(
                                            weights,
                                            shape_student,
                                            method="linear"
                                        )
                                    except:
                                        # Fallback to dimension-specific reshaping
                                        if len(weights.shape) == 2:
                                            weights = jnp.resize(weights, shape_student)
                                else:
                                    # Different number of dimensions, use a safe copy
                                    weights = jnp.zeros(shape_student)
                            
                            student_layer[param]["kernel"] = weights
                    
    # Copy embeddings with shape adjustment
    if "embeddings" in teacher_params and "embeddings" in student_params:
        if "kernel" in teacher_params["embeddings"] and "kernel" in student_params["embeddings"]:
            shape_student = student_params["embeddings"]["kernel"].shape
            weights = teacher_params["embeddings"]["kernel"]
            
            if weights.shape != shape_student:
                try:
                    # For 2D matrices, use resize
                    if len(weights.shape) == 2 and len(shape_student) == 2:
                        weights = jnp.resize(weights, shape_student)
                    else:
                        weights = jax.image.resize(
                            weights,
                            shape_student,
                            method="linear"
                        )
                except:
                    # Fallback to zeros
                    weights = jnp.zeros(shape_student)
                    
            student_params["embeddings"]["kernel"] = weights
        
    return student_state.replace(params=student_params)