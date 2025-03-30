"""TPU-optimized knowledge distillation implementation with linear path embeddings and Flash attention"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import copy
from typing import Any, Dict, Optional, Tuple, List, Union, Callable
from vishwamai.kernels.tpu.kernels import fp8_gemm_optimized, act_quant
from vishwamai.layers.layers import TPUGEMMLinear, TPURMSNorm
from vishwamai.transformer import EnhancedTransformerModel, create_train_state

class LinearPathDistillation(nn.Module):
    """Linear path embedding distillation layer optimized for TPU v5e"""
    hidden_dim: int
    dropout_rate: float = 0.1
    
    def setup(self):
        # Use TPU-optimized linear projection
        self.linear_projection = TPUGEMMLinear(features=self.hidden_dim)
        self.layer_norm = TPURMSNorm()
    
    def __call__(self, x, training=False):
        # Project and normalize with optimized kernels
        x = self.linear_projection(x)
        x = self.layer_norm(x)
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x

class DistillationTrainer:
    """TPU-optimized knowledge distillation trainer for 13B student"""
    
    def __init__(
        self,
        teacher_model: Any,
        student_config: Dict[str, Any],
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_flash_attn: bool = True,
        use_fp8: bool = True, 
        block_size: int = 128,  # Optimal for TPU v5e
        use_linear_path: bool = True
    ):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.use_flash_attn = use_flash_attn
        self.use_fp8 = use_fp8
        self.block_size = block_size
        self.use_linear_path = use_linear_path
        
        # Create 13B student model with TPU v5e optimizations
        self.student_model = EnhancedTransformerModel(
            vocab_size=student_config['vocab_size'],
            num_layers=40,  # Increased for 13B
            num_heads=24,   # Scaled up
            head_dim=128,   # Optimized for v5e
            hidden_dim=3072, # 13B architecture
            mlp_dim=12288,  # 4x hidden_dim
            max_seq_len=student_config['max_seq_len'],
            use_flash_attn=use_flash_attn,
            use_rms_norm=True,  # Better training stability
            dtype=jnp.bfloat16  # TPU v5e optimal
        )
        
        if use_linear_path:
            self.embedding_distill = LinearPathDistillation(hidden_dim=student_config['hidden_dim'])

    def compute_distillation_loss(
        self,
        student_logits: jnp.ndarray,
        teacher_logits: jnp.ndarray,
        student_embeddings: Optional[jnp.ndarray] = None,
        teacher_embeddings: Optional[jnp.ndarray] = None,
        labels: jnp.ndarray = None,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Enhanced distillation loss with embedding path"""
        # Compute regular distillation loss
        teacher_probs = jax.nn.softmax(teacher_logits / self.temperature, axis=-1)
        student_logits_temp = student_logits / self.temperature
        student_log_probs = jax.nn.log_softmax(student_logits_temp, axis=-1)
        distill_loss = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
        distill_loss = distill_loss * (self.temperature ** 2)
        
        # Add embedding path loss if enabled
        if self.use_linear_path and student_embeddings is not None and teacher_embeddings is not None:
            embedding_loss = jnp.mean(jnp.square(
                self.embedding_distill(student_embeddings) - teacher_embeddings
            ))
            distill_loss = distill_loss + 0.1 * embedding_loss  # Weighted combination
        
        # Hard target cross entropy loss
        if labels is not None:
            hard_loss = jax.nn.sparse_categorical_crossentropy(
                logits=student_logits,
                labels=labels,
                from_logits=True
            )
            if mask is not None:
                hard_loss = hard_loss * mask
                distill_loss = distill_loss * mask
            
            loss = self.alpha * jnp.mean(distill_loss) + (1 - self.alpha) * jnp.mean(hard_loss)
        else:
            loss = jnp.mean(distill_loss)
            
        return loss

    @jax.jit
    def train_step(
        self,
        student_state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Single training step with TPU v5e optimization"""
        
        def loss_fn(params):
            # Process teacher in memory-efficient chunks
            teacher_chunks = []
            for i in range(0, batch['input_ids'].shape[1], self.block_size):
                chunk = jax.lax.dynamic_slice(
                    batch['input_ids'],
                    (0, i),
                    (batch['input_ids'].shape[0], 
                     min(self.block_size, batch['input_ids'].shape[1] - i))
                )
                teacher_chunk = self.teacher_model(
                    chunk,
                    deterministic=True
                )
                teacher_chunks.append(teacher_chunk)
            teacher_logits = jnp.concatenate(teacher_chunks, axis=1)
            
            # Student forward pass with Flash attention
            student_logits = self.student_model.apply(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            # Compute loss with optimized kernels
            loss = self.compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=batch['labels'],
                mask=batch.get('attention_mask')
            )
            
            return loss, (student_logits, teacher_logits)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (student_logits, teacher_logits)), grads = grad_fn(student_state.params)
        
        # Update with TPU-optimized gradient application
        new_student_state = student_state.apply_gradients(grads=grads)
        
        # Track key metrics
        metrics = {
            'loss': loss,
            'student_perplexity': jnp.exp(jnp.mean(
                jax.nn.sparse_categorical_crossentropy(
                    logits=student_logits,
                    labels=batch['labels'],
                    from_logits=True
                )
            )),
            'teacher_perplexity': jnp.exp(jnp.mean(
                jax.nn.sparse_categorical_crossentropy(
                    logits=teacher_logits,
                    labels=batch['labels'],
                    from_logits=True
                )
            ))
        }
        
        return new_student_state, metrics

class IntermediateLayerDistillation(nn.Module):
    """
    TPU-optimized intermediate layer distillation.
    Projects and aligns intermediate representations between teacher and student.
    """
    teacher_dim: int
    student_dim: int
    temperature: float = 1.0
    
    @nn.compact
    def __call__(
        self,
        teacher_hidden: jnp.ndarray,
        student_hidden: jnp.ndarray
    ) -> jnp.ndarray:
        # Project student hidden states if dimensions don't match
        if self.student_dim != self.teacher_dim:
            student_proj = TPUGEMMLinear(
                features=self.teacher_dim,
                dtype=student_hidden.dtype
            )(student_hidden)
        else:
            student_proj = student_hidden
            
        # Normalize representations
        teacher_norm = TPURMSNorm(dtype=teacher_hidden.dtype)(teacher_hidden)
        student_norm = TPURMSNorm(dtype=student_hidden.dtype)(student_proj)
        
        # Compute cosine similarity loss
        if self.temperature != 1.0:
            teacher_norm = teacher_norm / self.temperature
            student_norm = student_norm / self.temperature
            
        # Use FP8 GEMM for faster computation
        teacher_quant, t_scale = act_quant(teacher_norm)
        student_quant, s_scale = act_quant(student_norm)
        
        similarity = fp8_gemm_optimized(
            student_quant, s_scale,
            teacher_quant, t_scale
        )
        
        return -jnp.mean(similarity)  # Negative since we want to maximize similarity

class ProgressiveLayerDropout(nn.Module):
    """
    Progressive layer dropout for efficient distillation training.
    Gradually increases dropout rate for deeper layers.
    """
    num_layers: int
    base_rate: float = 0.1
    max_rate: float = 0.3
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, layer_idx: int, training: bool = True) -> jnp.ndarray:
        if not training:
            return x
            
        # Compute progressive dropout rate
        dropout_rate = self.base_rate + (
            (self.max_rate - self.base_rate) * 
            (layer_idx / (self.num_layers - 1))
        )
        
        return nn.Dropout(rate=dropout_rate)(x, deterministic=not training)

def create_layer_mapping(
    teacher_layers: int,
    student_layers: int,
    strategy: str = 'uniform'
) -> Dict[int, int]:
    """
    Create mapping between teacher and student layers for intermediate distillation.
    
    Args:
        teacher_layers: Number of teacher layers
        student_layers: Number of student layers
        strategy: Mapping strategy ('uniform' or 'last_layers')
    """
    if strategy == 'uniform':
        # Uniformly space student layers to match teacher
        indices = jnp.linspace(0, teacher_layers - 1, student_layers)
        return {i: int(idx) for i, idx in enumerate(indices)}
    elif strategy == 'last_layers':
        # Focus on last layers of teacher
        return {
            i: teacher_layers - student_layers + i 
            for i in range(student_layers)
        }
    else:
        raise ValueError(f"Unknown mapping strategy: {strategy}")

def compute_attention_distillation_loss(
    student_attention: jnp.ndarray,
    teacher_attention: jnp.ndarray,
    temperature: float = 1.0,
    mask: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """
    Compute attention map distillation loss.
    
    Args:
        student_attention: Student attention weights
        teacher_attention: Teacher attention weights
        temperature: Distillation temperature
        mask: Optional attention mask
    """
    # Scale attention maps by temperature
    student_attention = student_attention / temperature
    teacher_attention = teacher_attention / temperature
    
    # Compute KL divergence
    teacher_probs = jax.nn.softmax(teacher_attention, axis=-1)
    student_log_probs = jax.nn.log_softmax(student_attention, axis=-1)
    
    loss = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
    
    if mask is not None:
        loss = loss * mask
        
    return jnp.mean(loss) * (temperature ** 2)

def compute_distillation_loss(
    student_logits: jnp.ndarray,
    teacher_logits: jnp.ndarray,
    labels: Optional[jnp.ndarray] = None,
    temperature: float = 2.0,
    alpha: float = 0.5,
    mask: Optional[jnp.ndarray] = None
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Compute distillation loss between student and teacher models.
    
    Args:
        student_logits: Logits from student model, shape [batch, seq_len, vocab_size]
        teacher_logits: Logits from teacher model, shape [batch, seq_len, vocab_size]
        labels: Optional ground truth labels for hard loss, shape [batch, seq_len]
        temperature: Distillation temperature (higher = softer probabilities)
        alpha: Weight for soft loss vs hard loss (1.0 = only soft loss)
        mask: Optional attention mask, shape [batch, seq_len]
        
    Returns:
        Tuple of (total loss, dictionary of metrics)
    """
    # Compute soft distillation loss (KL-divergence)
    teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
    student_logits_temp = student_logits / temperature
    student_log_probs = jax.nn.log_softmax(student_logits_temp, axis=-1)
    
    # KL divergence: teacher_probs * (log(teacher_probs) - log(student_probs))
    # Simplifying to: teacher_probs * -log(student_probs) since teacher_probs part is constant
    soft_loss = -jnp.sum(teacher_probs * student_log_probs, axis=-1)
    soft_loss = soft_loss * (temperature ** 2)  # Scale by temperature²
    
    if mask is not None:
        soft_loss = soft_loss * mask
        
    metrics = {"soft_loss": jnp.mean(soft_loss)}
    
    # Compute hard loss if labels are provided
    if labels is not None:
        hard_loss = jax.nn.sparse_categorical_crossentropy(
            logits=student_logits,
            labels=labels,
            from_logits=True
        )
        
        if mask is not None:
            hard_loss = hard_loss * mask
            
        metrics["hard_loss"] = jnp.mean(hard_loss)
        
        # Weighted combination of soft and hard loss
        total_loss = alpha * jnp.mean(soft_loss) + (1 - alpha) * jnp.mean(hard_loss)
    else:
        total_loss = jnp.mean(soft_loss)
    
    # Calculate additional metrics
    student_preds = jnp.argmax(student_logits, axis=-1)
    teacher_preds = jnp.argmax(teacher_logits, axis=-1)
    
    if labels is not None:
        student_accuracy = jnp.mean(student_preds == labels)
        teacher_accuracy = jnp.mean(teacher_preds == labels)
        metrics.update({
            "student_accuracy": student_accuracy,
            "teacher_accuracy": teacher_accuracy
        })
    
    # Agreement between student and teacher
    agreement = jnp.mean(student_preds == teacher_preds)
    metrics["teacher_student_agreement"] = agreement
    
    return total_loss, metrics

def create_student_model(
    config: Dict[str, Any],
    teacher_model: Any = None,
    reduction_factor: float = 0.5,
    rng: Optional[jnp.ndarray] = None
) -> Tuple[EnhancedTransformerModel, Any, Dict[str, Any]]:
    """
    Create a student model from a teacher model or configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        teacher_model: Optional teacher model to derive student architecture
        reduction_factor: Factor to reduce model size (if creating from teacher)
        rng: Optional JAX PRNGKey for initialization
        
    Returns:
        Tuple of (student model, initialized variables, student config)
    """
    if rng is None:
        rng = jax.random.PRNGKey(42)
    
    # If teacher model is provided, derive student config from it
    if teacher_model is not None:
        teacher_config = teacher_model.config
        
        # Calculate reduced dimensions based on reduction factor
        student_config = {
            "vocab_size": teacher_config.vocab_size,  # Keep same vocabulary
            "num_layers": max(2, int(teacher_config.num_layers * reduction_factor)),
            "num_heads": max(2, int(teacher_config.num_heads * reduction_factor)),
            "head_dim": teacher_config.head_dim,  # Usually keep same head dimension
            "hidden_dim": max(256, int(teacher_config.hidden_dim * reduction_factor)),
            "mlp_dim": max(512, int(teacher_config.mlp_dim * reduction_factor)),
            "max_seq_len": teacher_config.max_seq_len,
            "dropout_rate": config.get("dropout_rate", 0.1),
            "attention_dropout": config.get("attention_dropout", 0.1)
        }
    else:
        # Use provided config directly
        student_config = config
    
    # TPU-specific optimizations
    use_flash_attn = config.get("use_flash_attn", True)
    use_rms_norm = config.get("use_rms_norm", True)
    dtype = config.get("dtype", jnp.bfloat16)  # TPU optimal
    
    # Create student model with TPU optimizations
    student_model = EnhancedTransformerModel(
        vocab_size=student_config.get("vocab_size"),
        num_layers=student_config.get("num_layers"),
        num_heads=student_config.get("num_heads"),
        head_dim=student_config.get("head_dim"),
        hidden_dim=student_config.get("hidden_dim"),
        mlp_dim=student_config.get("mlp_dim"),
        max_seq_len=student_config.get("max_seq_len"),
        dropout_rate=student_config.get("dropout_rate", 0.1),
        attention_dropout=student_config.get("attention_dropout", 0.1),
        use_flash_attn=use_flash_attn,
        use_rms_norm=use_rms_norm,
        dtype=dtype
    )
    
    # Initialize model variables with dummy input
    dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
    variables = student_model.init(rng, dummy_input, deterministic=False)
    
    return student_model, variables, student_config

def initialize_from_teacher(
    student_state: Any,
    teacher_state: Any,
    method: str = "layer_mapping",
    mapping_strategy: str = "uniform"
) -> Any:
    """
    Initialize student model parameters from teacher model.
    
    Args:
        student_state: Student model training state
        teacher_state: Teacher model training state
        method: Initialization method - 'layer_mapping', 'first_layers', 'layer_random'
        mapping_strategy: Strategy for layer mapping ('uniform', 'last_layers')
        
    Returns:
        Updated student model training state
    """
    student_params = student_state.params
    teacher_params = teacher_state.params
    
    # Count student and teacher layers
    student_num_layers = sum(1 for k in student_params.keys() if k.startswith("layers_"))
    teacher_num_layers = sum(1 for k in teacher_params.keys() if k.startswith("layers_"))
    
    # Create deep copy of student params to modify
    new_student_params = copy.deepcopy(student_params)
    
    # Initialize embedding and output layers from teacher (vocabulary must match)
    if "embedding" in student_params and "embedding" in teacher_params:
        # Check shape compatibility
        s_emb_shape = student_params["embedding"]["embedding"].shape
        t_emb_shape = teacher_params["embedding"]["embedding"].shape
        
        if s_emb_shape[0] == t_emb_shape[0]:  # Vocab size must match
            if s_emb_shape[1] <= t_emb_shape[1]:  # Student embedding size <= teacher
                # Copy part or all of teacher embeddings
                new_student_params["embedding"]["embedding"] = teacher_params["embedding"]["embedding"][:, :s_emb_shape[1]]
    
    # Output projection has to match exactly, so only copy if dimensions match
    if "output_proj" in student_params and "output_proj" in teacher_params:
        s_out_shape = student_params["output_proj"]["kernel"].shape
        t_out_shape = teacher_params["output_proj"]["kernel"].shape
        
        if s_out_shape == t_out_shape:
            new_student_params["output_proj"] = teacher_params["output_proj"]
    
    # Handle layer initialization based on method
    if method == "layer_mapping":
        # Create mapping between teacher and student layers
        layer_map = create_layer_mapping(
            teacher_num_layers,
            student_num_layers,
            strategy=mapping_strategy
        )
        
        # Copy parameters from teacher to student based on mapping
        for student_idx, teacher_idx in layer_map.items():
            student_key = f"layers_{student_idx}"
            teacher_key = f"layers_{teacher_idx}"
            
            if student_key in student_params and teacher_key in teacher_params:
                # For each parameter in the layer
                for param_name in student_params[student_key]:
                    if param_name in teacher_params[teacher_key]:
                        s_param = student_params[student_key][param_name]
                        t_param = teacher_params[teacher_key][param_name]
                        
                        # Handle different parameter shapes
                        if isinstance(s_param, dict):
                            # Handle nested dictionaries (e.g., attention heads)
                            for sub_name in s_param:
                                if sub_name in t_param:
                                    s_shape = s_param[sub_name].shape
                                    t_shape = t_param[sub_name].shape
                                    
                                    if s_shape == t_shape:
                                        new_student_params[student_key][param_name][sub_name] = t_param[sub_name]
                                    elif len(s_shape) == len(t_shape):
                                        # Try to copy compatible parts
                                        slices = tuple(slice(0, min(s, t)) for s, t in zip(s_shape, t_shape))
                                        new_student_params[student_key][param_name][sub_name] = t_param[sub_name][slices]
                        else:
                            # Direct parameter tensors
                            s_shape = s_param.shape
                            t_shape = t_param.shape
                            
                            if s_shape == t_shape:
                                new_student_params[student_key][param_name] = t_param
                            elif len(s_shape) == len(t_shape):
                                # Try to copy compatible parts 
                                slices = tuple(slice(0, min(s, t)) for s, t in zip(s_shape, t_shape))
                                new_student_params[student_key][param_name] = t_param[slices]
    
    elif method == "first_layers":
        # Copy only the first N layers directly
        copy_layers = min(student_num_layers, teacher_num_layers // 2)
        
        for i in range(copy_layers):
            student_key = f"layers_{i}"
            teacher_key = f"layers_{i}"
            
            if student_key in student_params and teacher_key in teacher_params:
                # Similar parameter copying logic as above
                for param_name in student_params[student_key]:
                    if param_name in teacher_params[teacher_key]:
                        s_param = student_params[student_key][param_name]
                        t_param = teacher_params[teacher_key][param_name]
                        
                        if isinstance(s_param, dict):
                            for sub_name in s_param:
                                if sub_name in t_param:
                                    s_shape = s_param[sub_name].shape
                                    t_shape = t_param[sub_name].shape
                                    
                                    if s_shape == t_shape:
                                        new_student_params[student_key][param_name][sub_name] = t_param[sub_name]
                                    elif len(s_shape) == len(t_shape):
                                        slices = tuple(slice(0, min(s, t)) for s, t in zip(s_shape, t_shape))
                                        new_student_params[student_key][param_name][sub_name] = t_param[sub_name][slices]
                        else:
                            s_shape = s_param.shape
                            t_shape = t_param.shape
                            
                            if s_shape == t_shape:
                                new_student_params[student_key][param_name] = t_param
                            elif len(s_shape) == len(t_shape):
                                slices = tuple(slice(0, min(s, t)) for s, t in zip(s_shape, t_shape))
                                new_student_params[student_key][param_name] = t_param[slices]
    
    elif method == "layer_random":
        # Only keep embeddings and output layer from teacher, leave other layers random
        # (Already handled above)
        pass
    
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    # Update student state with new parameters
    new_student_state = student_state.replace(params=new_student_params)
    return new_student_state

# Export public API
__all__ = [
    'compute_distillation_loss',
    'create_student_model',
    'initialize_from_teacher',
    'DistillationTrainer',
    'IntermediateLayerDistillation',
    'LinearPathDistillation',
    'ProgressiveLayerDropout',
    'compute_attention_distillation_loss',
    'create_layer_mapping'
]