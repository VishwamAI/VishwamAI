"""TPU-optimized knowledge distillation implementation with linear path embeddings"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Dict, Optional, Tuple
from .kernels.kernel import fp8_gemm_optimized, act_quant
from .layers.layers import TPUGEMMLinear, RMSNorm

class LinearPathDistillation(nn.Module):
    """Linear path embedding distillation layer"""
    hidden_dim: int
    dropout_rate: float = 0.1
    
    def setup(self):
        self.linear_projection = TPUGEMMLinear(features=self.hidden_dim)
        self.layer_norm = RMSNorm()
    
    def __call__(self, x, training=False):
        x = self.linear_projection(x)
        x = self.layer_norm(x)
        if training:
            x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        return x

class DistillationTrainer:
    """TPU-optimized knowledge distillation trainer"""
    
    def __init__(
        self,
        teacher_model: Any,
        student_config: Dict[str, Any],
        temperature: float = 2.0,
        alpha: float = 0.5,
        use_flash_attn: bool = True,
        use_fp8: bool = True,
        block_size: int = 128,  # Optimal for TPU v2
        use_linear_path: bool = True
    ):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.use_flash_attn = use_flash_attn
        self.use_fp8 = use_fp8
        self.block_size = block_size
        self.use_linear_path = use_linear_path
        
        # Create student model with TPU optimizations
        self.student_model = EnhancedTransformerModel(
            vocab_size=student_config['vocab_size'],
            num_layers=student_config['num_layers'],
            num_heads=student_config['num_heads'],
            head_dim=student_config['head_dim'],
            hidden_dim=student_config['hidden_dim'],
            mlp_dim=student_config['mlp_dim'],
            max_seq_len=student_config['max_seq_len'],
            use_flash_attn=use_flash_attn,
            use_rms_norm=True,  # Better training stability
            dtype=jnp.bfloat16  # TPU v2/v3 optimal
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
        """
        Single training step with TPU optimization.
        """
        def loss_fn(params):
            # Get teacher logits in chunks to save memory
            teacher_chunks = []
            for i in range(0, batch['input_ids'].shape[1], self.block_size):
                chunk = jax.lax.dynamic_slice(
                    batch['input_ids'],
                    (0, i),
                    (batch['input_ids'].shape[0], min(self.block_size, batch['input_ids'].shape[1] - i))
                )
                teacher_chunk = self.teacher_model(
                    chunk,
                    deterministic=True
                )
                teacher_chunks.append(teacher_chunk)
            teacher_logits = jnp.concatenate(teacher_chunks, axis=1)
            
            # Student forward pass
            student_logits = self.student_model.apply(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            # Compute loss
            loss = self.compute_distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=batch['labels'],
                mask=batch.get('attention_mask')
            )
            
            return loss, (student_logits, teacher_logits)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (student_logits, teacher_logits)), grads = grad_fn(student_state.params)
        
        # Update student model
        new_student_state = student_state.apply_gradients(grads=grads)
        
        # Compute metrics
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

    def advanced_loss_functions(
        self,
        student_logits: jnp.ndarray,
        teacher_logits: jnp.ndarray,
        student_embeddings: Optional[jnp.ndarray] = None,
        teacher_embeddings: Optional[jnp.ndarray] = None,
        labels: jnp.ndarray = None,
        mask: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """Advanced loss functions for improved distillation performance"""
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

    def improved_training_strategies(
        self,
        student_state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """
        Improved training strategies for better distillation performance.
        """
        def loss_fn(params):
            # Get teacher logits in chunks to save memory
            teacher_chunks = []
            for i in range(0, batch['input_ids'].shape[1], self.block_size):
                chunk = jax.lax.dynamic_slice(
                    batch['input_ids'],
                    (0, i),
                    (batch['input_ids'].shape[0], min(self.block_size, batch['input_ids'].shape[1] - i))
                )
                teacher_chunk = self.teacher_model(
                    chunk,
                    deterministic=True
                )
                teacher_chunks.append(teacher_chunk)
            teacher_logits = jnp.concatenate(teacher_chunks, axis=1)
            
            # Student forward pass
            student_logits = self.student_model.apply(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            # Compute loss
            loss = self.advanced_loss_functions(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=batch['labels'],
                mask=batch.get('attention_mask')
            )
            
            return loss, (student_logits, teacher_logits)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (student_logits, teacher_logits)), grads = grad_fn(student_state.params)
        
        # Update student model
        new_student_state = student_state.apply_gradients(grads=grads)
        
        # Compute metrics
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
        teacher_norm = RMSNorm(dtype=teacher_hidden.dtype)(teacher_hidden)
        student_norm = RMSNorm(dtype=student_hidden.dtype)(student_proj)
        
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
