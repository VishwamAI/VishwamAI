"""TPU-optimized kernels for knowledge distillation."""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import numpy as np

@dataclass
class DistillationKernelConfig:
    """Configuration for distillation kernels."""
    temperature: float = 1.0
    alpha: float = 0.5  # Weight for distillation loss
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    max_sequence_length: int = 512
    vocab_size: int = 32000
    
@dataclass 
class DistillationOutput:
    """Output from distillation kernels."""
    student_logits: jnp.ndarray
    teacher_logits: jnp.ndarray
    attention_loss: float
    hidden_loss: float
    prediction_loss: float
    total_loss: float

class DistillationKernelManager:
    """Manages TPU kernels for knowledge distillation."""
    
    def __init__(self, config: DistillationKernelConfig):
        self.config = config
        
    def create_attention_distillation_kernel(
        self,
        num_heads: int,
        head_size: int
    ) -> Callable:
        """Create kernel for attention map distillation."""
        
        @jax.jit
        def attention_kernel(
            student_q: jnp.ndarray,
            student_k: jnp.ndarray,
            student_v: jnp.ndarray,
            teacher_q: jnp.ndarray,
            teacher_k: jnp.ndarray,
            teacher_v: jnp.ndarray,
            attention_mask: Optional[jnp.ndarray] = None
        ) -> Tuple[jnp.ndarray, float]:
            # Compute attention scores
            student_scores = jnp.matmul(student_q, student_k.transpose(-2, -1))
            teacher_scores = jnp.matmul(teacher_q, teacher_k.transpose(-2, -1))
            
            # Scale scores
            scale = 1.0 / jnp.sqrt(head_size)
            student_scores *= scale
            teacher_scores *= scale
            
            if attention_mask is not None:
                student_scores += attention_mask
                teacher_scores += attention_mask
                
            # Apply temperature scaling for distillation
            student_attn = jax.nn.softmax(
                student_scores / self.config.temperature,
                axis=-1
            )
            teacher_attn = jax.nn.softmax(
                teacher_scores / self.config.temperature,
                axis=-1
            )
            
            # Compute attention outputs
            student_output = jnp.matmul(student_attn, student_v)
            
            # Compute attention map distillation loss
            attention_loss = jnp.mean(
                jax.nn.kl_divergence(
                    jax.nn.log_softmax(teacher_scores, axis=-1),
                    jax.nn.log_softmax(student_scores, axis=-1)
                )
            )
            
            return student_output, attention_loss
            
        return attention_kernel
        
    def create_hidden_distillation_kernel(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None
    ) -> Callable:
        """Create kernel for hidden state distillation."""
        if intermediate_size is None:
            intermediate_size = self.config.intermediate_size
            
        @jax.jit
        def hidden_kernel(
            student_hidden: jnp.ndarray,
            teacher_hidden: jnp.ndarray,
            student_intermediate: Optional[jnp.ndarray] = None,
            teacher_intermediate: Optional[jnp.ndarray] = None
        ) -> Tuple[jnp.ndarray, float]:
            # Hidden state loss
            hidden_loss = jnp.mean(
                jnp.square(
                    student_hidden / jnp.norm(student_hidden, axis=-1, keepdims=True) -
                    teacher_hidden / jnp.norm(teacher_hidden, axis=-1, keepdims=True)
                )
            )
            
            # Add intermediate layer loss if provided
            if (student_intermediate is not None and 
                teacher_intermediate is not None):
                intermediate_loss = jnp.mean(
                    jnp.square(
                        student_intermediate / jnp.norm(student_intermediate, axis=-1, keepdims=True) -
                        teacher_intermediate / jnp.norm(teacher_intermediate, axis=-1, keepdims=True)
                    )
                )
                hidden_loss = 0.5 * (hidden_loss + intermediate_loss)
                
            return student_hidden, hidden_loss
            
        return hidden_kernel
        
    def create_prediction_distillation_kernel(
        self,
        vocab_size: Optional[int] = None
    ) -> Callable:
        """Create kernel for prediction layer distillation."""
        if vocab_size is None:
            vocab_size = self.config.vocab_size
            
        @jax.jit
        def prediction_kernel(
            student_logits: jnp.ndarray,
            teacher_logits: jnp.ndarray,
            labels: Optional[jnp.ndarray] = None
        ) -> Tuple[jnp.ndarray, float]:
            # Compute soft targets from teacher
            teacher_probs = jax.nn.softmax(
                teacher_logits / self.config.temperature,
                axis=-1
            )
            
            # Distillation loss
            prediction_loss = -jnp.mean(
                jnp.sum(
                    teacher_probs * jax.nn.log_softmax(
                        student_logits / self.config.temperature,
                        axis=-1
                    ),
                    axis=-1
                )
            )
            
            # Add hard target loss if labels provided
            if labels is not None:
                hard_loss = jax.nn.sparse_categorical_crossentropy(
                    labels,
                    student_logits
                )
                prediction_loss = (
                    self.config.alpha * prediction_loss +
                    (1 - self.config.alpha) * hard_loss
                )
                
            return student_logits, prediction_loss
            
        return prediction_kernel
        
    def distill_batch(
        self,
        student_outputs: Dict[str, jnp.ndarray],
        teacher_outputs: Dict[str, jnp.ndarray],
        labels: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> DistillationOutput:
        """Perform distillation on a batch of inputs."""
        # Create kernels
        attention_kernel = self.create_attention_distillation_kernel(
            self.config.num_heads,
            self.config.hidden_size // self.config.num_heads
        )
        hidden_kernel = self.create_hidden_distillation_kernel(
            self.config.hidden_size
        )
        prediction_kernel = self.create_prediction_distillation_kernel()
        
        # Attention distillation
        _, attention_loss = attention_kernel(
            student_outputs["query"],
            student_outputs["key"],  
            student_outputs["value"],
            teacher_outputs["query"],
            teacher_outputs["key"],
            teacher_outputs["value"],
            attention_mask
        )
        
        # Hidden state distillation
        _, hidden_loss = hidden_kernel(
            student_outputs["hidden_states"],
            teacher_outputs["hidden_states"],
            student_outputs.get("intermediate"),
            teacher_outputs.get("intermediate")
        )
        
        # Prediction distillation
        student_logits, prediction_loss = prediction_kernel(
            student_outputs["logits"],
            teacher_outputs["logits"],
            labels
        )
        
        # Combine losses
        total_loss = (
            attention_loss +
            hidden_loss +
            prediction_loss
        )
        
        return DistillationOutput(
            student_logits=student_logits,
            teacher_logits=teacher_outputs["logits"],
            attention_loss=attention_loss,
            hidden_loss=hidden_loss,
            prediction_loss=prediction_loss,
            total_loss=total_loss
        )
