"""
Training pipeline and utilities for VishwamAI transformer.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Any, Dict, Optional, Tuple, Callable
from functools import partial
from .transformer import (
    EnhancedTransformerModel,
    create_vishwamai_transformer,
    create_train_state
)
from .distill import (
    compute_distillation_loss,
    create_student_model,
    initialize_from_teacher
)
from .cot import ChainOfThoughtPrompting
from .tot import TreeOfThoughts
import flax
class VishwamAIPipeline:
    """Pipeline for training and inference with VishwamAI transformer."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        tokenizer: Any,
        model: Optional[Any] = None,
        teacher_model: Optional[Any] = None
    ):
        self.config = config
        self.tokenizer = tokenizer
        
        # Create or use provided model
        if model is None:
            self.model = create_vishwamai_transformer(config)
        else:
            self.model = model
            
        self.teacher_model = teacher_model
        
        # Initialize components
        self.cot = ChainOfThoughtPrompting(
            self.model,
            self.tokenizer,
            temperature=config.get('temperature', 0.7)
        )
        
        self.tot = TreeOfThoughts(
            self.model,
            self.tokenizer,
            temperature=config.get('temperature', 0.7),
            max_depth=config.get('tot_max_depth', 5),
            beam_width=config.get('tot_beam_width', 3)
        )
        
        # Training state
        self.state = None
        self.teacher_state = None
        
    def setup_training(
        self,
        learning_rate_schedule: Callable[[int], float],
        teacher_state: Optional[Any] = None
    ):
        """Setup training state and optimizer."""
        rng = jax.random.PRNGKey(self.config.get('seed', 42))
        
        if self.config.get('use_distillation', False) and teacher_state is not None:
            # Setup distillation training
            self.teacher_state = teacher_state
            self.state = create_train_state(
                rng,
                self.config,
                learning_rate_schedule
            )
            self.state = initialize_from_teacher(
                self.state,
                teacher_state,
                method=self.config.get('init_method', 'layer_random')
            )
        else:
            # Standard training
            self.state = create_train_state(
                rng,
                self.config,
                learning_rate_schedule
            )
    
    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Single training step."""
        
        if self.teacher_state is not None:
            # Distillation training step
            return self._distillation_train_step(
                state,
                self.teacher_state,
                batch,
                dropout_rng
            )
        else:
            # Standard training step
            return self._standard_train_step(
                state,
                batch,
                dropout_rng
            )
    
    def _standard_train_step(
        self,
        state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Standard training step without distillation."""
        
        def loss_fn(params):
            logits = state.apply_fn(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits,
                batch['labels']
            )
            return loss.mean(), logits
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        
        metrics = {
            'loss': loss,
            'learning_rate': state.opt_state.hyperparams['learning_rate']
        }
        
        return new_state, metrics
    
    def _distillation_train_step(
        self,
        state: Any,
        teacher_state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        """Training step with knowledge distillation."""
        
        def loss_fn(params):
            # Get student predictions
            student_logits = state.apply_fn(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            # Get teacher predictions
            teacher_logits = teacher_state.apply_fn(
                {'params': teacher_state.params},
                batch['input_ids'],
                deterministic=True
            )
            
            # Compute distillation loss
            loss, metrics = compute_distillation_loss(
                student_logits,
                teacher_logits,
                batch['labels'],
                temperature=self.config.get('temperature', 2.0),
                alpha=self.config.get('distill_alpha', 0.5)
            )
            
            return loss.mean(), (metrics, student_logits)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, _)), grads = grad_fn(state.params)
        
        new_state = state.apply_gradients(grads=grads)
        metrics['learning_rate'] = state.opt_state.hyperparams['learning_rate']
        
        return new_state, metrics
    
    @partial(jax.jit, static_argnums=(0,))
    def eval_step(
        self,
        state: Any,
        batch: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Evaluation step."""
        
        logits = state.apply_fn(
            {'params': state.params},
            batch['input_ids'],
            deterministic=True
        )
        
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits,
            batch['labels']
        )
        
        return {
            'loss': loss.mean(),
            'logits': logits
        }
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        mode: str = 'standard'
    ) -> Dict[str, Any]:
        """
        Generate text using specified mode.
        
        Args:
            prompt: Input prompt
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            mode: Generation mode ('standard', 'cot', or 'tot')
        """
        if mode == 'cot':
            return self.cot.reason(
                prompt,
                num_paths=self.config.get('num_reasoning_paths', 3)
            )
        elif mode == 'tot':
            return self.tot.reason(
                prompt,
                evaluation_criteria=self.config.get('tot_evaluation_criteria')
            )
        else:
            # Standard generation
            input_ids = self.tokenizer.encode(prompt)
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
            return {
                'text': self.tokenizer.decode(output[0]),
                'output_ids': output
            }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.state is not None:
            with open(path, 'wb') as f:
                f.write(flax.serialization.to_bytes(self.state))
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        if self.state is not None:
            with open(path, 'rb') as f:
                self.state = flax.serialization.from_bytes(
                    self.state,
                    f.read()
                )
        else:
            raise ValueError("Initialize training state before loading checkpoint")