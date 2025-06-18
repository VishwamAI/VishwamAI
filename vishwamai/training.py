"""
Training utilities and curriculum learning for VishwamAI.

Implements efficient training strategies, curriculum learning,
and parameter-efficient fine-tuning methods like LoRA.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from typing import Dict, Any, Optional, Callable, List, Tuple
import chex
from dataclasses import dataclass
import time
import json
import os

from .model import VishwamAIModel, ModelConfig


@dataclass
class TrainingConfig:
    """Configuration for training VishwamAI models."""
    
    # Model configuration
    model_config: ModelConfig
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    gradient_clip_norm: float = 1.0
    
    # Training schedule
    warmup_steps: int = 1000
    total_steps: int = 100000
    save_every: int = 1000
    eval_every: int = 500
    log_every: int = 100
    
    # Batch settings
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 2048
    
    # Mixed precision
    use_bfloat16: bool = True
    loss_scale: float = 2.0**15
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = None
    
    # LoRA settings
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    lora_modules: List[str] = None
    
    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    keep_top_k: int = 3
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = [
                {"name": "simple", "steps": 10000, "max_seq_len": 512},
                {"name": "medium", "steps": 30000, "max_seq_len": 1024},
                {"name": "complex", "steps": 60000, "max_seq_len": 2048},
            ]
        
        if self.lora_modules is None:
            self.lora_modules = ["attention", "feed_forward"]


class LoRALayer(nn.Module):
    """Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning."""
    
    features: int
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    
    def setup(self):
        # Low-rank matrices
        self.lora_A = self.param(
            'lora_A',
            nn.initializers.normal(stddev=0.02),
            (self.features, self.rank)
        )
        self.lora_B = self.param(
            'lora_B',
            nn.initializers.zeros,
            (self.rank, self.features)
        )
        
        self.dropout_layer = nn.Dropout(rate=self.dropout)
        self.scaling = self.alpha / self.rank
    
    def __call__(self, x: chex.Array, training: bool = True) -> chex.Array:
        """Apply LoRA adaptation."""
        
        # Apply dropout to input
        if training:
            x = self.dropout_layer(x, deterministic=False)
        
        # Low-rank transformation: x @ A @ B
        h = jnp.dot(x, self.lora_A)
        output = jnp.dot(h, self.lora_B)
        
        return output * self.scaling


class CurriculumTrainer:
    """Trainer with curriculum learning support."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.current_stage = 0
        self.stage_step = 0
        self.global_step = 0
        
        # Create model
        self.model = VishwamAIModel(config.model_config)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.state = None
        
        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'step_time': [],
        }
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with warmup and scheduling."""
        
        # Learning rate schedule
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            decay_steps=self.config.total_steps - self.config.warmup_steps,
            end_value=self.config.learning_rate * 0.1
        )
        
        # Optimizer chain
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip_norm),
            optax.adamw(
                learning_rate=schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.epsilon,
                weight_decay=self.config.weight_decay
            )
        )
        
        # Mixed precision wrapper
        if self.config.use_bfloat16:
            optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=5)
        
        return optimizer
    
    def initialize_state(self, rng_key: jax.random.PRNGKey) -> train_state.TrainState:
        """Initialize training state."""
        
        # Dummy input for initialization
        dummy_input = jnp.ones((1, self.config.max_seq_len), dtype=jnp.int32)
        
        # Initialize parameters
        params = self.model.init(rng_key, dummy_input, training=True)
        
        # Create training state
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.optimizer
        )
        
        return self.state
    
    def get_current_curriculum_config(self) -> Dict[str, Any]:
        """Get current curriculum stage configuration."""
        
        if not self.config.use_curriculum:
            return {"max_seq_len": self.config.max_seq_len}
        
        if self.current_stage >= len(self.config.curriculum_stages):
            return self.config.curriculum_stages[-1]
        
        return self.config.curriculum_stages[self.current_stage]
    
    def update_curriculum_stage(self):
        """Update curriculum stage if needed."""
        
        if not self.config.use_curriculum:
            return
        
        current_config = self.get_current_curriculum_config()
        
        # Check if we should advance to next stage
        if (self.stage_step >= current_config.get("steps", float('inf')) and 
            self.current_stage < len(self.config.curriculum_stages) - 1):
            
            self.current_stage += 1
            self.stage_step = 0
            
            new_config = self.get_current_curriculum_config()
            print(f"Advanced to curriculum stage {self.current_stage}: {new_config['name']}")
    
    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, chex.Array],
        rng_key: jax.random.PRNGKey
    ) -> Tuple[train_state.TrainState, Dict[str, Any]]:
        """Single training step."""
        
        def loss_fn(params):
            # Forward pass
            logits = state.apply_fn(
                params,
                batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                training=True,
                rngs={'dropout': rng_key}
            )
            
            # Compute loss (next token prediction)
            targets = batch['input_ids'][:, 1:]  # Shift targets
            logits = logits[:, :-1, :]  # Align logits
            
            # Cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, targets
            ).mean()
            
            return loss, logits
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        
        # Update parameters
        state = state.apply_gradients(grads=grads)
        
        # Compute metrics
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == batch['input_ids'][:, 1:])
        
        metrics = {
            'loss': loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.learning_rate if hasattr(self.optimizer, 'learning_rate') else 0.0
        }
        
        return state, metrics
    
    def eval_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, chex.Array]
    ) -> Dict[str, Any]:
        """Single evaluation step."""
        
        # Forward pass (no dropout)
        logits = state.apply_fn(
            state.params,
            batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            training=False
        )
        
        # Compute loss
        targets = batch['input_ids'][:, 1:]
        logits = logits[:, :-1, :]
        
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, targets
        ).mean()
        
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == targets)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def train(
        self,
        train_loader: Any,
        eval_loader: Optional[Any] = None,
        rng_key: jax.random.PRNGKey = None
    ):
        """Main training loop."""
        
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        
        if self.state is None:
            self.state = self.initialize_state(rng_key)
        
        # JIT compile training functions
        train_step_jit = jax.jit(self.train_step)
        eval_step_jit = jax.jit(self.eval_step) if eval_loader else None
        
        print(f"Starting training for {self.config.total_steps} steps")
        
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            if step >= self.config.total_steps:
                break
            
            # Update curriculum if needed
            self.update_curriculum_stage()
            
            # Training step
            step_start = time.time()
            rng_key, step_key = jax.random.split(rng_key)
            
            self.state, train_metrics = train_step_jit(self.state, batch, step_key)
            
            step_time = time.time() - step_start
            
            # Update counters
            self.global_step = step
            self.stage_step += 1
            
            # Log metrics
            if step % self.config.log_every == 0:
                self._log_metrics(train_metrics, step_time, step)
            
            # Evaluation
            if eval_loader and step % self.config.eval_every == 0:
                eval_metrics = self._evaluate(eval_step_jit, eval_loader)
                self._log_eval_metrics(eval_metrics, step)
            
            # Save checkpoint
            if step % self.config.save_every == 0:
                self._save_checkpoint(step)
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f} seconds")
    
    def _log_metrics(self, metrics: Dict[str, Any], step_time: float, step: int):
        """Log training metrics."""
        
        self.metrics['train_loss'].append(float(metrics['loss']))
        self.metrics['step_time'].append(step_time)
        
        curriculum_config = self.get_current_curriculum_config()
        
        print(f"Step {step:6d} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f} | "
              f"LR: {metrics['learning_rate']:.2e} | "
              f"Time: {step_time:.3f}s | "
              f"Stage: {curriculum_config.get('name', 'default')}")
    
    def _evaluate(self, eval_step_fn: Callable, eval_loader: Any) -> Dict[str, Any]:
        """Run evaluation loop."""
        
        eval_metrics = []
        
        for eval_batch in eval_loader:
            batch_metrics = eval_step_fn(self.state, eval_batch)
            eval_metrics.append(batch_metrics)
        
        # Average metrics
        avg_metrics = {}
        for key in eval_metrics[0].keys():
            avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in eval_metrics]))
        
        return avg_metrics
    
    def _log_eval_metrics(self, metrics: Dict[str, Any], step: int):
        """Log evaluation metrics."""
        
        self.metrics['eval_loss'].append(float(metrics['loss']))
        
        print(f"Eval {step:6d} | "
              f"Loss: {metrics['loss']:.4f} | "
              f"Acc: {metrics['accuracy']:.4f}")
    
    def _save_checkpoint(self, step: int):
        """Save model checkpoint."""
        
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{step}")
        
        # Create directory if it doesn't exist
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save state
        checkpoints.save_checkpoint(
            ckpt_dir=checkpoint_path,
            target=self.state,
            step=step,
            overwrite=True,
            keep=self.config.keep_top_k
        )
        
        # Save config
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, 'w') as f:
            # Convert config to dict for JSON serialization
            config_dict = {
                'model_config': self.config.model_config.__dict__,
                'training_config': {
                    k: v for k, v in self.config.__dict__.items()
                    if k != 'model_config' and not callable(v)
                }
            }
            json.dump(config_dict, f, indent=2)
        
        print(f"Saved checkpoint at step {step}")
    
    def load_checkpoint(self, checkpoint_path: str) -> train_state.TrainState:
        """Load model checkpoint."""
        
        self.state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_path,
            target=self.state
        )
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return self.state


def create_optimizer(
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    total_steps: int = 100000,
    gradient_clip_norm: float = 1.0
) -> optax.GradientTransformation:
    """Create optimizer with standard settings."""
    
    # Learning rate schedule
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps - warmup_steps,
        end_value=learning_rate * 0.1
    )
    
    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=weight_decay,
            b1=0.9,
            b2=0.95
        )
    )
    
    return optimizer


def setup_mixed_precision() -> Dict[str, Any]:
    """Setup mixed precision training configuration."""
    
    # Detect hardware capabilities
    devices = jax.devices()
    has_tpu = any(d.platform == "tpu" for d in devices)
    has_gpu = any(d.platform == "gpu" for d in devices)
    
    config = {
        'use_bfloat16': has_tpu,  # TPUs prefer bfloat16
        'use_fp16': has_gpu and not has_tpu,  # GPUs can use fp16
        'loss_scale': 2.0**15 if has_gpu else 1.0,
    }
    
    return config
