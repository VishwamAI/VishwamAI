"""JAX-based trainer implementation optimized for TPU."""
from typing import Any, Dict, Optional, Tuple, Callable
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.training import train_state
import optax
import orbax
from flax.training.common_utils import shard_batch
from flax.training.dynamic_scale import DynamicScale

from .callbacks import CheckpointCallback
from ..utils import MetricLogger, DistributedLogger

class TrainState(train_state.TrainState):
    """Custom train state with dynamic scaling for mixed precision."""
    dynamic_scale: Optional[DynamicScale]

class JAXTrainer:
    """JAX-based trainer optimized for TPU training.
    
    Implements efficient data-parallel training with optional model parallelism
    for large models. Supports mixed precision and gradient accumulation.
    
    Args:
        model: Flax model
        learning_rate: Base learning rate
        warmup_steps: Number of warmup steps
        max_steps: Maximum training steps
        weight_decay: Weight decay factor
        grad_accum_steps: Gradient accumulation steps
        max_grad_norm: Maximum gradient norm for clipping
        use_bf16: Whether to use bfloat16 precision
        num_train_epochs: Number of training epochs
        callbacks: Optional training callbacks
        logger: Optional metric logger
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 1,
        max_grad_norm: Optional[float] = 1.0,
        use_bf16: bool = True,
        num_train_epochs: Optional[int] = None,
        callbacks: Optional[list] = None,
        logger: Optional[Any] = None
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.weight_decay = weight_decay
        self.grad_accum_steps = grad_accum_steps
        self.max_grad_norm = max_grad_norm
        self.use_bf16 = use_bf16
        self.num_train_epochs = num_train_epochs
        self.callbacks = callbacks or []
        self.logger = logger
        
        # Initialize JAX random keys
        self.rng = jax.random.PRNGKey(0)
        
        # Setup TPU devices
        jax.distributed.initialize()
        self.num_devices = jax.device_count()
        self.device_mesh = self.create_device_mesh()
        
        # Create train state
        self.state = self.create_train_state()
        
    def create_device_mesh(self) -> Any:
        """Create TPU device mesh for optimal data/model parallelism."""
        devices = jax.devices()
        n = int(len(devices) ** 0.5)
        return jax.sharding.Mesh(
            devices, ('data', 'model')
        ).reshape((n, n))
        
    def create_train_state(self) -> TrainState:
        """Initialize training state with optimizer and schedules."""
        # Create learning rate schedule
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.learning_rate,
            warmup_steps=self.warmup_steps,
            decay_steps=self.max_steps,
        )
        
        # Create optimizer
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm) if self.max_grad_norm else optax.identity(),
            optax.adamw(
                learning_rate=lr_schedule,
                weight_decay=self.weight_decay,
                b1=0.9,
                b2=0.999,
                eps=1e-8
            )
        )
        
        # Initialize model
        rng, init_rng = jax.random.split(self.rng)
        dummy_input = jnp.ones((2, 32))  # Adjust shape based on your model
        variables = self.model.init(init_rng, dummy_input, train=False)
        
        # Create train state
        state = TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=optimizer,
            dynamic_scale=DynamicScale() if self.use_bf16 else None
        )
        
        # Replicate across devices
        state = flax.jax_utils.replicate(state)
        return state
        
    @staticmethod
    def compute_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
        """Compute cross entropy loss."""
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, labels
        ).mean()
        
    def train_step(
        self,
        state: TrainState,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[TrainState, Dict[str, float]]:
        """Single training step."""
        metrics = {}
        
        # Training function
        def loss_fn(params):
            outputs = self.model.apply(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            loss = self.compute_loss(outputs, batch['labels'])
            return loss, outputs
            
        # Get loss and gradients
        if state.dynamic_scale:
            # Mixed precision training
            dynamic_scale, is_finite, loss, grads = state.dynamic_scale.value_and_grad(
                loss_fn, has_aux=True
            )(state.params)
            
            # Handle non-finite gradients
            state = state.replace(dynamic_scale=dynamic_scale)
            metrics['is_finite'] = is_finite
            # Skip step if NaN/Inf
            state = jax.lax.cond(
                is_finite,
                lambda _: state.apply_gradients(grads=grads),
                lambda _: state,
                None
            )
        else:
            # Regular training
            loss, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            
        metrics['loss'] = loss
        return state, metrics
        
    def train_epoch(self, train_loader: Any, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        metrics = []
        
        for step, batch in enumerate(train_loader):
            # Generate dropout RNG
            dropout_rng = jax.random.fold_in(self.rng, step)
            
            # Shard batch across devices
            batch = shard_batch(batch)
            
            # Perform training step
            self.state, step_metrics = self.train_step(
                self.state, batch, dropout_rng
            )
            
            metrics.append(step_metrics)
            
            # Log metrics
            if step % 100 == 0:
                avg_metrics = jax.tree_map(lambda x: x.mean(), metrics[-100:])
                if self.logger:
                    self.logger.log_metrics(
                        avg_metrics, 
                        step=step + epoch * len(train_loader)
                    )
                    
        # Average metrics across steps
        epoch_metrics = jax.tree_map(lambda x: x.mean(), metrics)
        return epoch_metrics
        
    def train(
        self,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        resume_from: Optional[str] = None
    ) -> Dict[str, list]:
        """Full training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)
            
        history = {'train': [], 'val': []}
        
        for epoch in range(self.num_train_epochs):
            # Training
            self.model.train()
            train_metrics = self.train_epoch(train_loader, epoch)
            history['train'].append(train_metrics)
            
            # Validation
            if val_loader is not None:
                self.model.eval()
                val_metrics = self.evaluate(val_loader)
                history['val'].append(val_metrics)
                
            # Run callbacks
            for callback in self.callbacks:
                callback(
                    epoch=epoch,
                    metrics={**train_metrics, **val_metrics} if val_loader else train_metrics
                )
                
        return history
        
    def evaluate(self, val_loader: Any) -> Dict[str, float]:
        """Evaluate model on validation set."""
        metrics = []
        
        for batch in val_loader:
            # Shard batch
            batch = shard_batch(batch)
            
            # Forward pass
            outputs = self.model.apply(
                {'params': self.state.params},
                batch['input_ids'],
                deterministic=True
            )
            
            # Compute metrics
            loss = self.compute_loss(outputs, batch['labels'])
            metrics.append({'loss': loss})
            
        # Average metrics
        avg_metrics = jax.tree_map(lambda x: x.mean(), metrics)
        return avg_metrics
        
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax.checkpoint.SaveArgs(
            temporary_checkpoints=True,
            only_save_on_chief=True
        )
        
        state_dict = {
            'model': self.state.params,
            'optimizer': self.state.opt_state,
            'step': self.state.step
        }
        
        orbax_checkpointer.save(
            path,
            state_dict,
            save_args=save_args
        )
        
    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        state_dict = orbax_checkpointer.restore(path)
        
        self.state = self.state.replace(
            params=state_dict['model'],
            opt_state=state_dict['optimizer'],
            step=state_dict['step']
        )
