"""
TPU-optimized modeling utilities for VishwamAI.
"""
from typing import Dict, Optional, Any, Tuple, List, Union, Callable
import os
import json
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from flax import linen as nn
from flax.training import train_state
import optax
from functools import partial
import numpy as np
from google.cloud import storage
import logging

logger = logging.getLogger(__name__)

class ModelState:
    """TPU-optimized model state management."""
    def __init__(
        self,
        apply_fn: Callable,
        params: Dict,
        tx: Optional[optax.GradientTransformation] = None,
        rngs: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
        device_mesh: Optional[Any] = None
    ):
        self.apply_fn = apply_fn
        self.params = params
        self.tx = tx
        self.opt_state = tx.init(params) if tx is not None else None
        self.rngs = rngs or {}
        self.model_config = model_config or {}
        self.device_mesh = device_mesh
        self.step = 0
        self.best_metrics = {}
        self.dtype = jnp.bfloat16  # Default to bfloat16 for TPU

    @classmethod
    def create(
        cls,
        model: nn.Module,
        rng: jnp.ndarray,
        input_shape: Tuple,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        model_config: Optional[Dict] = None,
        use_bfloat16: bool = True,
        use_dualpipe: bool = True
    ) -> 'ModelState':
        """Create model state optimized for TPU."""
        # Initialize model
        dtype = jnp.bfloat16 if use_bfloat16 else jnp.float32
        variables = model.init(rng, jnp.ones(input_shape, dtype=dtype))
        
        # Create optimizer
        tx = create_optimizer(learning_rate, weight_decay)
        
        # Create device mesh for TPU
        device_mesh = create_device_mesh()
        
        return cls(
            apply_fn=model.apply,
            params=variables['params'],
            tx=tx,
            rngs={'dropout': rng},
            model_config=model_config,
            device_mesh=device_mesh
        )

    def apply_gradients(self, gradients: Dict) -> 'ModelState':
        """Apply gradients with TPU optimization."""
        updates, new_opt_state = self.tx.update(
            gradients, self.opt_state, self.params
        )
        new_params = optax.apply_updates(self.params, updates)
        
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state
        )

    def save_checkpoint(
        self,
        save_dir: str,
        keep: int = 2,
        overwrite: bool = False
    ) -> None:
        """Save checkpoint with TPU memory optimization."""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'step': self.step,
            'params': self.params,
            'opt_state': self.opt_state,
            'rngs': self.rngs,
            'best_metrics': self.best_metrics
        }
        
        if save_dir.startswith('gs://'):
            # Save to Google Cloud Storage
            client = storage.Client()
            bucket_name = save_dir.split('/')[2]
            bucket = client.get_bucket(bucket_name)
            
            # Save in chunks for large models
            chunk_size = 100 * 1024 * 1024  # 100MB chunks
            for i, (key, value) in enumerate(checkpoint.items()):
                blob = bucket.blob(f'checkpoint_{self.step}/chunk_{i}.npy')
                with blob.open('wb') as f:
                    np.save(f, value)
        else:
            # Local save with memory optimization
            save_path = os.path.join(save_dir, f'checkpoint_{self.step}.npy')
            with open(save_path, 'wb') as f:
                np.save(f, checkpoint)
        
        # Remove old checkpoints
        if keep > 0:
            checkpoints = sorted([
                f for f in os.listdir(save_dir)
                if f.startswith('checkpoint_')
            ])
            for old_ckpt in checkpoints[:-keep]:
                old_path = os.path.join(save_dir, old_ckpt)
                os.remove(old_path)

    @classmethod
    def load_checkpoint(
        cls,
        load_dir: str,
        step: Optional[int] = None,
        map_location: Optional[str] = None
    ) -> 'ModelState':
        """Load checkpoint with TPU optimization."""
        if load_dir.startswith('gs://'):
            # Load from Google Cloud Storage
            client = storage.Client()
            bucket_name = load_dir.split('/')[2]
            bucket = client.get_bucket(bucket_name)
            
            # Find checkpoint
            prefix = f'checkpoint_{step}' if step is not None else 'checkpoint_'
            blobs = list(bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                raise ValueError(f"No checkpoint found in {load_dir}")
            
            # Load chunks
            checkpoint = {}
            for blob in blobs:
                with blob.open('rb') as f:
                    chunk = np.load(f, allow_pickle=True)
                    checkpoint.update(chunk.item())
        else:
            # Local load
            if step is not None:
                load_path = os.path.join(load_dir, f'checkpoint_{step}.npy')
            else:
                checkpoints = sorted([
                    f for f in os.listdir(load_dir)
                    if f.startswith('checkpoint_')
                ])
                if not checkpoints:
                    raise ValueError(f"No checkpoint found in {load_dir}")
                load_path = os.path.join(load_dir, checkpoints[-1])
            
            with open(load_path, 'rb') as f:
                checkpoint = np.load(f, allow_pickle=True).item()
        
        # Place on correct device
        if map_location == 'tpu':
            device = jax.devices('tpu')[0]
        else:
            device = jax.devices('cpu')[0]
        
        with jax.default_device(device):
            return cls(
                apply_fn=checkpoint['apply_fn'],
                params=checkpoint['params'],
                tx=checkpoint['tx'],
                rngs=checkpoint.get('rngs', {}),
                model_config=checkpoint.get('model_config', {}),
                device_mesh=create_device_mesh()
            )

    def replace(self, **kwargs) -> 'ModelState':
        """Create new state with updated attributes."""
        return ModelState(
            apply_fn=kwargs.get('apply_fn', self.apply_fn),
            params=kwargs.get('params', self.params),
            tx=kwargs.get('tx', self.tx),
            rngs=kwargs.get('rngs', self.rngs),
            model_config=kwargs.get('model_config', self.model_config),
            device_mesh=kwargs.get('device_mesh', self.device_mesh)
        )

def create_optimizer(
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    warmup_steps: int = 2000,
    total_steps: int = 100000
) -> optax.GradientTransformation:
    """Create TPU-optimized optimizer."""
    # Learning rate schedule
    schedule_fn = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=0.0
    )
    
    # Optimizer chain
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adamw(
            learning_rate=schedule_fn,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay
        )
    )

def create_device_mesh(
    mesh_shape: Optional[Tuple[int, ...]] = None
) -> Any:
    """Create TPU device mesh for efficient sharding."""
    if mesh_shape is None:
        num_devices = jax.device_count()
        if num_devices >= 8:
            # 2D mesh for 8+ devices
            mesh_shape = (num_devices // 4, 4)
        else:
            # 1D mesh for fewer devices
            mesh_shape = (num_devices,)
    
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return jax.sharding.Mesh(devices, ('batch', 'model'))

@partial(jax.jit, static_argnums=(1,))
def cast_to_compute_dtype(
    params: Dict,
    compute_dtype: jnp.dtype = jnp.bfloat16
) -> Dict:
    """Cast parameters to compute dtype with TPU optimization."""
    return jax.tree_map(lambda x: x.astype(compute_dtype), params)

def get_tpu_memory_usage() -> Dict[str, float]:
    """Get TPU memory usage statistics."""
    try:
        devices = jax.devices()
        memory_stats = {}
        for i, d in enumerate(devices):
            mem = d.memory_stats()
            memory_stats[f'device_{i}'] = {
                'used_bytes': mem.get('bytes_in_use', 0),
                'peak_bytes': mem.get('peak_bytes_in_use', 0),
                'available_bytes': mem.get('bytes_available', 0)
            }
        return memory_stats
    except:
        logger.warning("Failed to get TPU memory stats")
        return {}

def get_optimal_batch_size(
    model: nn.Module,
    seq_length: int,
    target_batch_size: int,
    dtype: jnp.dtype = jnp.bfloat16
) -> int:
    """Calculate optimal batch size for TPU memory."""
    try:
        # Get TPU memory info
        device = jax.devices('tpu')[0]
        available_mem = device.memory_stats()['bytes_available']
        
        # Estimate parameter memory
        param_size = sum(np.prod(p.shape) for p in jax.tree_leaves(model.params))
        param_bytes = param_size * dtype.itemsize
        
        # Estimate activation memory per sample
        sample_shape = (1, seq_length)
        sample_activations = model.apply(
            {'params': model.params},
            jnp.ones(sample_shape, dtype=dtype)
        )
        activation_bytes = sum(
            np.prod(a.shape) * dtype.itemsize
            for a in jax.tree_leaves(sample_activations)
        )
        
        # Calculate max batch size
        memory_per_sample = activation_bytes + (param_bytes / target_batch_size)
        max_batch_size = int(available_mem * 0.8 / memory_per_sample)  # 80% of memory
        
        # Round down to multiple of 8 for TPU efficiency
        optimal_batch_size = (max_batch_size // 8) * 8
        
        return min(optimal_batch_size, target_batch_size)
    
    except:
        logger.warning("Failed to calculate optimal batch size")
        return target_batch_size

class MetricsTracker:
    """TPU-optimized training metrics tracker."""
    def __init__(self, metrics_history_size: int = 100):
        self.metrics_history_size = metrics_history_size
        self.metrics_history = {}
        self.step_metrics = {}
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float], step: int) -> None:
        """Update metrics with TPU-optimized aggregation."""
        # Update step metrics
        self.step_metrics = metrics
        
        # Update history
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            
            history = self.metrics_history[name]
            history.append(value)
            
            # Keep fixed size window
            if len(history) > self.metrics_history_size:
                history.pop(0)
            
            # Update best metrics
            if name not in self.best_metrics or (
                'loss' in name and value < self.best_metrics[name] or
                'accuracy' in name and value > self.best_metrics[name]
            ):
                self.best_metrics[name] = value
    
    def get_smoothed_metrics(self, window_size: int = 10) -> Dict[str, float]:
        """Get smoothed metrics for TPU monitoring."""
        smoothed = {}
        for name, history in self.metrics_history.items():
            if len(history) >= window_size:
                smoothed[name] = sum(history[-window_size:]) / window_size
        return smoothed

def create_model_parallel_train_step(
    base_train_step: Callable,
    num_partitions: Optional[int] = None
) -> Callable:
    """Create TPU model-parallel training step."""
    if num_partitions is None:
        num_partitions = jax.device_count()
    
    def split_batch(batch):
        return jax.tree_map(
            lambda x: x.reshape((num_partitions, -1) + x.shape[1:]),
            batch
        )
    
    @partial(jax.pmap, axis_name='batch')
    def parallel_train_step(state, batch, rng):
        # Split batch across devices
        batch = split_batch(batch)
        
        # Run training step
        outputs, new_state = base_train_step(state, batch, rng)
        
        # All-reduce gradients
        outputs['gradients'] = jax.lax.pmean(
            outputs['gradients'],
            axis_name='batch'
        )
        
        return outputs, new_state
    
    return parallel_train_step

if __name__ == "__main__":
    # Example usage
    rng = jax.random.PRNGKey(0)
    
    # Create dummy model
    model = nn.Dense(features=64)
    batch_shape = (16, 32)  # (batch_size, seq_len)
    
    # Initialize state
    state = ModelState.create(
        model=model,
        rng=rng,
        input_shape=batch_shape,
        use_bfloat16=True,
        use_dualpipe=True
    )
    
    # Test checkpoint save/load
    state.save_checkpoint('/tmp/model_ckpts/')
    loaded_state = ModelState.load_checkpoint('/tmp/model_ckpts/')
    
    # Get memory stats
    memory_usage = get_tpu_memory_usage()
    logger.info(f"TPU memory usage: {memory_usage}")
    
    # Calculate optimal batch size
    optimal_bs = get_optimal_batch_size(model, seq_length=32, target_batch_size=16)
    logger.info(f"Optimal batch size: {optimal_bs}")
