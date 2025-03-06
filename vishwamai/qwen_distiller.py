"""QwQ-32B optimized distillation implementation."""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, Tuple
from flax.training import train_state
import optax
import gc

from .distillation import VishwamaiShaalaTrainer
from .layer_mapping import create_feature_mapping_config
from .loss_functions import compute_distillation_loss
from .tensor_utils import (
    get_memory_usage,
    chunk_tensor,
    merge_chunks,
    apply_chunked
)

class QwQDistillationTrainer(VishwamaiShaalaTrainer):
    """Specialized trainer for QwQ-32B distillation with memory optimization."""

    def __init__(self, teacher_model, student_model, cfg):
        super().__init__(teacher_model, student_model, cfg)
        
        # Initialize feature mappings
        self.feature_mapping = create_feature_mapping_config(
            student_config=self.cfg.distillation.student_model.config,
            teacher_config=self.cfg.distillation.teacher_model.config
        )
        
        # Get TPU-specific settings
        self.num_devices = jax.device_count()
        self.chunk_size = cfg.memory_optimization.chunk_size
        self.clear_cache_steps = cfg.memory_optimization.clear_cache_steps
        
        print(f"Initialized QwQ distillation trainer:"
              f"\n - Devices: {self.num_devices}"
              f"\n - Chunk size: {self.chunk_size}"
              f"\n - Initial memory: {get_memory_usage():.2f}GB")
    
    def _initialize_training_state(self, rng: jnp.ndarray) -> train_state.TrainState:
        """Initialize training state with gradient accumulation."""
        # Initialize student model
        dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
        init_variables = self.student_model.init(rng, dummy_input)
        
        # Create optimizer with accumulation
        steps_per_update = (
            self.cfg.training.gradient_accumulation_steps *
            self.num_devices
        )
        
        tx = optax.chain(
            optax.clip_by_global_norm(self.cfg.training.max_grad_norm),
            optax.adam(
                learning_rate=self.cfg.training.learning_rate,
                b1=0.9,
                b2=0.999
            ),
            optax.update_every(steps_per_update)
        )
        
        return train_state.TrainState.create(
            apply_fn=self.student_model.__call__,
            params=init_variables['params'],
            tx=tx,
        )

    def train_step_with_grads(
        self, 
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray
    ) -> Tuple[train_state.TrainState, Dict[str, Any], Dict[str, jnp.ndarray]]:
        """Training step with memory-efficient gradient computation."""
        
        def loss_fn(params):
            # Process input in chunks if needed
            def forward_chunk(input_chunk):
                return self.student_model.apply(
                    {'params': params},
                    input_chunk,
                    attention_mask=batch.get('attention_mask'),
                    output_hidden_states=True,
                    output_attentions=True,
                    deterministic=False,
                    rngs={'dropout': rng}
                )
            
            if batch['input_ids'].shape[0] > self.chunk_size:
                student_outputs = apply_chunked(
                    forward_chunk,
                    batch['input_ids'],
                    self.chunk_size
                )
            else:
                student_outputs = forward_chunk(batch['input_ids'])
            
            # Get teacher outputs with gradient stopping
            teacher_outputs = jax.lax.stop_gradient(
                self.teacher_model(
                    batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    output_hidden_states=True,
                    output_attentions=True,
                    deterministic=True
                )
            )
            
            # Compute loss
            loss, metrics = compute_distillation_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                layer_mapping=self.feature_mapping,
                temperature=self.cfg.distillation_params.temperature,
                alpha_ce=self.cfg.distillation_params.alpha_ce,
                alpha_kd=self.cfg.distillation_params.alpha_kd,
                alpha_features=self.cfg.distillation_params.feature_loss_weight,
                alpha_attention=self.cfg.distillation_params.attention_loss_weight
            )
            
            return loss, (metrics, student_outputs)
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, outputs)), grads = grad_fn(state.params)
        
        # Clear memory
        if state.step % self.clear_cache_steps == 0:
            jax.clear_caches()
            gc.collect()
        
        return state, metrics, grads

    def save_checkpoint(
        self,
        state: train_state.TrainState,
        path: str,
        keep_n: int = 3,
        **extra_info
    ) -> None:
        """Save checkpoint with memory efficiency."""
        try:
            # Save in chunks if needed
            if state.params['embedding'].size > self.chunk_size:
                chunked_params = chunk_tensor(
                    state.params['embedding'],
                    self.chunk_size
                )
                for i, chunk in enumerate(chunked_params):
                    chunk_path = f"{path}_chunk_{i}"
                    super().save_checkpoint(
                        state.replace(params={'embedding_chunk': chunk}),
                        chunk_path
                    )
            else:
                super().save_checkpoint(state, path, **extra_info)
                
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(path, keep_n)
            
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            print(f"Memory usage: {get_memory_usage():.2f}GB")
            raise

    def _cleanup_old_checkpoints(self, path: str, keep_n: int) -> None:
        """Remove old checkpoints while keeping most recent n."""
        import glob
        import os
        
        checkpoints = sorted(glob.glob(f"{path}*"))
        for ckpt in checkpoints[:-keep_n]:
            try:
                os.remove(ckpt)
            except OSError:
                pass

if __name__ == "__main__":
    # Test usage
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create({
        'memory_optimization': {
            'chunk_size': 32,
            'clear_cache_steps': 10
        },
        'training': {
            'gradient_accumulation_steps': 16,
            'learning_rate': 1e-4,
            'max_grad_norm': 1.0
        },
        'distillation_params': {
            'temperature': 2.0,
            'alpha_ce': 0.2,
            'alpha_kd': 0.8,
            'feature_loss_weight': 0.1,
            'attention_loss_weight': 0.1
        }
    })
    
    trainer = QwQDistillationTrainer(None, None, cfg)
    print("Trainer initialized successfully")
