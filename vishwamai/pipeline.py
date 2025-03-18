"""
Training pipeline and utilities for VishwamAI transformer.
"""

import jax
import jax.numpy as jnp
import optax
from typing import Any, Dict, Optional, Tuple, Callable, Iterator
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
from vishwamai.thoughts.cot import ChainOfThoughtPrompting
from vishwamai.thoughts.tot import TreeOfThoughts
import flax

"""TPU-optimized data pipeline"""

import tensorflow as tf
import numpy as np

class TPUDataPipeline:
    """Data pipeline optimized for TPU training."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        devices: Optional[Any] = None
    ):
        self.config = config
        self.batch_size = config['training']['batch_size']
        self.block_size = config['optimization']['block_size']
        self.grad_accum_steps = config['training']['grad_accum_steps']
        self.devices = devices or jax.devices()
        self.num_devices = len(self.devices)
        
        # Compute global batch size
        self.global_batch_size = (
            self.batch_size * 
            self.grad_accum_steps * 
            self.num_devices
        )
        
        # Set up TF data pipeline
        tf.config.set_visible_devices([], 'GPU')  # Prevent TF from using GPU
        
    def create_dataset(
        self,
        file_pattern: str,
        is_training: bool = True
    ) -> tf.data.Dataset:
        """Create optimized dataset for TPU training."""
        
        files = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)
        
        def parse_example(example):
            features = {
                'input_ids': tf.io.FixedLenFeature([self.block_size], tf.int64),
                'labels': tf.io.FixedLenFeature([self.block_size], tf.int64),
            }
            
            # For distillation
            if self.config.get('distillation'):
                features['teacher_logits'] = tf.io.FixedLenFeature(
                    [self.block_size * self.config['model']['vocab_size']], 
                    tf.float32
                )
            
            parsed = tf.io.parse_single_example(example, features)
            
            # Cast to optimal TPU dtype
            if self.config['tpu']['use_bfloat16']:
                for k, v in parsed.items():
                    if v.dtype == tf.float32:
                        parsed[k] = tf.cast(v, tf.bfloat16)
            
            return parsed
        
        def read_tfrecord(filename):
            dataset = tf.data.TFRecordDataset(
                filename,
                compression_type='GZIP',
                buffer_size=self.block_size * 1024,  # 128KB buffer
                num_parallel_reads=tf.data.AUTOTUNE
            )
            return dataset
        
        # Create dataset pipeline
        dataset = files.interleave(
            read_tfrecord,
            cycle_length=tf.data.AUTOTUNE,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=not is_training
        )
        
        # Parse examples
        dataset = dataset.map(
            parse_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            # Shuffle before batching
            dataset = dataset.shuffle(
                self.global_batch_size * 10,
                reshuffle_each_iteration=True
            )
        
        # Optimize batch size for TPU
        dataset = dataset.batch(
            self.global_batch_size,
            drop_remainder=is_training
        )
        
        # Prefetch to device
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def preprocess_batch(
        self,
        batch: Dict[str, tf.Tensor]
    ) -> Dict[str, jnp.ndarray]:
        """Preprocess batch for TPU training."""
        # Convert to JAX arrays
        jax_batch = {
            k: jnp.array(v) 
            for k, v in batch.items()
        }
        
        # Reshape for devices and grad accumulation
        def reshape_for_devices(x):
            return x.reshape(
                self.num_devices,
                self.grad_accum_steps,
                self.batch_size,
                *x.shape[1:]
            )
        
        jax_batch = {
            k: reshape_for_devices(v)
            for k, v in jax_batch.items()
        }
        
        return jax_batch
    
    def get_training_iter(
        self,
        dataset: tf.data.Dataset
    ) -> Iterator[Dict[str, jnp.ndarray]]:
        """Get iterator for TPU training."""
        
        for batch in dataset.as_numpy_iterator():
            # Preprocess batch
            jax_batch = self.preprocess_batch(batch)
            
            # Split across devices
            device_batch = {
                k: jax.device_put_sharded(
                    list(v), 
                    self.devices
                )
                for k, v in jax_batch.items()
            }
            
            yield device_batch

def create_tpu_data_pipeline(config: Dict[str, Any]) -> TPUDataPipeline:
    """Create TPU-optimized data pipeline."""
    # Get TPU devices
    if config['tpu']['device_strategy'] == 'data_parallel':
        devices = jax.devices()
    else:
        # Set up custom device mesh if needed
        devices = jax.devices()[:config['tpu']['tpu_cores']]
    
    return TPUDataPipeline(config, devices)

class DistillationDataPipeline(TPUDataPipeline):
    """Data pipeline optimized for knowledge distillation on TPU."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        teacher_model: Any,
        devices: Optional[Any] = None
    ):
        super().__init__(config, devices)
        self.teacher_model = teacher_model
        
    def generate_teacher_logits(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        temperature: float = 2.0
    ) -> jnp.ndarray:
        """Generate teacher logits in parallel on TPU."""
        # Split input across devices
        device_inputs = jax.device_put_sharded(
            list(input_ids), 
            self.devices
        )
        
        if attention_mask is not None:
            device_masks = jax.device_put_sharded(
                list(attention_mask),
                self.devices
            )
        else:
            device_masks = None
        
        # Define parallel forward pass
        def forward_pass(x, mask=None):
            logits = self.teacher_model(
                x,
                attention_mask=mask,
                training=False
            )
            return logits / temperature
        
        # Run parallel forward passes
        p_forward = jax.pmap(forward_pass, axis_name='batch')
        logits = p_forward(device_inputs, device_masks)
        
        # Gather results
        return jax.device_get(logits)
    
    def create_distillation_dataset(
        self,
        file_pattern: str,
        is_training: bool = True,
        cache_teacher_outputs: bool = True
    ) -> tf.data.Dataset:
        """Create dataset for distillation with teacher outputs."""
        base_dataset = super().create_dataset(
            file_pattern,
            is_training
        )
        
        if not cache_teacher_outputs:
            # Generate teacher logits on the fly
            def add_teacher_outputs(batch):
                input_ids = batch['input_ids']
                attention_mask = batch.get('attention_mask')
                
                teacher_logits = self.generate_teacher_logits(
                    input_ids,
                    attention_mask,
                    temperature=self.config['distillation']['temperature']
                )
                
                batch['teacher_logits'] = teacher_logits
                return batch
                
            base_dataset = base_dataset.map(
                add_teacher_outputs,
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        return base_dataset

def prepare_distillation_data(
    config: Dict[str, Any],
    teacher_model: Any,
    train_files: str,
    eval_files: str
) -> Tuple[TPUDataPipeline, tf.data.Dataset, tf.data.Dataset]:
    """Prepare data pipeline and datasets for distillation."""
    
    pipeline = DistillationDataPipeline(
        config,
        teacher_model
    )
    
    train_dataset = pipeline.create_distillation_dataset(
        train_files,
        is_training=True
    )
    
    eval_dataset = pipeline.create_distillation_dataset(
        eval_files,
        is_training=False
    )
    
    return pipeline, train_dataset, eval_dataset

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