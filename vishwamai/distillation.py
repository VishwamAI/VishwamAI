"""
Efficient knowledge distillation module for VishwamAI with TPU optimizations.
"""
import jax
import jax.numpy as jnp
import flax
import optax
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional
import logging
from functools import partial

from .model import VishwamAIModel, ModelConfig
from .loss_functions import cross_entropy_loss, kl_divergence_loss
from .error_correction import ErrorCorrectionTrainer

logger = logging.getLogger(__name__)

def _shard_batch(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Shard batch across TPU devices."""
    return jax.tree_map(
        lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]), 
        batch
    )

def _unshard_batch(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Un-shard batch from TPU devices."""
    return jax.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), 
        batch
    )

class TrainingState(train_state.TrainState):
    """Extended train state with EMA, metrics, and ToT state."""
    ema_params: Optional[Dict] = None
    best_metrics: Optional[Dict] = None
    tot_state: Optional[Dict] = None  # Tree of Thoughts state

def create_optimizer(config):
    """Create optimizer with learning rate schedule."""
    # Create learning rate schedule function that takes step as argument
    def learning_rate_fn(step):
        warmup_steps = config.training.warmup_steps
        decay_steps = config.training.max_steps
        base_learning_rate = config.training.learning_rate
        
        # Linear warmup
        warmup_factor = jnp.minimum(1.0, step / warmup_steps)
        
        # Cosine decay
        decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step, decay_steps) / decay_steps))
        
        return base_learning_rate * warmup_factor * decay_factor
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=config.training.adam_beta1,
            b2=config.training.adam_beta2,
            weight_decay=config.training.weight_decay
        )
    )
    
    return tx

@partial(jax.jit, static_argnums=(1, 2, 3))
def create_learning_rate_scheduler(
    step,
    base_learning_rate=0.0001,
    warmup_steps=1000,
    decay_steps=100000,
):
    """Creates a learning rate schedule with TPU optimization."""
    rate = base_learning_rate
    
    # Linear warmup
    warmup_factor = jnp.minimum(1.0, step / warmup_steps)
    rate *= warmup_factor
    
    # Cosine decay
    decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step, decay_steps) / decay_steps))
    rate *= decay_factor
    
    return rate

class VishwamaiGuruKnowledge:
    """Enhanced teacher model knowledge handler with TPU and EPLB support."""
    
    def __init__(self, config):
        self.config = config
        self.base_temperature = config.distillation.kd_temperature
        self.base_alpha_kd = config.distillation.alpha_kd
        self.base_alpha_ce = config.distillation.alpha_ce
        self.error_threshold = config.distillation.get('error_threshold', 0.1)
        # EPLB-inspired balancing
        self.eplb_window_size = config.distillation.get('eplb_window_size', 100)
        self.eplb_threshold = config.distillation.get('eplb_threshold', 0.8)

    @partial(jax.jit, static_argnums=(0,))
    def compute_expert_weights(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Compute EPLB-inspired expert weights."""
        # Compute softmax-based expert weights
        raw_weights = jax.nn.softmax(jnp.mean(logits, axis=1))
        # Balance experts using EPLB-like normalization
        balanced_weights = raw_weights / (jnp.sum(raw_weights) + 1e-6)
        return balanced_weights

    @partial(jax.jit, static_argnums=(0,))
    def distill(
        self,
        teacher_logits: jnp.ndarray,
        student_logits: jnp.ndarray,
        labels: jnp.ndarray,
        step: int,
        error_rate: Optional[float] = None,
        temperature: Optional[float] = None,
        use_error_correction: bool = False,
        corrected_teacher_logits: Optional[jnp.ndarray] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Apply advanced knowledge distillation with EPLB and TPU optimizations."""
        # Convert to bfloat16 for TPU efficiency
        teacher_logits = teacher_logits.astype(jnp.bfloat16)
        student_logits = student_logits.astype(jnp.bfloat16)
        
        # Compute expert weights using EPLB-inspired balancing
        expert_weights = self.compute_expert_weights(student_logits)
        
        # Adaptive Temperature with TPU optimization
        if temperature is None:
            temperature = self.base_temperature
            if error_rate is not None:
                temperature *= (1 + jnp.tanh(error_rate / self.error_threshold))

        # Dynamic Alpha Weighting
        progress = step / self.config.training.max_steps
        alpha_kd = self.base_alpha_kd * (1 - progress) + 0.1 * progress
        alpha_ce = self.base_alpha_ce * progress + 0.1 * (1 - progress)
        
        # Use corrected teacher logits if provided
        effective_teacher_logits = (
            corrected_teacher_logits if use_error_correction and corrected_teacher_logits is not None 
            else teacher_logits
        )
        
        # Apply expert balancing to logits
        balanced_student_logits = student_logits * expert_weights[:, None, None]
        
        # KL Divergence Loss with TPU optimization
        kd_loss = kl_divergence_loss(balanced_student_logits, effective_teacher_logits, temperature)
        
        # Cross-Entropy Loss
        mask = (labels != 0).astype(jnp.bfloat16)  # TPU-optimized dtype
        ce_loss = cross_entropy_loss(balanced_student_logits, labels)
        
        # Combined Loss
        combined_loss = alpha_kd * kd_loss + alpha_ce * ce_loss
        
        # EPLB metrics
        expert_usage = jnp.mean(expert_weights > self.eplb_threshold)
        expert_balance = 1.0 - jnp.std(expert_weights)
        
        # Metrics
        metrics = {
            "kd_loss": float(kd_loss),
            "ce_loss": float(ce_loss),
            "temperature": float(temperature),
            "alpha_kd": float(alpha_kd),
            "alpha_ce": float(alpha_ce),
            "expert_usage": float(expert_usage),
            "expert_balance": float(expert_balance)
        }
        
        return combined_loss, metrics

class VishwamaiShaalaTrainer:
    """Advanced trainer with TPU and EPLB optimizations."""
    
    def __init__(self, teacher_model, student_model, cfg):
        """Initialize distillation trainer."""
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.cfg = cfg
        self.guru = VishwamaiGuruKnowledge(cfg)
        
        # Initialize error correction and tot
        self.error_trainer = ErrorCorrectionTrainer(
            config=cfg,
            transformer=student_model,
            tokenizer=cfg.training.get('tokenizer'),
            use_tot=cfg.training.get('use_tot', True),
            use_mod=False
        )

    def create_train_state(self, rng: jax.random.PRNGKey) -> TrainingState:
        """Create initial training state with TPU optimizations."""
        # Create optimizer
        tx = create_optimizer(self.cfg)
        
        # TPU-optimized initialization
        if not hasattr(self.student_model, 'params') or self.student_model.params is None:
            dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
            init_rng = jax.random.split(rng, jax.device_count())
            params = jax.pmap(self.student_model.init)(init_rng, dummy_input)['params']
            self.student_model = self.student_model.bind({'params': params})
        
        # Initialize error correction parameters with TPU placement
        init_features = jnp.ones((1, 16, self.cfg.model.hidden_size), dtype=jnp.bfloat16)
        self.error_trainer.init_params(rng, init_features)
        
        # Create TrainingState replicated across devices
        state = TrainingState.create(
            apply_fn=self.student_model.__call__,
            params=self.student_model.params,
            tx=tx,
            ema_params=None,
            step=0,
            best_metrics={
                'loss': float('inf'),
                'accuracy': 0.0,
                'kd_loss': float('inf'),
                'ce_loss': float('inf')
            },
            tot_state={
                'enabled': self.cfg.training.get('use_tot', False),
                'search_strategy': self.cfg.training.get('tot_search_strategy', 'beam'),
                'thoughts_per_batch': 0,
                'best_thought_score': 0.0
            }
        )
        
        return jax.device_put_replicated(state, jax.local_devices())

    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def _pmapped_train_step(self, state: TrainingState, batch: Dict[str, jnp.ndarray], rng: jax.random.PRNGKey):
        """TPU-optimized training step implementation."""
        rng, dropout_rng, tot_rng = jax.random.split(rng, 3)
        
        def loss_fn(params):
            # Student forward pass with bfloat16
            student_outputs = state.apply_fn(
                {'params': params}, 
                batch['input_ids'].astype(jnp.int32),
                rngs={'dropout': dropout_rng}
            )
            student_logits = student_outputs['logits'].astype(jnp.bfloat16)
            
            # Teacher forward pass (frozen)
            teacher_outputs = jax.lax.stop_gradient(
                self.teacher_model(batch['input_ids'], deterministic=True)
            )
            teacher_logits = teacher_outputs['logits'].astype(jnp.bfloat16)
            
            # Error Correction with TPU optimization
            correction_outputs = self.error_trainer.apply_error_correction(
                logits=student_logits,
                features=student_outputs['hidden_states'].astype(jnp.bfloat16),
                labels=batch.get('labels'),
                training=True,
                rng_key=dropout_rng
            )
            corrected_student_logits = correction_outputs['corrected_logits']
            
            # ToT Enhancement (if enabled)
            tot_outputs = None
            if state.tot_state['enabled'] and jnp.mean(correction_outputs['error_probs']) > self.error_trainer.state.error_threshold:
                tot_outputs = self._handle_tot(teacher_outputs, tot_rng, batch)
                if tot_outputs is not None:
                    corrected_student_logits = self._blend_tot_outputs(
                        corrected_student_logits, 
                        tot_outputs, 
                        params
                    )
            
            # Distillation Loss with EPLB
            distill_loss, distill_metrics = self.guru.distill(
                teacher_logits=teacher_logits,
                student_logits=corrected_student_logits,
                labels=batch['labels'],
                step=state.step,
                error_rate=jnp.mean(correction_outputs['error_probs']),
                use_error_correction=True,
                corrected_teacher_logits=teacher_outputs.get('corrected_logits', teacher_logits)
            )
            
            return distill_loss, {
                'logits': corrected_student_logits,
                'hidden_states': correction_outputs['corrected_features'],
                'metrics': {
                    **distill_metrics,
                    'error_correction_rate': jnp.mean(correction_outputs['correction_mask'].astype(float)),
                    'tot_triggered': float(self.error_trainer.state.tot_triggered)
                }
            }
        
        # Compute gradients with TPU optimization
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # All-reduce gradients across devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        state = state.apply_gradients(grads=grads)
        
        # Update metrics and ToT state
        state = self._update_training_state(state, loss, aux)
        
        return {'loss': loss, **aux}, state

    def train_step(self, state: TrainingState, batch: Dict[str, jnp.ndarray], rng: jax.random.PRNGKey) -> Tuple[Dict, TrainingState]:
        """TPU-distributed training step."""
        # Shard batch across devices
        sharded_batch = _shard_batch(batch)
        sharded_rng = jax.random.split(rng, jax.device_count())
        
        # Run training step on all devices
        outputs, new_state = self._pmapped_train_step(state, sharded_batch, sharded_rng)
        
        # Combine results from all devices
        outputs = jax.device_get(outputs)
        return outputs, new_state

    @partial(jax.jit, static_argnums=(0,))
    def eval_step(self, state: TrainingState, batch: Dict[str, jnp.ndarray]) -> Dict:
        """TPU-optimized evaluation step."""
        # Forward passes with bfloat16
        student_outputs = state.apply_fn(
            {'params': state.params}, 
            batch['input_ids'], 
            deterministic=True
        )
        student_logits = student_outputs['logits'].astype(jnp.bfloat16)
        
        teacher_outputs = jax.lax.stop_gradient(
            self.teacher_model(batch['input_ids'], deterministic=True)
        )
        teacher_logits = teacher_outputs['logits'].astype(jnp.bfloat16)
        
        # Error Correction
        correction_outputs = self.error_trainer.apply_error_correction(
            logits=student_logits,
            features=student_outputs['hidden_states'].astype(jnp.bfloat16),
            labels=batch.get('labels'),
            training=False
        )
        corrected_student_logits = correction_outputs['corrected_logits']
        
        # Distillation Loss with EPLB
        distill_loss, distill_metrics = self.guru.distill(
            teacher_logits=teacher_logits,
            student_logits=corrected_student_logits,
            labels=batch['labels'],
            step=state.step,
            error_rate=jnp.mean(correction_outputs['error_probs']),
            use_error_correction=True,
            corrected_teacher_logits=teacher_outputs.get('corrected_logits', teacher_logits)
        )
        
        # Metrics with expert balancing info
        metrics = {
            **distill_metrics,
            'error_correction_rate': jnp.mean(correction_outputs['correction_mask'].astype(float))
        }
        
        return {
            'loss': distill_loss,
            'logits': corrected_student_logits,
            'hidden_states': correction_outputs['corrected_features'],
            'metrics': metrics
        }

# Example Usage
if __name__ == "__main__":
    from omegaconf import OmegaConf
    from vishwamai.tokenizer import VishwamAITokenizer
    
    # Dummy config with TPU settings
    cfg = OmegaConf.create({
        'training': {
            'learning_rate': 1e-4,
            'warmup_steps': 1000,
            'max_steps': 10000,
            'max_grad_norm': 1.0,
            'adam_beta1': 0.9,
            'adam_beta2': 0.999,
            'weight_decay': 0.01,
            'use_tot': True,
            'tot_search_strategy': 'beam',
            'use_tpu': True,
            'tpu_cores': 8
        },
        'distillation': {
            'kd_temperature': 2.0,
            'alpha_kd': 0.7,
            'alpha_ce': 0.3,
            'error_threshold': 0.1,
            'eplb_window_size': 100,
            'eplb_threshold': 0.8
        },
        'model': {
            'hidden_size': 512,
            'use_bfloat16': True
        }
    })
    
    # Initialize models
    teacher_config = ModelConfig(hidden_size=512, num_layers=6, num_attention_heads=8, vocab_size=32000)
    student_config = ModelConfig(hidden_size=256, num_layers=4, num_attention_heads=4, vocab_size=32000)
    
    # Place models on TPU
    devices = jax.devices("tpu")
    with jax.default_device(devices[0]):
        teacher_model = VishwamAIModel(teacher_config)
        student_model = VishwamAIModel(student_config)
    
    tokenizer = VishwamAITokenizer(vocab_size=32000)
    tokenizer.train(["dataset.txt"], "tokenizer_output")
    cfg.training.tokenizer = tokenizer
    
    # Initialize trainer
    trainer = VishwamaiShaalaTrainer(teacher_model, student_model, cfg)
    rng = jax.random.PRNGKey(0)
    state = trainer.create_train_state(rng)
    
    # Dummy batch
    batch = {
        'input_ids': jnp.ones((16, 16), dtype=jnp.int32),  # Larger batch for TPU
        'labels': jnp.ones((16, 16), dtype=jnp.int32)
    }
    
    # Test training step
    outputs, new_state = trainer.train_step(state, batch, rng)
    print(f"Loss: {outputs['loss']}, Metrics: {outputs['metrics']}")
