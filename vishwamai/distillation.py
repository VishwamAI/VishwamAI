"""
Knowledge distillation module for VishwamAI.
"""
import jax
import jax.numpy as jnp
import flax
import optax
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional
import logging

# Fix circular imports by using the main classes directly
# and importing loss functions from the new module
from .model import VishwamAIModel, ModelConfig
from .loss_functions import cross_entropy_loss, kl_divergence_loss

logger = logging.getLogger(__name__)

# Define TrainingState class directly to avoid circular import
class TrainingState(train_state.TrainState):
    """Extended train state with EMA and other training metrics."""
    # Fix the order: non-default arguments need to come before default arguments
    ema_params: Optional[Dict] = None
    best_metrics: Optional[Dict] = None
    tot_state: Optional[Dict] = None  # New field for Tree of Thoughts state

def create_optimizer(config):
    """Create optimizer with learning rate schedule."""
    lr_schedule = create_learning_rate_scheduler(
        base_learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.max_steps,
    )
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=config.training.adam_beta1,
            b2=config.training.adam_beta2,
            weight_decay=config.training.weight_decay
        )
    )
    
    return tx, lr_schedule

def create_learning_rate_scheduler(
    factors="constant * linear_warmup * cosine_decay",
    base_learning_rate=0.0001,
    warmup_steps=1000,
    decay_steps=100000,
):
    """Creates a learning rate schedule."""
    factors = [f.strip() for f in factors.split('*')]
    
    def schedule(step):
        """Calculate learning rate based on step."""
        rate = 1.0
        for factor in factors:
            if factor == 'constant':
                rate *= base_learning_rate
            elif factor == 'linear_warmup':
                rate *= jnp.minimum(1.0, step / warmup_steps)
            elif factor == 'cosine_decay':
                rate *= 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step, decay_steps) / decay_steps))
            else:
                raise ValueError(f"Unknown factor: {factor}")
        return rate
    
    return schedule

class VishwamaiGuruKnowledge:
    """Teacher model knowledge handler for distillation."""
    
    def __init__(self, config):
        self.config = config
        self.temperature = config.distillation.kd_temperature
        self.alpha_kd = config.distillation.alpha_kd
        self.alpha_ce = config.distillation.alpha_ce
    
    def distill(self, teacher_logits, student_logits, labels, temperature=None):
        """Apply knowledge distillation with temperature scaling."""
        if temperature is None:
            temperature = self.temperature
            
        # Use the kl_divergence_loss from loss_functions module
        kd_loss = kl_divergence_loss(student_logits, teacher_logits, temperature)
        
        # Compute cross-entropy with ground truth
        mask = (labels != 0).astype(jnp.float32)
        ce_loss = cross_entropy_loss(student_logits, labels)
        
        # Combine losses with alpha weights
        combined_loss = self.alpha_kd * kd_loss + self.alpha_ce * ce_loss
        
        return combined_loss, {"kd_loss": kd_loss, "ce_loss": ce_loss}

class VishwamaiShaalaTrainer:
    """Enhanced trainer for knowledge distillation."""
    
    def __init__(self, teacher_model, student_model, cfg):
        """Initialize distillation trainer."""
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.cfg = cfg
        self.guru = VishwamaiGuruKnowledge(cfg)
        
    def create_train_state(self, rng):
        """Create initial training state for student model."""
        # Create optimizer from config
        tx, _ = create_optimizer(self.cfg)
        
        # Initialize parameters if necessary
        if not hasattr(self.student_model, 'params') or self.student_model.params is None:
            dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
            params = self.student_model.init(rng, dummy_input)['params']
            self.student_model = self.student_model.bind({'params': params})
        
        # Create TrainingState
        state = TrainingState.create(
            apply_fn=self.student_model.__call__,
            params=self.student_model.params,
            tx=tx,
            ema_params=None,
            step=0,
            best_metrics={
                'loss': float('inf'),
                'accuracy': 0.0,
            },
            tot_state={
                'enabled': self.cfg.training.get('use_tot', False),
                'search_strategy': self.cfg.training.get('tot_search_strategy', 'beam'),
                'thoughts_per_batch': 0,
                'best_thought_score': 0.0
            }
        )
        
        return state
