"""Budget model training implementation with credit-efficient optimizations."""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, Tuple
import optax
from ..transformer import TPUTrainingState, TPUOptimizer
from ..configs.budget_model_config import BudgetModelConfig
from ..tpu_credit_manager import TPUCreditManager, TPUCreditConfig

class BudgetModelTrainer:
    """Training manager optimized for budget model."""
    
    def __init__(
        self,
        config: BudgetModelConfig,
        credit_config: Optional[TPUCreditConfig] = None
    ):
        self.config = config
        self.credit_manager = TPUCreditManager(
            credit_config or TPUCreditConfig()
        )
        self.current_batch_size = config.training_config["batch_size"]
        
    def create_train_state(
        self,
        model: Any,
        learning_rate: float,
        rng: jnp.ndarray
    ) -> TPUTrainingState:
        """Initialize training state with memory optimizations."""
        
        # Create learning rate schedule
        schedule_fn = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=self.config.training_config["warmup_steps"],
            decay_steps=self.config.training_config["max_steps"],
            end_value=learning_rate * 0.1
        )
        
        # Create optimizer with gradient clipping and weight decay
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(
                learning_rate=schedule_fn,
                b1=0.9,
                b2=0.98,
                eps=1e-8,
                weight_decay=0.01
            )
        )
        
        # Initialize model with dummy batch
        dummy_batch = jnp.ones((
            self.current_batch_size,
            self.config.model_config["max_position_embeddings"]
        ), dtype=jnp.int32)
        
        variables = model.init(rng, dummy_batch)
        
        # Create training state
        return TPUTrainingState(
            params=variables["params"],
            opt_state=optimizer.init(variables["params"]),
            model_fn=model.apply,
            tx=optimizer
        )
    
    @partial(jax.jit, donate_argnums=(0,))
    def train_step(
        self,
        state: TPUTrainingState,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray
    ) -> Tuple[TPUTrainingState, Dict[str, jnp.ndarray]]:
        """Execute single training step with credit monitoring."""
        
        def loss_fn(params):
            logits = state.model_fn(
                {"params": params},
                batch["input_ids"],
                deterministic=False,
                rngs={"dropout": rng}
            )
            
            # Calculate loss with flash attention
            loss = TPUOptimizer.memory_efficient_attention(
                query=logits,
                key=logits,
                value=logits,
                mask=None  # Auto-regressive masking handled by attention
            )
            return loss.mean()
        
        # Get loss and gradients
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        
        # Apply gradients with efficient ops
        new_state = state.apply_gradients(grads=grads)
        
        metrics = {
            "loss": loss,
            "learning_rate": state.tx.learning_rate
        }
        
        # Update credit tracking
        self.credit_manager.update_usage({
            "compute_util": 0.95,  # Estimated utilization
            "memory_gb": self.config.get_estimated_memory()
        })
        
        # Dynamically adjust batch size if needed
        metrics["batch_size"] = self.maybe_adjust_batch_size(loss)
        
        return new_state, metrics
    
    def maybe_adjust_batch_size(self, current_loss: float) -> int:
        """Dynamically adjust batch size based on loss and credit usage."""
        if not self.config.tpu_config["dynamic_batch_size"]:
            return self.current_batch_size
            
        credit_ratio = self.credit_manager.get_credit_ratio()
        
        # Reduce batch size if credit usage is high
        if credit_ratio > 0.8:
            self.current_batch_size = max(16, self.current_batch_size // 2)
            
        # Increase batch size if loss is stable and credits available
        elif credit_ratio < 0.5 and current_loss < 2.0:
            self.current_batch_size = min(128, self.current_batch_size * 2)
            
        return self.current_batch_size
    
    def train(
        self,
        train_ds: Any,
        model: Any,
        learning_rate: float = None,
        num_steps: Optional[int] = None
    ):
        """Run training loop with credit monitoring."""
        rng = jax.random.PRNGKey(0)
        learning_rate = learning_rate or self.config.training_config["learning_rate"]
        num_steps = num_steps or self.config.training_config["max_steps"]
        
        # Initialize training state
        state = self.create_train_state(model, learning_rate, rng)
        
        # Training loop with credit checks
        for step in range(num_steps):
            rng, dropout_rng = jax.random.split(rng)
            batch = next(train_ds)
            
            # Check if we need resource optimization
            if self.credit_manager.should_enable_optimizations()["reduce_precision"]:
                batch = jax.tree_map(
                    lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x,
                    batch
                )
            
            state, metrics = self.train_step(state, batch, dropout_rng)
            
            # Log progress
            if step % self.config.monitoring_config["log_steps"] == 0:
                credit_metrics = self.credit_manager.get_credit_metrics()
                print(f"Step {step}: loss = {metrics['loss']:.4f}, "
                      f"credits used = {credit_metrics['credits_used']:.2f}, "
                      f"batch size = {metrics['batch_size']}")
            
            # Save checkpoint
            if step % self.config.training_config["save_steps"] == 0:
                self._save_checkpoint(state, step)
        
        return state
    
    def _save_checkpoint(self, state: TPUTrainingState, step: int):
        """Save training checkpoint."""
        checkpoint = {
            "step": step,
            "params": state.params,
            "opt_state": state.opt_state,
            "credit_usage": self.credit_manager.get_credit_metrics(),
            "config": self.config
        }
        # Save checkpoint logic here