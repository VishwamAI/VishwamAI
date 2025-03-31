"""TPU-optimized optimizers with memory-efficient updates."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Dict, NamedTuple, Optional, Tuple, Any
from functools import partial

class OptimizerState(NamedTuple):
    """Optimizer state with efficient memory layout."""
    params: Dict[str, jnp.ndarray]
    moments: Dict[str, jnp.ndarray]
    velocities: Optional[Dict[str, jnp.ndarray]] = None
    scaling_factor: Optional[jnp.ndarray] = None

class TPUOptimizer:
    """Base class for TPU-optimized optimizers."""
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        use_bfloat16: bool = True,
        pipeline_updates: bool = True,
        fuse_ops: bool = True
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.use_bfloat16 = use_bfloat16
        self.pipeline_updates = pipeline_updates
        self.fuse_ops = fuse_ops
        
    def init_state(self, params: Dict[str, jnp.ndarray]) -> OptimizerState:
        """Initialize optimizer state with correct memory layout."""
        moments = {
            k: jnp.zeros_like(v, dtype=jnp.bfloat16 if self.use_bfloat16 else v.dtype)
            for k, v in params.items()
        }
        return OptimizerState(params=params, moments=moments)

class TPUAdam(TPUOptimizer):
    """TPU-optimized Adam with memory-efficient updates."""
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OptimizerState,
        grads: Dict[str, jnp.ndarray],
        step: int
    ) -> OptimizerState:
        """Update parameters with fused operations."""
        if self.fuse_ops:
            return self._fused_update(state, grads, step)
        return self._standard_update(state, grads, step)
        
    def _fused_update(
        self,
        state: OptimizerState,
        grads: Dict[str, jnp.ndarray],
        step: int
    ) -> OptimizerState:
        """Fused Adam update for TPU efficiency."""
        # Bias correction
        bias_correction1 = 1 - self.beta1 ** (step + 1)
        bias_correction2 = 1 - self.beta2 ** (step + 1)
        scaled_lr = self.learning_rate * jnp.sqrt(bias_correction2) / bias_correction1
        
        def update_param(param_key):
            param = state.params[param_key]
            grad = grads[param_key]
            moment1 = state.moments[param_key]
            moment2 = state.velocities[param_key] if state.velocities else None
            
            # Fused update computation
            def fused_op(param, grad, m1, m2=None):
                # Update biased first moment estimate
                m1_new = self.beta1 * m1 + (1 - self.beta1) * grad
                
                # Update biased second moment estimate
                if m2 is not None:
                    m2_new = self.beta2 * m2 + (1 - self.beta2) * grad * grad
                    denom = jnp.sqrt(m2_new) + self.epsilon
                else:
                    m2_new = None
                    denom = 1.0
                
                # Compute update
                update = m1_new / denom
                
                # Apply weight decay
                if self.weight_decay > 0:
                    update = update + self.weight_decay * param
                
                # Update parameter
                param_new = param - scaled_lr * update
                
                return param_new, m1_new, m2_new
            
            # Apply fused update
            new_param, new_m1, new_m2 = fused_op(param, grad, moment1, moment2)
            
            # Update state dictionaries
            state.params[param_key] = new_param
            state.moments[param_key] = new_m1
            if state.velocities:
                state.velocities[param_key] = new_m2
                
            return None
            
        # Process all parameters with pipeline parallelism
        if self.pipeline_updates:
            jax.lax.map(update_param, jnp.array(list(state.params.keys())))
        else:
            for param_key in state.params:
                update_param(param_key)
                
        return state

class TPULion(TPUOptimizer):
    """TPU-optimized Lion optimizer with memory-efficient updates."""
    
    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
        **kwargs
    ):
        super().__init__(
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            **kwargs
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OptimizerState,
        grads: Dict[str, jnp.ndarray],
        step: int
    ) -> OptimizerState:
        """Memory-efficient Lion update."""
        
        def update_param(param_key):
            param = state.params[param_key]
            grad = grads[param_key]
            moment = state.moments[param_key]
            
            # Fused Lion update
            def fused_lion_update(param, grad, moment):
                # Update momentum with sign of gradients
                update = jnp.sign(self.beta1 * moment + (1 - self.beta1) * grad)
                
                # Apply weight decay
                if self.weight_decay > 0:
                    param = param * (1 - self.learning_rate * self.weight_decay)
                
                # Update parameter
                new_param = param - self.learning_rate * update
                
                # Update momentum
                new_moment = self.beta2 * moment + (1 - self.beta2) * grad
                
                return new_param, new_moment
            
            # Apply update
            new_param, new_moment = fused_lion_update(param, grad, moment)
            
            # Update state
            state.params[param_key] = new_param
            state.moments[param_key] = new_moment
            
            return None
            
        # Process updates with pipeline parallelism
        if self.pipeline_updates:
            jax.lax.map(update_param, jnp.array(list(state.params.keys())))
        else:
            for param_key in state.params:
                update_param(param_key)
                
        return state

class TPUAdafactor(TPUOptimizer):
    """Memory-efficient Adafactor implementation for TPU."""
    
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: Optional[float] = None,
        decay_rate: float = 0.8,
        epsilon1: float = 1e-30,
        epsilon2: float = 1e-3,
        clipping_threshold: Optional[float] = 1.0,
        weight_decay: float = 0.0,
        min_dim_size_to_factor: int = 128,
        **kwargs
    ):
        super().__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.decay_rate = decay_rate
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2
        self.clipping_threshold = clipping_threshold
        self.weight_decay = weight_decay
        self.min_dim_size_to_factor = min_dim_size_to_factor
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        state: OptimizerState,
        grads: Dict[str, jnp.ndarray],
        step: int
    ) -> OptimizerState:
        """Memory-efficient Adafactor update."""
        
        def should_use_factored_second_moment(shape):
            """Determine if we should use factored second moment statistics."""
            return (
                len(shape) >= 2 and
                shape[-1] >= self.min_dim_size_to_factor and
                shape[-2] >= self.min_dim_size_to_factor
            )
            
        def update_param(param_key):
            param = state.params[param_key]
            grad = grads[param_key]
            moment = state.moments[param_key]
            scaling = state.scaling_factor
            
            def fused_adafactor_update(param, grad, moment, scaling):
                shape = param.shape
                
                # Update factored second moments
                if should_use_factored_second_moment(shape):
                    grad_squared = grad * grad
                    factored_dims = tuple(range(len(shape)-2))
                    row_mean = jnp.mean(grad_squared, axis=factored_dims + (len(shape)-1,))
                    col_mean = jnp.mean(grad_squared, axis=factored_dims + (len(shape)-2,))
                    
                    new_scaling = scaling * self.decay_rate + (1 - self.decay_rate)
                    row_mean = row_mean / new_scaling
                    col_mean = col_mean / new_scaling
                    
                    row_update = jnp.sqrt(row_mean + self.epsilon1)
                    col_update = jnp.sqrt(col_mean + self.epsilon1)
                    y = grad / (jnp.expand_dims(row_update, -1) * 
                              jnp.expand_dims(col_update, -2))
                else:
                    grad_squared = grad * grad
                    new_moment = moment * self.decay_rate + grad_squared * (1 - self.decay_rate)
                    y = grad / jnp.sqrt(new_moment + self.epsilon1)
                
                # Apply clipping
                if self.clipping_threshold is not None:
                    clipping_denom = jnp.maximum(
                        1.0,
                        jnp.sqrt(jnp.mean(y * y)) / self.clipping_threshold
                    )
                    y = y / clipping_denom
                
                # Apply weight decay
                if self.weight_decay > 0:
                    param = param * (1 - self.learning_rate * self.weight_decay)
                
                # Update parameter
                new_param = param - self.learning_rate * y
                
                return new_param, new_moment, new_scaling
            
            # Apply update
            new_param, new_moment, new_scaling = fused_adafactor_update(
                param, grad, moment, scaling
            )
            
            # Update state
            state.params[param_key] = new_param
            state.moments[param_key] = new_moment
            state.scaling_factor = new_scaling
            
            return None
            
        # Process updates
        if self.pipeline_updates:
            jax.lax.map(update_param, jnp.array(list(state.params.keys())))
        else:
            for param_key in state.params:
                update_param(param_key)
                
        return state