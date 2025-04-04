"""Training utilities for TPU optimization."""

import jax
import jax.numpy as jnp
from typing import Any, Dict, Optional
from flax.training import train_state
import optax

def create_train_state(
    model: Any,
    optimizer: optax.GradientTransformation,
    rng: jnp.ndarray,
    input_shape: Optional[Dict[str, Any]] = None
) -> train_state.TrainState:
    """Create training state with TPU optimizations.
    
    Args:
        model: The model to train
        optimizer: Optax optimizer
        rng: JAX random number generator key
        input_shape: Optional input shapes for initialization
        
    Returns:
        TrainState object containing initialized parameters and optimizer state
    """
    if input_shape is None:
        input_shape = {"input_ids": (1, 128)}
        
    # Initialize parameters
    params = model.init(
        rng,
        jnp.ones((input_shape["input_ids"][0], input_shape["input_ids"][1]), dtype=jnp.int32),
        deterministic=True
    )
    
    # Create training state
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer
    )