"""Test configuration and shared fixtures for VishwamAI."""

import os
import pytest
import jax
import numpy as np

# Force CPU device
os.environ["JAX_PLATFORM_NAME"] = "cpu"

@pytest.fixture(scope="session")
def rng_key():
    """Provide a fixed random key for reproducible tests."""
    return jax.random.PRNGKey(42)

@pytest.fixture(scope="session")
def dummy_input():
    """Create dummy input data for testing."""
    batch_size = 2
    seq_len = 128
    return np.ones((batch_size, seq_len), dtype=np.int32)

@pytest.fixture(scope="session")
def device_setup():
    """Set up CPU device for testing."""
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_enable_x64', False)