import os
import sys
import jax
import jax.numpy as jnp

print("Python version:", sys.version)
print("JAX version:", jax.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# Try to initialize JAX with GPU
try:
    x = jnp.ones((1000, 1000))
    jax.devices()
    print("\nJAX devices available:", jax.devices())
    print("JAX memory usage:", jax.device_memory_stats())
except Exception as e:
    print("\nError initializing JAX GPU:", str(e))
    print("\nTrying CPU-only mode...")
    os.environ["JAX_PLATFORMS"] = "cpu"
    print("JAX devices (CPU-only):", jax.devices())
