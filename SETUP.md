# VishwamAI Setup Guide

This guide provides detailed instructions for setting up VishwamAI with all its enhanced components.

## Prerequisites

### Hardware Requirements
```
GPU: NVIDIA A100 (80GB) or V100 (32GB)
RAM: 64GB minimum
Storage: 1TB+ SSD
CUDA: 11.8 or higher
```

### Software Dependencies
```bash
# Core dependencies
python>=3.9
pytorch>=2.4.1
transformers>=4.34.0
numpy>=1.24.0
wandb>=0.15.0
```

## Installation

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n vishwamai python=3.9
conda activate vishwamai

# Install PyTorch with CUDA support
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install VishwamAI

```bash
# Clone repository
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI

# Install package
pip install -e .
```

### 3. Verify Installation

```python
import torch
from vishwamai import EnhancedVishwamAI

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test model loading
model = EnhancedVishwamAI(
    model_path="kasinadhsarma/vishwamai-model",
    use_memory=True,
    use_tree=True,
    use_cache=True
)
```

## Component Setup

### 1. Neural Memory

```python
from vishwamai.neural_memory import ReasoningMemoryTransformer, MemoryConfig

# Configure memory
memory_config = MemoryConfig(
    hidden_size=8192,
    memory_size=2048,
    num_memory_layers=3
)

# Initialize memory
memory = ReasoningMemoryTransformer(memory_config)
```

### 2. Tree of Thoughts

```python
from vishwamai.tree_of_thoughts import TreeOfThoughts, TreeConfig

# Configure tree search
tree_config = TreeConfig(
    beam_width=4,
    max_depth=3,
    temperature=0.7
)

# Initialize tree
tree = TreeOfThoughts(model, tree_config)
```

### 3. Cache Augmentation

```python
from vishwamai.cache_augmentation import DifferentiableCacheAugmentation, CacheConfig

# Configure cache
cache_config = CacheConfig(
    hidden_size=8192,
    num_heads=8,
    max_cache_length=65536
)

# Initialize cache
cache = DifferentiableCacheAugmentation(cache_config)
```

## Environment Variables

Set these environment variables for optimal performance:

```bash
# Model configuration
export VISHWAMAI_MODEL_PATH="/path/to/model"
export VISHWAMAI_CACHE_DIR="/path/to/cache"

# Hardware optimization
export CUDA_VISIBLE_DEVICES="0,1"  # For multi-GPU
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
export TORCH_DISTRIBUTED_DEBUG="INFO"

# Memory management
export MAX_MEMORY_ALLOCATION="70GB"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

## GPU Memory Optimization

### 1. Memory Efficiency

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    outputs = model.generate(...)
```

### 2. Batch Size Optimization

```python
# Automatic batch size finder
from vishwamai.utils import find_optimal_batch_size

optimal_batch_size = find_optimal_batch_size(
    model,
    starting_batch_size=8,
    gpu_memory_threshold=0.9
)
```

## Monitoring Setup

### 1. Weights & Biases

```python
import wandb

wandb.login()
wandb.init(project="vishwamai", name="training_run_1")
```

### 2. TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir ./runs
```

## Common Issues

### 1. Out of Memory (OOM)

```python
# Solution 1: Reduce memory components
memory_config.memory_size = 1024
cache_config.max_cache_length = 32768

# Solution 2: Enable memory efficient attention
model.enable_memory_efficient_attention()
```

### 2. CUDA Issues

```bash
# Verify CUDA setup
python -c "import torch; print(torch.cuda.is_available())"

# Clear CUDA cache
torch.cuda.empty_cache()
```

### 3. Component Conflicts

```python
# Ensure consistent dimensions
assert memory_config.hidden_size == model.config.hidden_size
assert cache_config.hidden_size == model.config.hidden_size
```

## Performance Tuning

### 1. Cache Warmup

```python
# Warm up cache before inference
model.cache_module.warmup(
    sample_inputs,
    num_iterations=100
)
```

### 2. Memory Preloading

```python
# Preload common patterns
model.memory_module.preload_patterns(
    pattern_dataset,
    batch_size=32
)
```

### 3. Tree Search Optimization

```python
# Adjust beam search parameters
model.tree_module.optimize_beam_width(
    validation_data,
    min_width=2,
    max_width=8
)
```

## Next Steps

1. Check the [Training Guide](TRAINING.md) for training instructions
2. Review [Model Card](MODEL_CARD.md) for model details
3. Explore examples in `examples/` directory
4. Join our [Discord community](https://discord.gg/vishwamai) for support

For more detailed documentation, visit [vishwamai.readthedocs.io](https://vishwamai.readthedocs.io).
