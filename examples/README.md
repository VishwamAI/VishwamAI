# Vishwamai Examples

This directory contains example scripts demonstrating the capabilities of Vishwamai.

## Quick Start Examples

### Basic Model Usage
```bash
# On Linux/Mac:
./quickstart.sh

# On Windows:
quickstart.bat
```

This demonstrates:
- Basic model creation and usage
- Different model sizes
- Precision modes
- Simple inference

## Advanced Features

### Mixture of Experts and Parallel Models
```bash
# On Linux/Mac:
./run_advanced_demo.sh

# On Windows:
run_advanced_demo.bat
```

Demonstrates advanced features:
- Mixture of Experts (MoE) models
- Multi-GPU parallel execution
- Advanced transformers with:
  - Flash Attention
  - Fused operations
  - Parallel attention
  - RMS normalization
  - Gated MLPs

### Performance Testing
```bash
python benchmark_precision.py
```

Measures model performance with:
- Different precision modes (FP16, FP32, BF16)
- Memory usage analysis
- Throughput benchmarking
- Scaling characteristics

## Example Configurations

### 1. Mixture of Experts Model
```python
from vishwamai.model import create_expert_model

model = create_expert_model(
    num_experts=8,
    num_experts_per_token=2,
    expert_capacity=32,
    dtype="fp16"
)
```

### 2. Parallel Model
```python
from vishwamai.model import create_parallel_model

model = create_parallel_model(
    hidden_size=2048,
    num_attention_heads=32,
    tensor_parallel_size=4  # Number of GPUs
)
```

### 3. Advanced Transformer
```python
from vishwamai.model import ModelArgs, UnifiedConfig, create_model

config = ModelArgs(
    hidden_size=1024,
    unified=UnifiedConfig(
        transformer=dict(
            use_flash_attention=True,
            fused_qkv=True,
            use_parallel_attention=True
        ),
        mlp=dict(
            gated_mlp=True,
            activation_fn="swiglu"
        )
    )
)

model = create_model(config=config)
```

## Hardware Requirements

### Minimum Requirements
- NVIDIA T4 GPU
- 16GB GPU memory
- CUDA 11.0+
- PyTorch 2.0+

### Recommended for Advanced Features
- Multiple GPUs for parallel execution
- A100 GPU for maximum performance
- 32GB+ GPU memory
- CUDA 11.8+
- Latest PyTorch with CUDA support

## Performance Tips

1. Memory Optimization
```python
from vishwamai.utils.t4_utils import enable_t4_optimizations

# Enable platform-specific optimizations
enable_t4_optimizations()

# Use memory-efficient attention
config = ModelArgs(
    use_flash_attention=True,
    prealloc_cache=True
)
```

2. Multi-GPU Training
```python
config = ModelArgs(
    unified=UnifiedConfig(
        parallel=dict(
            tensor_parallel_size=4,
            sequence_parallel=True
        )
    )
)
```

3. Mixed Precision Training
```python
config = ModelArgs(
    dtype="fp16",
    use_mixed_precision=True,
    gradient_checkpointing=True
)
```

## Running Tests

```bash
# Run all tests
cd tests
./run_precision_tests.sh

# Run specific test
pytest test_precision.py -v
```

## Common Issues

1. Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use Flash Attention
- Enable mixed precision

2. Multi-GPU Issues
- Ensure NCCL is properly installed
- Check GPU connectivity
- Use latest CUDA drivers

3. Performance Issues
- Enable autotuning: `enable_t4_optimizations()`
- Use appropriate batch size
- Monitor GPU utilization

## Additional Resources

- [Vishwamai Documentation](../docs/)
- [A100 Optimizations](../docs/A100_OPTIMIZATIONS.md)
- [GPU Testing Guide](../docs/GPU_TESTING.md)
- [Contributing Guidelines](../CONTRIBUTING.md)

## References

- Flash Attention: https://arxiv.org/abs/2205.14135
- Mixture of Experts: https://arxiv.org/abs/2006.16668
- Parallel Training: https://arxiv.org/abs/1909.08053
