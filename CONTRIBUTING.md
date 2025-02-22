# Contributing to Vishwamai

Thank you for your interest in contributing to Vishwamai! This document provides guidelines and best practices for contributing to the project.

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/vishwamai.git
cd vishwamai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r tests/requirements.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style Guidelines

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Add docstrings to all public functions, classes, and methods
- Keep functions focused and modular
- Use meaningful variable names

## Precision Handling Guidelines

1. **Type Declarations**
- Always explicitly declare precision types in model configurations
- Use appropriate precision settings based on hardware capabilities
```python
config = ModelArgs(
    dtype="fp16",
    use_mixed_precision=True,
    gradient_precision="fp32"
)
```

2. **Memory Management**
- Implement gradient checkpointing for large models
- Clear CUDA cache when benchmarking
- Track and log memory usage
```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

3. **Numerical Stability**
- Use mixed precision training with FP32 gradients
- Implement gradient clipping
- Add validation for NaN/Inf values

## Testing Requirements

1. **Precision Tests**
- Test all supported precision modes (FP16, FP32, BF16)
- Verify numerical accuracy against FP32 baseline
- Check memory usage for different precisions
```bash
cd tests
./run_precision_tests.sh
```

2. **Performance Tests**
- Benchmark model throughput
- Measure memory efficiency
- Test scaling with sequence length
```bash
cd examples
python benchmark_precision.py
```

3. **Hardware Tests**
- Test on T4 GPU with different batch sizes
- Verify Flash Attention functionality
- Check Tensor Core utilization

## Pull Request Process

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit:
```bash
git add .
git commit -m "feat: description of your changes"
```

3. Run all tests:
```bash
pytest tests/
```

4. Update documentation:
- Add docstrings for new functions
- Update README.md if needed
- Add example usage if applicable

5. Submit Pull Request:
- Fill out the PR template
- Link relevant issues
- Describe testing performed

## Best Practices

1. **Model Implementation**
- Use factory functions for model creation
- Support dynamic configuration
- Implement proper cleanup in destructors

2. **Configuration Management**
- Use ModelArgs for configuration
- Validate configuration parameters
- Document default values

3. **Error Handling**
- Add proper error messages
- Validate input shapes and types
- Handle edge cases gracefully

## Documentation

1. **Code Documentation**
- Add detailed docstrings
- Include type hints
- Explain complex algorithms

2. **Usage Examples**
- Provide practical examples
- Include configuration samples
- Document common use cases

3. **Performance Notes**
- Document memory requirements
- Note precision trade-offs
- Include benchmarks

## Reporting Issues

When reporting issues, please include:
- Detailed description of the problem
- Steps to reproduce
- System information:
  - Python version
  - PyTorch version
  - GPU model and driver version
- Error messages and stack traces
- Minimal reproducible example

## License

By contributing to Vishwamai, you agree that your contributions will be licensed under the Apache License 2.0.

## Questions?

Feel free to:
- Open an issue for questions
- Join our Discord community
- Contact maintainers directly

Thank you for helping improve Vishwamai!
