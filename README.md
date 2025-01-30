# VishwamAI

VishwamAI is a sophisticated machine learning library focusing on efficient model quantization, advanced tokenization, and mathematical reasoning capabilities.

## Features

- **Advanced Tokenization**: Conceptual tokenizer with semantic clustering and special token handling
- **Efficient Quantization**: Support for FP8 and BF16 quantization
- **Mathematical Reasoning**: Integration with GSM8K dataset for advanced mathematical problem-solving
- **Model Architecture**: Flexible transformer-based architecture with configurable parameters
- **Training Utilities**: Support for distributed training, mixed precision, and gradient accumulation

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from vishwamai.model import VishwamaiModel
from vishwamai.conceptual_tokenizer import ConceptualTokenizer

# Initialize tokenizer and model
tokenizer = ConceptualTokenizer()
model = VishwamaiModel()

# Example usage
text = "Solve: If John has 5 apples and gives 2 to Mary, how many does he have left?"
tokens = tokenizer.encode(text)
output = model.generate(tokens)
```

## Testing

Run the test suite:

```bash
pytest -v
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.1.0
- CUDA toolkit (for GPU support)
- Additional dependencies listed in setup.py

## Project Structure

```
vishwamai/
├── conceptual_tokenizer.py   # Advanced tokenization implementation
├── kernel.py                 # CUDA kernels and quantization
├── model.py                 # Core model architecture
├── training.py              # Training utilities
└── configs/                 # Model configurations
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Troubleshooting Workflow Failures

If you encounter issues with the GitHub Actions workflow, here are some common troubleshooting steps:

1. **Check Dependencies**: Ensure all required dependencies are listed in `requirements.txt` and `vishwamai/requirements.txt`. Missing or incompatible packages can cause workflow failures.

2. **Verify Configuration**: Double-check the configurations in `.github/workflows/ci.yml` and `pytest.ini`. Incorrect configurations can lead to test discovery or execution issues.

3. **Review Logs**: Examine the logs from the failed workflow run. Logs often provide detailed error messages that can help identify the root cause of the failure.

4. **Run Locally**: Try running the tests and workflow steps locally on your machine. This can help isolate whether the issue is specific to the GitHub Actions environment.

5. **Update Dependencies**: Ensure that all dependencies are up to date. Sometimes, newer versions of packages include bug fixes that can resolve workflow issues.

6. **Seek Help**: If you're unable to resolve the issue, consider seeking help from the community or the project maintainers. Provide detailed information about the failure and the steps you've already taken to troubleshoot.

By following these steps, you can effectively troubleshoot and resolve common workflow failures.
