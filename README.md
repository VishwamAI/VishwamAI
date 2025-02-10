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
