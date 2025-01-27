# VishwamAI

## Overview

VishwamAI is a cutting-edge project focused on the development and training of generative AI models. Our goal is to create advanced AI systems capable of understanding and generating human-like text, solving complex mathematical problems, and more.

## Features

- **Generative AI Models**: Train and deploy state-of-the-art generative AI models.
- **Mathematical Reasoning**: Specialized models for solving mathematical problems.
- **Custom Tokenization**: Advanced tokenization techniques for better text understanding.
- **Flexible Architecture**: Easily configurable model architecture to suit various needs.
- **Continuous Integration**: Automated testing and deployment using GitHub Actions.

## Installation

To get started with VishwamAI, follow these steps to install the necessary dependencies:

1. Clone the repository:
   ```bash
   git clone https://github.com/VishwamAI/VishwamAI.git
   cd VishwamAI
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running Tests

To run the tests, use the following command:
```bash
pytest
```

### Training Models

To train a model, use the provided training scripts. For example, to train a model on the mathematical dataset, run:
```bash
python train_math.py --output-dir math_models --model-size 2b --batch-size 32 --num-epochs 10
```

For general training, use:
```bash
python train.py --train-data path/to/train_data.txt --val-data path/to/val_data.txt --output-dir models --model-size 2b --batch-size 32 --num-epochs 10
```

### Tokenization for Math, Physics, and Biology

The `ConceptualTokenizer` class in `vishwamai/conceptual_tokenizer.py` now includes subject-specific tokens for math, physics, and biology. This allows for more precise tokenization and understanding of subject-specific texts.

To use the subject-specific tokens, ensure that the `ConceptualTokenizer` is initialized with the appropriate subject-specific tokens. For example:
```python
tokenizer = ConceptualTokenizer(
    vocab_size=32000,
    max_length=512
)
tokenizer.subject_specific_tokens.update({
    "math": 6,
    "physics": 7,
    "biology": 8
})
```

### Using the Jupyter Notebook for Math Dataset Integration

A new Jupyter Notebook file `math/vishwamai_math_integration.ipynb` has been added to demonstrate how to integrate a small math dataset with VishwamAI. The notebook includes sections for loading the dataset, initializing the model, training the model, and evaluating the results.

To use the notebook, follow these steps:

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook math/vishwamai_math_integration.ipynb
   ```

2. Follow the instructions in the notebook to load the dataset, initialize the model, train the model, and evaluate the results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

We welcome contributions to VishwamAI! To contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes and commit them with clear and concise messages.
4. Push your changes to your forked repository.
5. Create a pull request to the main repository.

For any issues or feature requests, please open an issue on GitHub.

Happy coding!
