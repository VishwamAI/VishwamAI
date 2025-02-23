# Contributing to Vishwamai

We love your input! We want to make contributing to Vishwamai as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code Style

We use several tools to maintain code quality:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run formatters
black vishwamai/
isort vishwamai/

# Run linters
flake8 vishwamai/
mypy vishwamai/

# Run tests
pytest tests/
```

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints for function arguments and return values
- Write docstrings in Google style
- Maximum line length is 88 characters (Black default)
- Use double quotes for strings

## Development Setup

1. Clone the repository
```bash
git clone https://github.com/organization/vishwamai.git
cd vishwamai
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -e ".[dev]"
```

4. Set up pre-commit hooks
```bash
pre-commit install
```

## Testing

We use pytest for our test suite:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_model.py

# Run tests with coverage report
pytest --cov=vishwamai tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files with `test_` prefix
- Use descriptive test names that explain what is being tested
- Include unit tests for new functionality
- Add integration tests for complex features

## Documentation

We use Sphinx for documentation:

```bash
# Build documentation
cd docs
make html
```

### Writing Documentation

- Keep docstrings up to date
- Add examples for non-obvious functionality
- Update README.md for significant changes
- Add architecture diagrams for complex systems

## Pull Request Process

1. Update the README.md with details of changes to the interface
2. Update the documentation with details of any new functionality
3. The PR may be merged once you have the sign-off from maintainers

## Release Process

1. Update version number in `vishwamai/__init__.py`
2. Update CHANGELOG.md
3. Create a GitHub release
4. Build and upload to PyPI

## Project Structure

Follow the existing project structure:

```plaintext
vishwamai/
├── model/          # Model architecture components
├── data/          # Data processing and datasets
├── training/      # Training utilities
├── utils/         # Helper functions
└── scripts/       # Command-line tools
```

## Types of Contributions

### Bug Reports

Report bugs at https://github.com/organization/vishwamai/issues

When reporting a bug, please include:

- Your operating system name and version
- Any details about your local setup that might be helpful in troubleshooting
- Detailed steps to reproduce the bug

### Feature Requests

We welcome feature requests. Please provide:

- Detailed explanation of the feature
- Use cases and benefits
- Potential implementation approach

### Code Contributions

1. Search existing issues and PRs to avoid duplication
2. Discuss major changes in an issue first
3. Follow the coding standards
4. Include tests and documentation
5. Keep PRs focused and atomic

## Community

- Join our [Discord server](https://discord.gg/vishwamai)
- Subscribe to our [mailing list](https://groups.google.com/g/vishwamai)
- Follow us on [Twitter](https://twitter.com/vishwamai)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## References

- [Project Documentation](https://vishwamai.readthedocs.io/)
- [Technical Guide](docs/technical.md)
- [Architecture Overview](docs/architecture.mermaid)
- [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

Feel free to reach out to the maintainers or open an issue for any questions about contributing.

Thank you for your contributions to make Vishwamai better! 🚀
