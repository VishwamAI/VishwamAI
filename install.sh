#!/bin/bash

# Clone repository
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Setup git LFS
git lfs install
git lfs pull

# Configure git
git config --local user.name "Your Name"
git config --local user.email "your.email@example.com"

# Setup pre-commit hooks
pip install pre-commit
pre-commit install

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p training_visualizations
mkdir -p evaluation_reports

echo "Installation complete!"
