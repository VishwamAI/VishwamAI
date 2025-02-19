#!/bin/bash

# Install LFS
apt-get update
apt-get install git-lfs

# Initialize LFS
git lfs install

# Install dependencies
pip install torch accelerate datasets transformers
pip install matplotlib seaborn pandas

# Create necessary directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p training_visualizations

# Install VishwamAI
pip install -e .
pip install -r requirements.txt

echo "Colab setup complete!"
