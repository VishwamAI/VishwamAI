#!/usr/bin/env bash

# Enable strict mode
set -euo pipefail

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR/../.."

# Check for Hugging Face token
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable is not set"
    echo "Please set your Hugging Face token first:"
    echo "export HF_TOKEN='your_token_here'"
    exit 1
fi

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Python3 is required but not installed. Aborting."
    exit 1
fi

# Install required packages
echo "Installing packages..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA support if available
if command -v nvidia-smi &> /dev/null; then
    echo "Installing PyTorch (GPU version)..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "Installing PyTorch (CPU version)..."
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other dependencies..."
python3 -m pip install transformers datasets huggingface-hub accelerate
python3 -m pip install tqdm pandas numpy scikit-learn

echo "Running training script..."
python3 "$PROJECT_ROOT/vishwamai/examples/pretrain_and_upload.py"

echo "Training complete!"
