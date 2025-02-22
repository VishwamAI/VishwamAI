#!/bin/bash

echo "Setting up GPU environment for VishwamAI..."

# Deactivate virtual environment if it exists
deactivate 2>/dev/null || true

# Create fresh virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Clean existing installations to avoid conflicts
echo "Cleaning existing installations..."
pip uninstall -y deepspeed pydantic torch triton

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies from requirements file
echo "Installing dependencies..."
pip install -r requirements-gpu.txt

# Verify installations
echo "Verifying installations..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"
python3 -c "import pydantic; print(f'Pydantic version: {pydantic.__version__}')"

echo "Setup complete! Activate the environment with: source venv/bin/activate"

# Test GPU setup
echo "Testing GPU setup..."
python3 vishwamai/training/gpu_testing.py
