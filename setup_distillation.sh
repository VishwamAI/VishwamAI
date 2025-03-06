#!/bin/bash

# Setup script for VishwamAI QwQ-32B distillation

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up VishwamAI QwQ-32B distillation environment...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${YELLOW}Detected Python version: ${python_version}${NC}"

# Create virtual environment
echo -e "${GREEN}Creating Python virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install requirements
echo -e "${GREEN}Installing requirements...${NC}"
pip install -r requirements.txt

# Set up TPU environment variables
echo -e "${GREEN}Setting up TPU environment...${NC}"
echo 'export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95' >> ~/.bashrc
echo 'export JAX_PLATFORM_NAME="tpu"' >> ~/.bashrc

# Create necessary directories
echo -e "${GREEN}Creating project directories...${NC}"
mkdir -p checkpoints
mkdir -p logs
mkdir -p data

# Initialize Aim for experiment tracking
echo -e "${GREEN}Initializing Aim...${NC}"
aim init

# Verify JAX installation and TPU detection
echo -e "${YELLOW}Verifying JAX and TPU setup...${NC}"
python3 - <<EOF
import jax
print("\nJAX devices found:")
print(jax.devices())
print("\nDevice count:", jax.device_count())
EOF

# Download QwQ model files
echo -e "${YELLOW}Starting QwQ-32B model download...${NC}"
python3 - <<EOF
from huggingface_hub import snapshot_download
import os

print("Downloading QwQ-32B model files...")
try:
    model_path = snapshot_download(
        "Qwen/QwQ-32B",
        allow_patterns=["*.safetensors", "config.json", "tokenizer.model"],
        local_files_only=False,
        resume_download=True
    )
    shard_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
    print(f"\nFound {len(shard_files)} safetensor shards")
    assert len(shard_files) == 14, f"Expected 14 safetensor files, found {len(shard_files)}"
    print("Model files downloaded successfully")
except Exception as e:
    print(f"Error downloading model: {str(e)}")
EOF

# Set up environment variables for the project
echo -e "${GREEN}Setting up project environment variables...${NC}"
echo "export PYTHONPATH=$PYTHONPATH:$(pwd)" >> ~/.bashrc
echo "export VISHWAMAI_HOME=$(pwd)" >> ~/.bashrc

# Create log file for tracking setup
echo -e "${GREEN}Creating setup log...${NC}"
cat > setup_log.txt <<EOF
VishwamAI QwQ-32B Distillation Setup Log
----------------------------------------
Setup Date: $(date)
Python Version: ${python_version}
JAX Version: $(pip freeze | grep "jax==")
Number of TPU devices: $(python3 -c "import jax; print(len(jax.devices()))")
Model Path: $(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/QwQ-32B', allow_patterns=['config.json']))")
EOF

echo -e "${GREEN}Setup completed! Please verify the following:${NC}"
echo -e "1. Check setup_log.txt for installation details"
echo -e "2. Run 'source ~/.bashrc' to load environment variables"
echo -e "3. Start training with: jupyter notebook train_vishwamai_distillation.ipynb"
echo -e "\n${YELLOW}For monitoring:${NC}"
echo -e "Run 'aim up --host 0.0.0.0 --port 43800' to start the dashboard"

source ~/.bashrc
