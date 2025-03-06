#!/bin/bash

# Run script for QwQ-32B distillation with memory optimization
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check for TPU
if ! python3 -c "import jax; assert len(jax.devices('tpu')) > 0" 2>/dev/null; then
    echo -e "${RED}No TPU devices found!${NC}"
    exit 1
fi

# Get available memory in GB
AVAILABLE_MEM=$(python3 -c "import psutil; print(int(psutil.virtual_memory().available / (1024**3)))")
echo -e "${GREEN}Available memory: ${AVAILABLE_MEM}GB${NC}"

# Check minimum memory requirement (128GB recommended)
if [ "$AVAILABLE_MEM" -lt 128 ]; then
    echo -e "${YELLOW}Warning: Less than recommended memory (128GB) available${NC}"
    echo -e "Continue anyway? [y/N]"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Set environment variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORM_NAME="tpu"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Create required directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p data

# Verify QwQ path
if [ -z "$QWQ_PATH" ]; then
    echo -e "${RED}QWQ_PATH environment variable not set!${NC}"
    echo "Please set QWQ_PATH to point to QwQ-32B model directory"
    exit 1
fi

# Run memory optimization test
echo -e "${GREEN}Running memory optimization test...${NC}"
python3 test_memory_loading.py

# Check test result
if [ $? -ne 0 ]; then
    echo -e "${RED}Memory optimization test failed!${NC}"
    exit 1
fi

# Start Aim tracking server
echo -e "${GREEN}Starting Aim server...${NC}"
aim up --host 0.0.0.0 --port 43800 &
AIM_PID=$!

# Function to cleanup on exit
cleanup() {
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $AIM_PID 2>/dev/null
    jax.clear_caches() 2>/dev/null
}
trap cleanup EXIT

# Main training
echo -e "${GREEN}Starting distillation training...${NC}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Run training with logging
{
    echo "Training started at: $(date)"
    echo "Available memory: ${AVAILABLE_MEM}GB"
    echo "TPU devices: $(python3 -c "import jax; print(jax.devices())")"
    echo "-------------------"
    
    jupyter nbconvert --to python train_vishwamai_distillation.ipynb --output tmp_training
    python3 tmp_training.py
    rm tmp_training.py
    
    echo "-------------------"
    echo "Training completed at: $(date)"
} 2>&1 | tee "$LOG_FILE"

echo -e "${GREEN}Training complete! Logs saved to: $LOG_FILE${NC}"

# Verify output model
if [ -d "final_vishwamai_model" ]; then
    echo -e "${GREEN}Verifying final model...${NC}"
    python3 -c "
from vishwamai.model import load_model
model = load_model('final_vishwamai_model')
print('Model verification successful')
"
else
    echo -e "${RED}Final model not found!${NC}"
    exit 1
fi

echo -e "${GREEN}Distillation process completed successfully!${NC}"
