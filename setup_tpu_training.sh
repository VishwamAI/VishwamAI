#!/usr/bin/env bash
# shellcheck disable=SC1090

set -euo pipefail
IFS=$'\n\t'

# Setup script for TPU training environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function log() {
    echo "[$(date +'%Y-%m-%dT%H:%M:%S%z')]: $*"
}

function error() {
    log "ERROR: $*" >&2
    exit 1
}

function check_requirements() {
    log "Checking requirements..."
    
    # Check if running on TPU VM
    if ! command -v tensorflow-version &> /dev/null; then
        error "This script must be run on a TPU VM instance"
    fi
    
    # Check Python version
    local python_version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ ! $python_version =~ ^3\.[789]$ ]]; then
        error "Python version 3.7-3.9 required, found: $python_version"
    fi
}

function setup_virtualenv() {
    log "Setting up Python virtual environment..."
    
    if [[ -d "vishwamai_env" ]]; then
        log "Virtual environment exists, activating..."
    else
        python3 -m venv vishwamai_env || error "Failed to create virtual environment"
    fi
    
    source vishwamai_env/bin/activate || error "Failed to activate virtual environment"
}

function install_dependencies() {
    log "Installing system dependencies..."
    sudo apt-get update || error "Failed to update package list"
    sudo apt-get install -y python3-pip python3-dev || error "Failed to install system packages"
    
    log "Configuring TensorFlow..."
    pip uninstall -y tensorflow
    pip install tensorflow-cpu || error "Failed to install tensorflow-cpu"
    
    log "Installing PyTorch and XLA..."
    pip install torch==2.0.0 || error "Failed to install PyTorch"
    pip install 'torch_xla[tpu]>=2.0' -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp39-cp39-linux_x86_64.whl || error "Failed to install PyTorch/XLA"
    
    log "Installing additional dependencies..."
    pip install transformers datasets accelerate wandb sentencepiece || error "Failed to install ML packages"
    pip install numpy pandas matplotlib jupyter || error "Failed to install utility packages"
}

function setup_repository() {
    log "Setting up VishwamAI repository..."
    
    if [[ -d "VishwamAI" ]]; then
        log "Updating existing repository..."
        cd VishwamAI || error "Failed to enter repository directory"
        git pull || error "Failed to update repository"
        cd "$SCRIPT_DIR" || error "Failed to return to script directory"
    else
        git clone https://github.com/VishwamAI/VishwamAI.git || error "Failed to clone repository"
    fi
    
    log "Installing VishwamAI package..."
    cd VishwamAI || error "Failed to enter repository directory"
    pip install -e . || error "Failed to install package"
    cd "$SCRIPT_DIR" || error "Failed to return to script directory"
}

function verify_installation() {
    log "Verifying installation..."
    python3 -c "
import torch
import torch_xla
import torch_xla.core.xla_model as xm
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch/XLA version: {torch_xla.__version__}')
" || error "Failed to verify PyTorch/XLA installation"
}

function setup_dataset() {
    log "Downloading GSM8K dataset..."
    python3 -c "from datasets import load_dataset; load_dataset('openai/gsm8k', 'main')" || error "Failed to download dataset"
}

function setup_directories() {
    log "Setting up output directories..."
    mkdir -p checkpoints logs results || error "Failed to create output directories"
}

function configure_environment() {
    log "Configuring environment variables..."
    # shellcheck disable=SC2155
    export TPU_NUM_CORES=$(python3 -c "import torch_xla.core.xla_model as xm; print(xm.xrt_world_size())")
    export WANDB_PROJECT="vishwamai-gsm8k"
    export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PWD}/VishwamAI"
}

function main() {
    log "Starting TPU training environment setup..."
    
    check_requirements
    setup_virtualenv
    install_dependencies
    setup_repository
    verify_installation
    setup_dataset
    setup_directories
    configure_environment
    
    log "Setup complete!"
    log "Number of TPU cores available: $TPU_NUM_CORES"
    echo
    log "To start training:"
    echo "1. Activate the virtual environment: source vishwamai_env/bin/activate"
    echo "2. Open Jupyter: jupyter notebook --ip=0.0.0.0 --port=8888"
    echo "3. Navigate to GSM8K_Training.ipynb"
    echo
    log "Note: Configure your wandb API key before training:"
    echo "wandb login"
}

# Execute main function
main "$@"
