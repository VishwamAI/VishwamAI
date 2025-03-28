#!/bin/bash

# Setup script for VishwamAI test environment

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up VishwamAI test environment...${NC}"

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}Some operations might require sudo privileges${NC}"
fi

# Function to check command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}$1 could not be found${NC}"
        return 1
    fi
    return 0
}

# Check and setup Google Cloud SDK
setup_gcloud() {
    if ! check_command gcloud; then
        echo -e "${YELLOW}Installing Google Cloud SDK...${NC}"
        curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-latest-linux-x86_64.tar.gz
        tar -xf google-cloud-cli-latest-linux-x86_64.tar.gz
        ./google-cloud-sdk/install.sh --quiet
        source ./google-cloud-sdk/path.bash.inc
        gcloud init --quiet
        echo -e "${GREEN}Google Cloud SDK installed successfully${NC}"
    fi
}

# Setup Python environment
setup_python() {
    echo -e "${YELLOW}Setting up Python environment...${NC}"
    if ! check_command python3; then
        echo -e "${RED}Python3 not found. Please install Python 3.8 or higher${NC}"
        exit 1
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate

    # Install base requirements
    pip install --upgrade pip
    pip install -r tests/requirements-test.txt
}

# Setup platform-specific dependencies
setup_platform() {
    case $1 in
        tpu)
            echo -e "${YELLOW}Setting up TPU environment...${NC}"
            pip install "jax[tpu]>=0.4.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
            pip install tensorflow==2.15.0
            export TPU_NAME="local"
            export XRT_TPU_CONFIG="localservice;0;localhost:51011"
            ;;
        gpu)
            echo -e "${YELLOW}Setting up GPU environment...${NC}"
            pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/cu118
            pip install nvidia-cuda-runtime-cu11 nvidia-cuda-nvrtc-cu11
            export CUDA_VISIBLE_DEVICES=0
            ;;
        cpu)
            echo -e "${YELLOW}Setting up CPU environment...${NC}"
            pip install numpy>=1.20.0 scipy>=1.8.0
            ;;
        *)
            echo -e "${RED}Invalid platform: $1${NC}"
            exit 1
            ;;
    esac
}

# Main setup
main() {
    local platform=$1

    # Setup base environment
    setup_gcloud
    setup_python

    # Setup platform-specific environment
    if [ -n "$platform" ]; then
        setup_platform $platform
    else
        # Setup all platforms
        setup_platform tpu
        setup_platform gpu
        setup_platform cpu
    fi

    # Install VishwamAI package
    pip install -e ".[all]"

    echo -e "${GREEN}Environment setup complete!${NC}"
    echo -e "${YELLOW}Run tests with: ./tests/run_tests.py --platform $platform${NC}"
}

# Parse command line arguments
case "$1" in
    tpu|gpu|cpu)
        main $1
        ;;
    all)
        main
        ;;
    *)
        echo -e "Usage: $0 {tpu|gpu|cpu|all}"
        echo -e "Example:"
        echo -e "  $0 tpu    # Setup TPU environment"
        echo -e "  $0 gpu    # Setup GPU environment"
        echo -e "  $0 cpu    # Setup CPU environment"
        echo -e "  $0 all    # Setup all environments"
        exit 1
        ;;
esac
