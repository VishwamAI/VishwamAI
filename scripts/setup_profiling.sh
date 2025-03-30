#!/usr/bin/env bash
# Strict error handling
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}Setting up CUDA profiling tools${NC}"

# Make scripts executable
chmod +x scripts/profile_cuda.py
chmod +x scripts/visualize_profile.py

# Check dependencies
echo -e "${YELLOW}Checking Python dependencies...${NC}"
python3 -c "import numpy" 2>/dev/null || {
    echo -e "${RED}numpy not found. Installing...${NC}"
    pip3 install numpy
}

python3 -c "import matplotlib" 2>/dev/null || {
    echo -e "${RED}matplotlib not found. Installing...${NC}"
    pip3 install matplotlib
}

python3 -c "import seaborn" 2>/dev/null || {
    echo -e "${RED}seaborn not found. Installing...${NC}"
    pip3 install seaborn
}

# Check CUDA tools
echo -e "${YELLOW}Checking CUDA tools...${NC}"

if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}CUDA toolkit not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

if ! command -v nvprof &> /dev/null; then
    echo -e "${YELLOW}nvprof not found. Some profiling features may be limited.${NC}"
fi

# Create profile results directory
mkdir -p profile_results
mkdir -p profile_visualizations

# Create sample configuration
cat > profile_config.json << 'EOL'
{
    "profile_settings": {
        "metrics": ["sm_efficiency", "achieved_occupancy", "dram_read_throughput", "dram_write_throughput"],
        "events": ["warps_launched", "threads_launched", "l2_read_transactions", "l2_write_transactions"],
        "iterations": 10,
        "warmup_iterations": 2
    },
    "visualization_settings": {
        "plot_format": "png",
        "show_memory_analysis": true,
        "show_kernel_analysis": true,
        "generate_report": true
    }
}
EOL

echo -e "${GREEN}Setup complete!${NC}"
echo "Usage:"
echo "  1. Run profiling: ./scripts/profile_cuda.py --binary <path_to_binary>"
echo "  2. Visualize results: ./scripts/visualize_profile.py --input <profile_results_file>"

# Add README for profiling
cat > PROFILING.md << 'EOL'
# CUDA Kernel Profiling Guide

This guide explains how to use the profiling tools for analyzing and optimizing the CUDA kernels in the knowledge distillation pipeline.

## Prerequisites

- CUDA Toolkit
- Python 3.6 or higher
- Required Python packages: numpy, matplotlib, seaborn

## Quick Start

1. Run profiling:
   ```bash
   ./scripts/profile_cuda.py --binary build/test_cuda_distillation
   ```

2. Visualize results:
   ```bash
   ./scripts/visualize_profile.py --input profile_results/results_*.json
   ```

## Configuration

Edit `profile_config.json` to customize:
- Metrics to collect
- Number of iterations
- Visualization settings

## Output

Results are saved in:
- `profile_results/`: Raw profiling data
- `profile_visualizations/`: Generated plots and reports

## Analyzing Results

The visualization tool generates:
1. Kernel performance plots
2. Memory usage analysis
3. Overall metrics summary
4. HTML report with detailed analysis

## Troubleshooting

Common issues and solutions:
1. Missing CUDA tools: Install CUDA toolkit
2. Permission denied: Run setup_profiling.sh
3. Memory errors: Adjust batch size or sequence length
EOL

# Make this script itself executable
chmod +x scripts/setup_profiling.sh

echo -e "${GREEN}Created PROFILING.md with usage instructions${NC}"
