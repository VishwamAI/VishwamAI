#!/usr/bin/env bash

# Script to run comprehensive precision testing suite

# Ensure script fails on error
set -e

# Create symlink to root requirements if it doesn't exist
if [ ! -f "tests/requirements.txt" ] && [ -f "requirements.txt" ]; then
    ln -sf ../requirements.txt tests/requirements.txt
fi

# Find Python executable
PYTHON_CMD=$(which python3 || which python)
if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python not found. Please ensure Python is installed and in PATH"
    exit 1
fi

# Set environment variables
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0+PTX"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Installing test requirements...${NC}"
"$PYTHON_CMD" -m pip install -r tests/requirements.txt
"$PYTHON_CMD" -m pip install -e .

# Function to run tests with specific precision mode
run_precision_tests() {
    local precision=$1
    echo -e "${BLUE}Running tests for $precision precision...${NC}"
    
    # Run tests with specific markers
    "$PYTHON_CMD" -m pytest tests/test_precision.py tests/test_tree_planning.py tests/test_information_retrieval.py \
        -v \
        --precision-mode=$precision \
        -m "precision" \
        --benchmark-only \
        --benchmark-autosave \
        --html=test_reports/${precision}_report.html
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $precision precision tests passed${NC}"
    else
        echo -e "${RED}✗ $precision precision tests failed${NC}"
        return 1
    fi
}

# Create test reports directory
mkdir -p test_reports

echo -e "${BLUE}Starting precision test suite...${NC}"
echo "================================================"

# Check CUDA availability
"$PYTHON_CMD" -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# Create results directory for benchmarks
mkdir -p .benchmarks

# Run tests for each precision mode
PRECISION_MODES=("fp16" "fp32" "fp64" "bf16")
FAILED=0

for mode in "${PRECISION_MODES[@]}"; do
    if ! run_precision_tests $mode; then
        FAILED=1
    fi
    echo "------------------------------------------------"
done

# Run mixed precision tests
echo -e "${BLUE}Running mixed precision tests...${NC}"
"$PYTHON_CMD" -m pytest tests/test_precision.py::test_mixed_precision_training \
    tests/test_tree_planning.py::test_tree_planning_numerical_stability \
    tests/test_information_retrieval.py::test_retrieval_with_mixed_precision \
    -v \
    --html=test_reports/mixed_precision_report.html

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Mixed precision tests passed${NC}"
else
    echo -e "${RED}✗ Mixed precision tests failed${NC}"
    FAILED=1
fi

# Run memory benchmarks
echo -e "${BLUE}Running memory benchmarks...${NC}"
"$PYTHON_CMD" -m pytest tests/test_precision.py::test_precision_memory_usage \
    tests/test_tree_planning.py::test_tree_planning_memory_usage \
    tests/test_information_retrieval.py::test_retrieval_memory_efficiency \
    -v \
    --benchmark-only \
    --html=test_reports/memory_benchmarks.html

# Generate comparison report
"$PYTHON_CMD" -m pytest-benchmark compare

echo "================================================"

# Show test report location
echo -e "${BLUE}Test reports generated in test_reports/ directory${NC}"
echo -e "${BLUE}Benchmark results available in .benchmarks/ directory${NC}"

# Exit with appropriate status
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All precision tests completed successfully${NC}"
    exit 0
else
    echo -e "${RED}Some precision tests failed${NC}"
    exit 1
fi
