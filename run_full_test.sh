#!/bin/bash

# Script to run full testing workflow for VishwamAI

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
TEST_DURATION=1  # hours
REPORT_DIR="test_reports"
VISUALIZATION_DIR="visualizations"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            TEST_DURATION="$2"
            shift 2
            ;;
        --report-dir)
            REPORT_DIR="$2"
            shift 2
            ;;
        --viz-dir)
            VISUALIZATION_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create directories
mkdir -p "$REPORT_DIR"
mkdir -p "$VISUALIZATION_DIR"

# Generate timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="$REPORT_DIR/results_${TIMESTAMP}.json"
REPORT_FILE="$REPORT_DIR/report_${TIMESTAMP}.md"

echo -e "${GREEN}Starting VishwamAI Test Workflow${NC}"
echo "Timestamp: $TIMESTAMP"
echo "Duration: $TEST_DURATION hours"

# Step 1: Setup test environment
echo -e "\n${YELLOW}Setting up test environment...${NC}"
./setup_test_env.sh all || {
    echo -e "${RED}Environment setup failed${NC}"
    exit 1
}

# Step 2: Create cloud resources
echo -e "\n${YELLOW}Creating cloud resources...${NC}"
./manage_test_resources.sh create tpu
./manage_test_resources.sh create gpu
./manage_test_resources.sh create cpu

# Function to cleanup resources
cleanup() {
    echo -e "\n${YELLOW}Cleaning up resources...${NC}"
    ./manage_test_resources.sh delete all
}

# Set trap for cleanup
trap cleanup EXIT

# Step 3: Run tests on each platform
echo -e "\n${YELLOW}Running tests...${NC}"

# TPU tests
echo -e "\n${GREEN}Running TPU tests...${NC}"
./tests/run_tests.py --platform tpu --benchmark \
    --output="${REPORT_DIR}/tpu_${TIMESTAMP}.json"

# GPU tests
echo -e "\n${GREEN}Running GPU tests...${NC}"
./tests/run_tests.py --platform gpu --benchmark \
    --output="${REPORT_DIR}/gpu_${TIMESTAMP}.json"

# CPU tests
echo -e "\n${GREEN}Running CPU tests...${NC}"
./tests/run_tests.py --platform cpu --benchmark \
    --output="${REPORT_DIR}/cpu_${TIMESTAMP}.json"

# Step 4: Combine results
echo -e "\n${YELLOW}Combining test results...${NC}"
python3 - << EOF
import json
import glob

# Load all results
results = {}
for platform in ['tpu', 'gpu', 'cpu']:
    file = f"${REPORT_DIR}/{platform}_${TIMESTAMP}.json"
    try:
        with open(file) as f:
            results[platform] = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Results for {platform} not found")

# Save combined results
with open("${RESULTS_FILE}", 'w') as f:
    json.dump(results, f, indent=2)
EOF

# Step 5: Generate report
echo -e "\n${YELLOW}Generating test report...${NC}"
./generate_test_report.py \
    --benchmark-file "${RESULTS_FILE}" \
    --duration "$TEST_DURATION" \
    --output "$REPORT_FILE"

# Step 6: Create visualizations
echo -e "\n${YELLOW}Creating visualizations...${NC}"
./visualize_test_results.py \
    --results "${RESULTS_FILE}" \
    --output-dir "$VISUALIZATION_DIR"

# Step 7: Update history
echo -e "\n${YELLOW}Updating test history...${NC}"
python3 - << EOF
import json
import datetime
from pathlib import Path

history_file = Path("${REPORT_DIR}/test_history.json")
results_file = Path("${RESULTS_FILE}")

# Load current results
with open(results_file) as f:
    results = json.load(f)

# Create history entry
entry = {
    'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'run_id': "${TIMESTAMP}",
    'tpu_time': results.get('tpu', {}).get('time_ms', 0),
    'gpu_time': results.get('gpu', {}).get('time_ms', 0),
    'cpu_time': results.get('cpu', {}).get('time_ms', 0)
}

# Load and update history
history = []
if history_file.exists():
    with open(history_file) as f:
        history = json.load(f)

history.append(entry)

# Save updated history
with open(history_file, 'w') as f:
    json.dump(history, f, indent=2)
EOF

# Step 8: Generate cost estimate
echo -e "\n${YELLOW}Generating cost estimate...${NC}"
./manage_test_resources.sh estimate all "$TEST_DURATION"

echo -e "\n${GREEN}Test workflow complete!${NC}"
echo "Results: ${RESULTS_FILE}"
echo "Report: ${REPORT_FILE}"
echo "Visualizations: ${VISUALIZATION_DIR}"

# Optional: Open report
if command -v xdg-open >/dev/null; then
    xdg-open "$REPORT_FILE"
elif command -v open >/dev/null; then
    open "$REPORT_FILE"
fi
