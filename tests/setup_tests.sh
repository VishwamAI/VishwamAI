#!/usr/bin/env bash

# Setup script for test environment
# Exit on error
set -e

# Source directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Print commands and their arguments as they are executed
set -x

# Function to check command success
check_cmd() {
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed"
        exit 1
    fi
}

# Make test scripts executable
chmod +x run_precision_tests.sh || { echo "Error making run_precision_tests.sh executable"; exit 1; }
chmod +x run_precision_tests.bat || { echo "Error making run_precision_tests.bat executable"; exit 1; }

# Create symbolic link to requirements.txt if not exists
if [ ! -f "requirements.txt" ]; then
    ln -sf ../requirements.txt requirements.txt
    check_cmd "Creating symlink to requirements.txt"
fi

# Create necessary directories
mkdir -p test_reports
check_cmd "Creating test_reports directory"

mkdir -p .benchmarks
check_cmd "Creating .benchmarks directory"

# Find Python executable
PYTHON_CMD=$(which python3 || which python)
if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Python not found. Please ensure Python is installed and in PATH"
    exit 1
fi

# Install test dependencies
"$PYTHON_CMD" -m pip install -r requirements.txt
check_cmd "Installing requirements"

"$PYTHON_CMD" -m pip install -e ..
check_cmd "Installing package in development mode"

# Verify installation
"$PYTHON_CMD" -c "import vishwamai; print(f'Vishwamai version: {vishwamai.__version__}')"
check_cmd "Verifying installation"

# Turn off command printing
set +x

echo "✓ Test environment setup completed successfully"
