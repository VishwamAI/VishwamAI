#!/usr/bin/env bash
# Run advanced model demonstrations

# Exit on error
set -e

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add parent directory to Python path
export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"

# Install requirements if needed
if [ ! -f "requirements.txt" ] && [ -f "../requirements.txt" ]; then
    ln -sf ../requirements.txt requirements.txt
fi

echo "Installing requirements..."
python3 -m pip install -r requirements.txt
python3 -m pip install -e ..

echo "Running advanced model demonstrations..."
python3 advanced_model_demo.py
