#!/bin/bash
# Simple pretraining script for VishwamAI model

# Default values
BATCH_SIZE=4
EPOCHS=3
OUTPUT_DIR="./pretrain_output"
CONFIG_PATH="configs/config_optimized.json"

# Function to display usage
usage() {
    echo "Usage: $0 [-b BATCH_SIZE] [-e EPOCHS] [-o OUTPUT_DIR] [-c CONFIG_PATH]"
    echo "Options:"
    echo "  -b : Batch size (default: 4)"
    echo "  -e : Number of epochs (default: 3)"
    echo "  -o : Output directory (default: ./pretrain_output)"
    echo "  -c : Config file path (default: configs/config_optimized.json)"
    exit 1
}

# Parse command line arguments
while getopts "b:e:o:c:h" opt; do
    case $opt in
        b) BATCH_SIZE=$OPTARG ;;
        e) EPOCHS=$OPTARG ;;
        o) OUTPUT_DIR=$OPTARG ;;
        c) CONFIG_PATH=$OPTARG ;;
        h) usage ;;
        ?) usage ;;
    esac
done

# Create output directory
mkdir -p $OUTPUT_DIR

# Start training
echo "Starting pretraining with:"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Output directory: $OUTPUT_DIR"
echo "Config path: $CONFIG_PATH"

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

# Run training script
python -m vishwamai.examples.train_model \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --epochs $EPOCHS \
    --train_dataset "gsm8k" \
    --eval_dataset "cais/mmlu" \
    --disable_cache

# Save training completion status
if [ $? -eq 0 ]; then
    echo "Training completed successfully" > $OUTPUT_DIR/status.txt
    echo "$(date)" >> $OUTPUT_DIR/status.txt
else
    echo "Training failed" > $OUTPUT_DIR/status.txt
    echo "$(date)" >> $OUTPUT_DIR/status.txt
fi

# Print final status
echo "Training process completed. Check $OUTPUT_DIR/status.txt for details."
