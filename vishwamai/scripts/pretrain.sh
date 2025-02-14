#!/bin/bash

# Pretraining script for VishwamAI model
# Usage: ./pretrain.sh [config_path]

set -e  # Exit on error

# Default paths
CONFIG_PATH=${1:-"configs/config_671b.json"}
OUTPUT_DIR="pretrain_output"
LOG_DIR="logs"

# Create directories
mkdir -p $OUTPUT_DIR $LOG_DIR

# Log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/pretrain_$TIMESTAMP.log"

# Environment setup
echo "Setting up environment..." | tee -a $LOG_FILE

# CUDA setup
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs
export CUDA_LAUNCH_BLOCKING=1
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
export TORCH_DISTRIBUTED_DEBUG=INFO

# Memory management
export MAX_MEMORY_ALLOCATION="70GB"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Performance tuning
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())" | tee -a $LOG_FILE
python -c "import torch; print('GPU Count:', torch.cuda.device_count())" | tee -a $LOG_FILE
python -c "import torch; print('GPU Name:', torch.cuda.get_device_name(0))" | tee -a $LOG_FILE

# Install dependencies if needed
pip install -r requirements.txt | tee -a $LOG_FILE

# Verify dataset access
echo "Verifying dataset access..." | tee -a $LOG_FILE
python -c "from datasets import load_dataset; load_dataset('gsm8k', split='train')" | tee -a $LOG_FILE

# Run pretraining
echo "Starting pretraining..." | tee -a $LOG_FILE

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    vishwamai/examples/pretrain_and_upload.py \
    --config_path $CONFIG_PATH \
    --output_dir $OUTPUT_DIR \
    --num_epochs 3 \
    --batch_size 8 \
    --gradient_accumulation 4 \
    --learning_rate 1.2e-4 \
    --fp16 \
    --gradient_checkpointing \
    --use_memory \
    --use_tree \
    --use_cache \
    2>&1 | tee -a $LOG_FILE

TRAINING_STATUS=$?

if [ $TRAINING_STATUS -eq 0 ]; then
    echo "Training completed successfully!" | tee -a $LOG_FILE
    
    # Upload to Hub
    echo "Uploading to Hugging Face Hub..." | tee -a $LOG_FILE
    python vishwamai/examples/pretrain_and_upload.py \
        --mode upload_only \
        --model_path $OUTPUT_DIR \
        2>&1 | tee -a $LOG_FILE
        
    if [ $? -eq 0 ]; then
        echo "Upload completed successfully!" | tee -a $LOG_FILE
    else
        echo "Upload failed! Check $LOG_FILE for details." | tee -a $LOG_FILE
        exit 1
    fi
else
    echo "Training failed! Check $LOG_FILE for details." | tee -a $LOG_FILE
    exit 1
fi

# Monitor GPU utilization in background
nvidia-smi dmon -i 0 -s u -d 1 > "$LOG_DIR/gpu_stats_$TIMESTAMP.log" &
NVIDIA_SMI_PID=$!

# Cleanup
cleanup() {
    kill $NVIDIA_SMI_PID
    echo "Cleanup completed" | tee -a $LOG_FILE
}

trap cleanup EXIT

# Generate training summary
echo "Generating training summary..." | tee -a $LOG_FILE
python -c "
from pathlib import Path
import json

log_file = Path('$LOG_FILE')
out_dir = Path('$OUTPUT_DIR')

metrics = {
    'training_time': None,
    'peak_memory': None,
    'final_loss': None
}

# Parse log file for metrics
with log_file.open() as f:
    for line in f:
        if 'Training completed in' in line:
            metrics['training_time'] = line.split()[-2]
        if 'GPU memory usage:' in line:
            metrics['peak_memory'] = line.split()[-2]
        if 'final loss:' in line:
            metrics['final_loss'] = line.split()[-1]

# Save summary
with (out_dir / 'training_summary.json').open('w') as f:
    json.dump(metrics, f, indent=2)
" | tee -a $LOG_FILE

echo "Pretraining pipeline completed!" | tee -a $LOG_FILE
echo "Logs available at: $LOG_FILE"
echo "Training summary available at: $OUTPUT_DIR/training_summary.json"
