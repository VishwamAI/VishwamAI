#!/bin/bash

# HuggingFace pre-training script for Vishwamai models with TPU support

# Default values
MODEL_TYPE="moe_mla_transformer"
MODEL_SIZE="base"
DATA_PATH=""
OUTPUT_DIR="checkpoints/"
TOKENIZER_PATH=""
CONFIG_PATH=""

# TPU configuration
TPU_NAME=""
TPU_ZONE="us-central1-f"
NUM_TPU_CORES=8
TPU_TOPOLOGY="2x2"  # or "2x2x2" for v3-32

# Training hyperparameters
BATCH_SIZE=32
GRAD_ACC_STEPS=4
MAX_STEPS=100000
WARMUP_STEPS=2000
SAVE_STEPS=1000
EVAL_STEPS=500
LR=5e-4
MAX_LENGTH=2048

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tokenizer_path)
            TOKENIZER_PATH="$2"
            shift 2
            ;;
        --config_path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --grad_acc_steps)
            GRAD_ACC_STEPS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --save_steps)
            SAVE_STEPS="$2"
            shift 2
            ;;
        --eval_steps)
            EVAL_STEPS="$2"
            shift 2
            ;;
        --learning_rate)
            LR="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --tpu_name)
            TPU_NAME="$2"
            shift 2
            ;;
        --tpu_zone)
            TPU_ZONE="$2"
            shift 2
            ;;
        --num_tpu_cores)
            NUM_TPU_CORES="$2"
            shift 2
            ;;
        --tpu_topology)
            TPU_TOPOLOGY="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data_path is required"
    exit 1
fi

if [ -z "$TOKENIZER_PATH" ]; then
    echo "Error: --tokenizer_path is required"
    exit 1
fi

if [ -z "$TPU_NAME" ]; then
    echo "Error: --tpu_name is required"
    exit 1
fi

# Create output directory
mkdir -p $OUTPUT_DIR

# Prepare TPU specific environment variables
export TPU_NAME=$TPU_NAME
export TPU_CHIPS_PER_HOST_BOUNDS=$TPU_TOPOLOGY
export TPU_HOST_BOUNDS=$TPU_TOPOLOGY
export XLA_USE_BF16=1  # Enable bfloat16 for better performance

# Prepare training script command
TRAINING_ARGS="--model_type $MODEL_TYPE \
               --model_size $MODEL_SIZE \
               --data_path $DATA_PATH \
               --output_dir $OUTPUT_DIR \
               --tokenizer_path $TOKENIZER_PATH \
               --per_device_train_batch_size $BATCH_SIZE \
               --gradient_accumulation_steps $GRAD_ACC_STEPS \
               --max_steps $MAX_STEPS \
               --warmup_steps $WARMUP_STEPS \
               --save_steps $SAVE_STEPS \
               --eval_steps $EVAL_STEPS \
               --learning_rate $LR \
               --max_length $MAX_LENGTH \
               --use_tpu \
               --tpu_name $TPU_NAME \
               --tpu_zone $TPU_ZONE \
               --num_tpu_cores $NUM_TPU_CORES \
               --bf16 \
               --use_flash_attention \
               --gradient_checkpointing \
               --ddp_find_unused_parameters False"

# Add optional config path
if [ ! -z "$CONFIG_PATH" ]; then
    TRAINING_ARGS="$TRAINING_ARGS --config_path $CONFIG_PATH"
fi

# Export Python path
export PYTHONPATH="."

# Run training
echo "Starting TPU training..."
echo "Model type: $MODEL_TYPE"
echo "Model size: $MODEL_SIZE"
echo "TPU name: $TPU_NAME"
echo "TPU topology: $TPU_TOPOLOGY"
echo "Number of TPU cores: $NUM_TPU_CORES"
echo "Batch size per core: $BATCH_SIZE"
echo "Gradient accumulation steps: $GRAD_ACC_STEPS"
echo "Total batch size: $((BATCH_SIZE * NUM_TPU_CORES * GRAD_ACC_STEPS))"

python vishwamai/scripts/train_model.py \
    $TRAINING_ARGS \
    2>&1 | tee $OUTPUT_DIR/training.log

echo "Training completed!"
echo "Logs saved to: $OUTPUT_DIR/training.log"
