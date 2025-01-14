import argparse
from vishwamai.training import train_model
from vishwamai.conceptual_tokenizer import ConceptualTokenizer
import json
import os
import subprocess
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import logging
from transformers import HfArgumentParser

def parse_args():
    parser = argparse.ArgumentParser(description="Train VishwamAI DeepMind Math Model with Hugging Face")
    parser.add_argument('--config', type=str, required=True, help='Path to the config JSON file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data Parquet file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to the validation data Parquet file')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def main():
    args = parse_args()
    setup_logging()
    
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(args.train_data), exist_ok=True)
    
    # Generate parquet files
    try:
        subprocess.run(
            ["python3", os.path.abspath(os.path.join(os.path.dirname(__file__), "data/generate_parquet.py"))],
            check=True
        )
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to generate parquet files.")
    
    # Validate config file path
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found at path: {args.config}")
    
    # Validate training data path
    if not os.path.isfile(args.train_data):
        raise FileNotFoundError(f"Training data file not found at path: {args.train_data}")
    
    # Validate validation data path
    if not os.path.isfile(args.val_data):
        raise FileNotFoundError(f"Validation data file not found at path: {args.val_data}")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initiate training with Hugging Face Trainer
    model = train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        config=config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        local_rank=args.local_rank
    )
    
    # Optionally, save the trained tokenizer
    tokenizer = ConceptualTokenizer.from_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
