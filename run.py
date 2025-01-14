import argparse
from vishwamai.training import train_model
import json
import os  # Added import
import subprocess

def parse_args():
    parser = argparse.ArgumentParser(description="Train VishwamAI DeepMind Math Model")
    parser.add_argument('--config', type=str, required=True, help='Path to the config JSON file')
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data Parquet file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to the validation data Parquet file')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    return parser.parse_args()

def main():
    args = parse_args()
    
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
    
    # Initiate training
    train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        config=config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()