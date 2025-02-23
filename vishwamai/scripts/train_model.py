"""Main training script."""
import argparse
from pathlib import Path
import yaml

from vishwamai.model import build_model
from vishwamai.data import build_dataset, build_dataloader
from vishwamai.training import Trainer
from vishwamai.utils import setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        
    # Setup model and training
    model = build_model(config["model"])
    train_dataset = build_dataset(config["data"], split="train")
    train_loader = build_dataloader(train_dataset, config["training"])
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        **config["training"]
    )
    
    # Train
    trainer.train()

if __name__ == "__main__":
    main()