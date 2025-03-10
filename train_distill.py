"""
Training script for VishwamAI transformer with DuckDB logging.
"""

import os
import json
import time
import jax
from typing import Any, Dict
from vishwamai.training import create_trainer
from vishwamai.transformer import create_vishwamai_transformer
from safetensors.flax import save_file

def load_config(config_path: str) -> Dict[str, Any]:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_training_data_loader(
    data_path: str,
    batch_size: int,
    sequence_length: int
):
    """Create data loader for training data."""
    # Implement your data loading logic here
    # This should return batched data in the format expected by the model
    pass

def main():
    # Load configuration
    config = load_config('vishwamai/configs/config_16b.json')
    
    # Setup TPU devices and distributed training
    num_devices = jax.device_count()
    devices = jax.devices()[:num_devices]
    
    # Create experiment name with timestamp
    experiment_name = f"vishwamai_{int(time.time())}"
    
    # Setup data loaders
    train_loader = create_training_data_loader(
        config['train_data_path'],
        config['training']['batch_size'],
        config['model_config']['max_seq_len']
    )
    
    eval_loader = create_training_data_loader(
        config['eval_data_path'],
        config['training']['batch_size'],
        config['model_config']['max_seq_len']
    )
    
    # Create trainer
    trainer = create_trainer(
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_name=experiment_name,
        db_path="training_logs.db"  # DuckDB database path
    )
    
    print(f"Starting training on {num_devices} devices...")
    print(f"Experiment name: {experiment_name}")
    print(f"Logging metrics to: training_logs.db")
    
    # Run training
    trainer.train()
    
    # Save model as .safetensors
    save_dir = 'final_model'
    os.makedirs(save_dir, exist_ok=True)
    params = jax.device_get(trainer.pipeline.state.params)
    try:
        save_file(params, f"{save_dir}/model.safetensors")
    except Exception as e:
        print(f"Error saving model: {e}")
        return
    print("Training completed!")
    print("Final logs exported to 'logs' directory")
    print(f"Database available at: training_logs.db")
    print(f"Model saved as .safetensors in {save_dir}")

if __name__ == "__main__":
    main()
