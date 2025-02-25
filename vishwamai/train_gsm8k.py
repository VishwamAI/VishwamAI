import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from datasets import load_dataset
from vishwamai.training import train, create_train_state
from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.tokenizer import VishwamAITokenizer
from omegaconf import OmegaConf
import os
import logging
from safetensors.flax import save_file
from typing import Dict, Iterator
import random

logger = logging.getLogger(__name__)

class GSM8KProcessor:
    """Processor for GSM8K dataset."""
    
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.dataset.max_length
    
    def format_example(self, example):
        """Format a GSM8K example for training."""
        question = example['question']
        answer = example['answer']
        # Extract final answer from solution
        final_answer = answer.split('####')[-1].strip()
        # Format as instruction and response
        formatted_text = f"Question: {question}\nLet's solve this step by step:\n{answer}\nFinal Answer: {final_answer}"
        return formatted_text
    
    def tokenize_function(self, examples):
        """Tokenize a batch of formatted examples."""
        # Format and tokenize each example
        formatted_texts = [self.format_example(ex) for ex in examples]
        
        tokenized = self.tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
        )
        
        # Create labels for autoregressive training
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def prepare_dataset(self, dataset):
        """Prepare GSM8K dataset."""
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.config.dataset.num_workers,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def collate_fn(self, examples):
        """Collate examples into a batch."""
        batch = {
            "input_ids": np.array([example["input_ids"] for example in examples]),
            "attention_mask": np.array([example["attention_mask"] for example in examples]),
            "labels": np.array([example["labels"] for example in examples]),
        }
        return batch

def create_gsm8k_dataloader(config, split="train") -> Iterator:
    """Create data loader for GSM8K dataset."""
    # Load GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    
    # Initialize tokenizer
    tokenizer = VishwamAITokenizer(
        vocab_size=config.model.vocab_size,
        model_prefix=config.model.name
    )
    
    # Process dataset
    data_processor = GSM8KProcessor(tokenizer, config)
    processed_dataset = data_processor.prepare_dataset(dataset)
    
    def data_iterator():
        """Iterator that yields batches."""
        epoch = 0
        while True:
            # Shuffle at epoch start
            indices = list(range(len(processed_dataset)))
            random.shuffle(indices)
            
            # Create batches
            for i in range(0, len(indices), config.dataset.batch_size):
                batch_indices = indices[i:i + config.dataset.batch_size]
                examples = [processed_dataset[idx] for idx in batch_indices]
                yield data_processor.collate_fn(examples)
            
            epoch += 1
            logger.info(f"Completed epoch {epoch}")
    
    return data_iterator()

def save_model_safetensors(params: Dict, save_path: str):
    """Save model parameters in safetensors format."""
    # Convert parameters to numpy for safetensors compatibility
    numpy_params = jax.tree_map(lambda x: np.array(x), params)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save parameters
    save_file(numpy_params, save_path)
    logger.info(f"Saved model to {save_path}")

def setup_tpu_cluster():
    """Set up JAX TPU cluster configuration."""
    # Get available devices
    devices = jax.devices()
    logger.info(f"Available devices: {devices}")
    
    # Create device mesh for data parallel training
    device_count = len(devices)
    device_mesh = np.array(devices).reshape(device_count)
    
    # Create mesh with data parallel sharding
    mesh = Mesh(device_mesh, ('data',))
    
    # Create sharding rules
    data_sharding = NamedSharding(mesh, P('data'))
    
    return mesh, data_sharding

def main():
    # Load GSM8K specific config
    config_path = "vishwamai/configs/training/gsm8k.yaml"
    config = OmegaConf.load(config_path)
    
    # Set up TPU configuration with modern sharding
    mesh, sharding = setup_tpu_cluster()
    
    # Initialize model with GSM8K specific configuration
    model_config = ModelConfig(**config.model)
    model = VishwamAIModel(model_config)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = create_gsm8k_dataloader(config, split="train")
    val_dataloader = create_gsm8k_dataloader(config, split="validation")
    
    # Create checkpoints directory
    checkpoint_dir = config.checkpointing.dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Custom checkpoint saving function for safetensors
    def save_checkpoint_hook(state, path):
        save_model_safetensors(state.params, f"{path}.safetensors")
    
    logger.info("Starting training...")
    with mesh:
        final_state = train(
            model,
            config,
            train_dataloader,
            val_dataloader=val_dataloader,
            num_steps=config.max_steps,
            log_every=config.logging_steps,
            eval_every=config.eval_steps,
            checkpoint_dir=checkpoint_dir,
            save_checkpoint_fn=save_checkpoint_hook,
            sharding=sharding
        )
    
    # Save final model in safetensors format
    save_model_safetensors(
        final_state.params,
        os.path.join(checkpoint_dir, "gsm8k_final.safetensors")
    )
    
    # Log final metrics
    logger.info("Training completed!")
    logger.info(f"Best loss: {final_state.best_metrics['loss']:.4f}")
    logger.info(f"Best accuracy: {final_state.best_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
