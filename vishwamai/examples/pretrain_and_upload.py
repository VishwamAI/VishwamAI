"""
Script to pretrain VishwamAI model and upload to Hugging Face Hub.
Handles all components and provides progress monitoring.
"""

import os
import time
import torch
import wandb
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from huggingface_hub import HfApi, Repository
from transformers import TrainingArguments
from ..model import VishwamAIModel
from ..neural_memory import ReasoningMemoryTransformer, MemoryConfig
from ..tree_of_thoughts import TreeOfThoughts, TreeConfig
from ..cache_augmentation import DifferentiableCacheAugmentation, CacheConfig
from ..trainer import VishwamAIPretrainer

@dataclass
class PretrainConfig:
    """Configuration for pretraining process."""
    # Model paths
    output_dir: str = "pretrain_output"
    hub_model_id: str = "kasinadhsarma/vishwamai-model"
    
    # Training params
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation: int = 4
    learning_rate: float = 1.2e-4
    
    # Component configs
    memory_size: int = 2048
    tree_beam_width: int = 4
    cache_size: int = 65536
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Monitoring
    wandb_project: str = "vishwamai"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 10

class ModelPretrainer:
    """Handles pretraining and uploading of VishwamAI model."""
    
    def __init__(self, config: PretrainConfig):
        self.config = config
        self.setup_environment()
        
    def setup_environment(self):
        """Initialize directories and logging."""
        os.makedirs(self.config.output_dir, exist_ok=True)
        wandb.init(project=self.config.wandb_project)
        self.start_time = time.time()
        
    def initialize_components(self):
        """Initialize model and all enhancement components."""
        print("Initializing components...")
        
        # Base model
        self.model = VishwamAIModel.from_pretrained(
            "kasinadhsarma/vishwamai-base"
        ).to(self.config.device)
        
        # Memory module
        self.memory = ReasoningMemoryTransformer(
            MemoryConfig(
                hidden_size=self.model.config.hidden_size,
                memory_size=self.config.memory_size
            )
        ).to(self.config.device)
        
        # Tree module
        self.tree = TreeOfThoughts(
            model=self.model,
            config=TreeConfig(
                beam_width=self.config.tree_beam_width
            )
        ).to(self.config.device)
        
        # Cache module
        self.cache = DifferentiableCacheAugmentation(
            CacheConfig(
                hidden_size=self.model.config.hidden_size,
                max_cache_length=self.config.cache_size
            )
        ).to(self.config.device)
        
        print("Components initialized successfully.")
        
    def setup_trainer(self, train_dataset, eval_dataset):
        """Configure trainer with all components."""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            fp16=True,
            gradient_checkpointing=True,
            report_to=["tensorboard", "wandb"],
            push_to_hub=True,
            hub_model_id=self.config.hub_model_id
        )
        
        self.trainer = VishwamAIPretrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            memory_module=self.memory,
            tree_module=self.tree,
            cache_module=self.cache
        )
        
    def train(self):
        """Execute training process."""
        print("Starting training...")
        
        try:
            self.trainer.train()
            training_time = (time.time() - self.start_time) / 3600
            print(f"Training completed in {training_time:.2f} hours")
            
        except Exception as e:
            print(f"Training failed: {str(e)}")
            raise
            
    def upload_to_hub(self):
        """Upload model and components to Hugging Face Hub."""
        print(f"Uploading to {self.config.hub_model_id}...")
        
        try:
            # Create repository
            api = HfApi()
            repo_url = api.create_repo(
                repo_id=self.config.hub_model_id,
                exist_ok=True
            )
            
            # Clone repository
            repo = Repository(
                local_dir=self.config.output_dir,
                clone_from=repo_url
            )
            
            # Save all components
            self.trainer.save_model()
            self.memory.save_pretrained(self.config.output_dir)
            self.tree.save_pretrained(self.config.output_dir)
            self.cache.save_pretrained(self.config.output_dir)
            
            # Push to hub
            repo.push_to_hub()
            print("Upload completed successfully!")
            
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            raise

def main():
    """Main execution function."""
    # Load datasets
    from datasets import load_dataset
    
    train_datasets = []
    for ds_name in ["gsm8k", "leetcode", "math"]:
        try:
            ds = load_dataset(ds_name, split="train")
            train_datasets.append(ds)
        except Exception as e:
            print(f"Failed to load {ds_name}: {str(e)}")
    
    if not train_datasets:
        raise ValueError("No training datasets available")
    
    from datasets import concatenate_datasets
    combined_train = concatenate_datasets(train_datasets)
    eval_dataset = load_dataset("mmlu", split="validation")
    
    # Initialize pretrainer
    config = PretrainConfig()
    pretrainer = ModelPretrainer(config)
    
    # Run pretraining pipeline
    pretrainer.initialize_components()
    pretrainer.setup_trainer(combined_train, eval_dataset)
    pretrainer.train()
    pretrainer.upload_to_hub()

if __name__ == "__main__":
    main()
