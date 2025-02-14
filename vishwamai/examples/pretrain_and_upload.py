import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi, upload_folder, HfFolder
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from vishwamai.model_utils import load_model
from vishwamai.trainer import Trainer, TrainingArgs

def setup_tokenizer():
    """Initialize BERT tokenizer with custom configuration"""
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased",
        model_max_length=512,  # Reduced for CPU testing
        do_lower_case=True,
        truncation_side="right",
        padding_side="right",
        use_fast=True
    )
    
    # Add special tokens for our model's needs
    special_tokens = {
        "additional_special_tokens": [
            "[MEMORY]",  # For neural memory module
            "[REASONING]",  # For tree of thoughts
            "[CACHE]",  # For cache augmentation
            "[STEP]",  # For step-by-step reasoning
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def prepare_text(examples):
    """Prepare text for BERT-style encoding"""
    if "question" in examples:
        text = examples["question"]
        if "solution" in examples:
            text = f"[STEP] Question: {text} [STEP] Solution: {examples['solution']}"
    else:
        text = examples["text"] if "text" in examples else str(examples)
    text = f"[MEMORY] [CACHE] [REASONING] {text}"
    return text

def prepare_datasets(tokenizer, max_length=512):
    """Load and prepare datasets for pretraining"""
    print("Loading datasets...")
    # Load small subset for testing
    datasets = {
        "gsm8k": load_dataset("openai/gsm8k", split="train[:100]"),  # Limited samples for testing
        "mmlu": load_dataset("cais/mmlu", split="train[:100]"),
        "mmlu_pro": load_dataset("TIGER-Lab/MMLU-Pro", split="train[:100]"),
        "mmmlu": load_dataset("openai/MMMLU", split="train[:100]")
    }
    
    def tokenize_function(examples):
        text = prepare_text(examples)
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
            return_token_type_ids=True
        )
    
    tokenized_datasets = {}
    for name, dataset in datasets.items():
        print(f"Processing {name} dataset...")
        tokenized_datasets[name] = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
    
    return tokenized_datasets

def create_dataloaders(tokenized_datasets, batch_size=4):  # Smaller batch size for CPU
    """Create DataLoaders for training"""
    def collate_fn(examples):
        input_ids = torch.stack([example['input_ids'] for example in examples])
        attention_mask = torch.stack([example['attention_mask'] for example in examples])
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    dataloaders = {}
    for name, dataset in tokenized_datasets.items():
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=False  # Disabled for CPU training
        )
    
    return dataloaders

def pretrain_model(model, train_dataloader, eval_dataloader=None):
    """Pretrain the model"""
    training_args = TrainingArgs(
        output_dir="pretrain_checkpoints",
        num_epochs=1,  # Reduced for testing
        batch_size=4,  # Smaller batch size
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=10,  # Reduced for testing
        weight_decay=0.1,
        max_grad_norm=1.0,
        save_steps=50,  # More frequent saving for testing
        logging_steps=10,
        use_fsdp=False,  # Disabled for CPU
        mixed_precision=False,  # Disabled for CPU
        gradient_checkpointing=False  # Disabled for CPU
    )
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        args=training_args
    )
    
    print("Starting pretraining...")
    trainer.train()
    
    return trainer

def upload_to_hub(model_dir: str, repo_name: str, token: str = None):
    """Upload model to Hugging Face Hub"""
    if token:
        HfFolder.save_token(token)
    
    print(f"Uploading model to {repo_name}...")
    api = HfApi()
    
    try:
        api.create_repo(repo_name, private=True)
    except Exception as e:
        print(f"Repository creation error (might already exist): {e}")
    
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"Model uploaded successfully to: https://huggingface.co/{repo_name}")

def main():
    # Initialize tokenizer
    tokenizer = setup_tokenizer()
    
    # Load and prepare datasets
    print("Preparing datasets...")
    tokenized_datasets = prepare_datasets(tokenizer)
    dataloaders = create_dataloaders(tokenized_datasets)
    
    # Create a small combined dataset for testing
    combined_dataloader = DataLoader(
        torch.utils.data.ConcatDataset([dl.dataset for dl in dataloaders.values()]),
        batch_size=4,  # Small batch size for CPU
        shuffle=True,
        pin_memory=False
    )
    
    # Load model with reduced size for CPU testing
    print("Loading model...")
    config_path = Path(__file__).parent.parent / "configs" / "config_optimized.json"
    
    # Force CPU mode and smaller model size
    model = load_model(
        config_path,
        device="cpu",
        hidden_size=768,  # Reduced size
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=1024
    )
    
    # Pretrain model
    trainer = pretrain_model(
        model,
        train_dataloader=combined_dataloader,
        eval_dataloader=dataloaders['gsm8k']
    )
    
    # Save model and tokenizer
    output_dir = "vishwamai_pretrained"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving model...")
    state_dict = model.state_dict()
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(output_dir)
    
    # Copy config and model card
    print("Copying configuration files...")
    import shutil
    shutil.copy("MODEL_CARD.md", os.path.join(output_dir, "README.md"))
    shutil.copy(config_path, os.path.join(output_dir, "config.json"))
    
    # Upload to Hugging Face Hub
    token = os.getenv("HF_TOKEN")
    if not token:
        print("No HF_TOKEN found in environment. Please log in using `huggingface-cli login`")
    
    print("Uploading to Hugging Face Hub...")
    upload_to_hub(
        model_dir=output_dir,
        repo_name="kasinadhsarma/vishwamai-model",
        token=token
    )

if __name__ == "__main__":
    torch.manual_seed(42)
    main()
