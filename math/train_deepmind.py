import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
from vishwamai.training import VishwamaiTrainer
from vishwamai.model import VishwamaiModel, VishwamaiConfig
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

def load_math_dataset(categories=None):
    """Load DeepMind mathematics dataset"""
    if categories is None:
        categories = [
            "algebra__linear_1d",
            "arithmetic__mul_div_multiple",
            "arithmetic__add_sub_multiple",
            "calculus__differentiate"
        ]
    
    datasets = []
    for category in categories:
        dataset = load_dataset('math_dataset/' + category)
        datasets.append(dataset)
    
    # Combine datasets
    train_datasets = [d['train'] for d in datasets]
    test_datasets = [d['test'] for d in datasets]
    
    def preprocess_function(examples):
        """Convert examples to model inputs"""
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        for question, answer in zip(examples["question"], examples["answer"]):
            # Combine question and answer for training
            text = f"Q: {question} A: {answer}"
            # Tokenize (implement actual tokenization based on your tokenizer)
            tokenized = ConceptualTokenizer.encode(text)
            model_inputs["input_ids"].append(tokenized)
            model_inputs["attention_mask"].append([1] * len(tokenized))
            
        return model_inputs
    
    # Process datasets
    train_dataset = train_datasets[0].map(
        preprocess_function,
        batched=True,
        remove_columns=train_datasets[0].column_names
    )
    test_dataset = test_datasets[0].map(
        preprocess_function,
        batched=True,
        remove_columns=test_datasets[0].column_names
    )
    
    return train_dataset, test_dataset

def train_math_model(
    output_dir: str = "math_model",
    model_size: str = "base",
    learning_rate: float = 1e-4,
    batch_size: int = 16,
    num_epochs: int = 3
):
    """Train model on mathematics dataset"""
    
    # Initialize configuration
    config = VishwamaiConfig(
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=3072,
        max_position_embeddings=2048
    )
    
    # Initialize tokenizer
    tokenizer_config = ConceptualTokenizerConfig(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings,
        concept_tokens=["[MATH]", "[ALGEBRA]", "[ARITHMETIC]", "[CALCULUS]"]
    )
    tokenizer = ConceptualTokenizer(tokenizer_config)
    
    # Initialize model
    model = VishwamaiModel(config)
    
    # Load dataset
    train_dataset, eval_dataset = load_math_dataset()
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=default_data_collator
    )
    
    # Initialize trainer
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        device="cuda" if torch.cuda.is_available() else "cpu",
        optimizer_class=torch.optim.AdamW,
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR
    )
    
    # Train
    trainer.train(
        num_epochs=num_epochs,
        save_dir=output_dir,
        evaluation_steps=100,
        save_steps=1000,
        gradient_accumulation_steps=4
    )
    
    return trainer

if __name__ == "__main__":
    train_math_model()
