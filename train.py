import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
from typing import Optional
import wandb

from vishwamai.architecture import init_model, create_config_from_template
from vishwamai.training import VishwamaiTrainer
from vishwamai.conceptual_tokenizer import ConceptualTokenizer
from vishwamai.generate import GenerationConfig, VishwamaiGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer.batch_encode_with_concepts(
            [text], 
            max_length=self.max_length
        )
        return {
            "input_ids": encoded["token_ids"][0],
            "concept_ids": encoded["concept_ids"][0]
        }

def load_data(data_path: str):
    """Load training data from file"""
    with open(data_path, 'r') as f:
        return [line.strip() for line in f]

def train_model(
    train_data_path: str,
    val_data_path: Optional[str] = None,
    model_size: str = "2b",
    output_dir: str = "models",
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    use_wandb: bool = True
):
    """Main training function"""
    
    # Initialize wandb
    if use_wandb:
        wandb.init(project="vishwamai", config={
            "model_size": model_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        })

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    config = create_config_from_template(
        model_size,
        output_path=output_dir / "config.json"
    )

    # Initialize tokenizer
    tokenizer = ConceptualTokenizer(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings
    )

    # Load and prepare data
    logger.info("Loading training data...")
    train_texts = load_data(train_data_path)
    train_dataset = TextDataset(train_texts, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = None
    if val_data_path:
        val_texts = load_data(val_data_path)
        val_dataset = TextDataset(val_texts, tokenizer)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )

    # Initialize model
    logger.info(f"Initializing {model_size} model...")
    model = init_model(config)

    # Initialize trainer
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        optimizer_class=torch.optim.AdamW,
        use_wandb=use_wandb
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train(
        num_epochs=num_epochs,
        save_dir=output_dir,
        evaluation_steps=100,
        save_steps=1000,
        logging_steps=10,
        gradient_accumulation_steps=4,
        max_grad_norm=1.0,
        fp16=True
    )

    # Save final model and tokenizer
    logger.info("Saving final model...")
    trainer.save_model(output_dir / "final_model")
    tokenizer.save(output_dir / "tokenizer.json")

    # Test generation
    generator = VishwamaiGenerator(
        model=model,
        tokenizer=tokenizer,
        config=GenerationConfig()
    )
    
    test_output = generator.generate(
        "This is a test prompt to verify the model works.",
        max_length=100
    )
    logger.info(f"Test generation output: {test_output[0]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str)
    parser.add_argument("--model-size", type=str, default="2b")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    
    train_model(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_size=args.model_size,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb
    )
