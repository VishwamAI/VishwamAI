import torch
from pathlib import Path
import logging
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from vishwamai.architecture import init_model, create_config_from_template
from vishwamai.training import VishwamaiTrainer
from vishwamai.conceptual_tokenizer import ConceptualTokenizer
from vishwamai.generate import GenerationConfig, VishwamaiGenerator  # Add VishwamaiGenerator import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_math_dataloaders(batch_size: int = 32):
    """Load and prepare DeepMind mathematics dataset"""
    logger.info("Loading mathematics dataset...")
    
    # Load multiple mathematics categories
    categories = [
        "algebra__linear_1d",
        "arithmetic__mul_div_multiple",
        "arithmetic__add_sub_multiple",
        "calculus__differentiate"
    ]
    
    train_datasets = []
    test_datasets = []
    
    for category in categories:
        dataset = load_dataset('math_dataset/' + category)
        train_datasets.append(dataset['train'])
        test_datasets.append(dataset['test'])
    
    # Combine datasets
    train_dataset = concatenate_datasets(train_datasets)
    test_dataset = concatenate_datasets(test_datasets)
    
    logger.info(f"Loaded {len(train_dataset)} training examples and {len(test_dataset)} test examples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=math_collate_fn
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=math_collate_fn
    )
    
    return train_loader, val_loader

def math_collate_fn(batch, tokenizer=None):
    """Custom collate function for math problems"""
    if tokenizer is None:
        tokenizer = ConceptualTokenizer(vocab_size=32000, max_length=512)
        
    questions = [item['question'] for item in batch]
    answers = [item['answer'] for item in batch]
    
    # Combine question and answer with special separator
    combined_texts = [q + " [SEP] " + a for q, a in zip(questions, answers)]
    
    # Tokenize using instance method instead of class method
    tokenized = tokenizer.batch_encode_with_concepts(
        texts=combined_texts,
        max_length=512,
        max_concepts=8
    )
    
    return {
        'input_ids': tokenized['token_ids'],
        'concept_ids': tokenized['concept_ids']
    }

def train_math_model(
    output_dir: str = "math_models",
    model_size: str = "2b",
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    use_wandb: bool = True
):
    """Train Vishwamai model on mathematics dataset"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and tokenizer
    config = create_config_from_template(
        model_size,
        output_path=output_dir / "config.json",
        vocab_size=32000,  # Smaller vocabulary for math-specific tokenizer
        max_position_embeddings=1024
    )
    
    tokenizer = ConceptualTokenizer(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings
    )
    
    # Add subject-specific tokens for math, physics, and biology
    tokenizer.subject_specific_tokens.update({
        "math": 6,
        "physics": 7,
        "biology": 8
    })
    
    # Create dataloaders
    train_loader, val_loader = create_math_dataloaders(batch_size)
    
    # Initialize model
    logger.info(f"Initializing {model_size} model for mathematics training...")
    model = init_model(config, tokenizer=tokenizer)
    
    # Initialize trainer
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        optimizer_class=torch.optim.AdamW,
        use_wandb=use_wandb
    )
    
    # Train
    logger.info("Starting mathematics training...")
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
    
    # Test model on math problems
    logger.info("Testing model on mathematics problems...")
    test_problems = [
        "Solve for x: 2x + 3 = 11",
        "What is the derivative of x^2 + 3x?",
        "Calculate: 125 ÷ 5 × 3"
    ]
    
    generator = VishwamaiGenerator(
        model=model,
        tokenizer=tokenizer,
        config=GenerationConfig(
            max_length=100,
            temperature=0.7,
            top_p=0.9
        )
    )
    
    for problem in test_problems:
        answer = generator.generate(problem)
        logger.info(f"\nProblem: {problem}\nAnswer: {answer[0]}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="math_models")
    parser.add_argument("--model-size", type=str, default="2b")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    
    train_math_model(
        output_dir=args.output_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb
    )
