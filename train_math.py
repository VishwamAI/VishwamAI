import torch
from pathlib import Path
import logging
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

from vishwamai.architecture import init_model, create_config_from_template
from vishwamai.training import VishwamaiTrainer
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
from vishwamai.generate import GenerationConfig, VishwamaiGenerator  # Add VishwamaiGenerator import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_math_dataloaders(batch_size: int = 32, dataset_type: str = "gsm8k"):
    """Load and prepare mathematics dataset"""
    logger.info(f"Loading {dataset_type} dataset...")
    
    if dataset_type == "gsm8k":
        # Load GSM8K dataset from parquet files
        train_dataset = load_dataset(
            'parquet',
            data_files={'train': 'gsm8k/train-00000-of-00001.parquet'},
            split='train'
        )
        test_dataset = load_dataset(
            'parquet',
            data_files={'test': 'gsm8k/test-00000-of-00001.parquet'},
            split='test'
        )
    else:
        # Load DeepMind mathematics dataset
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

def math_collate_fn(batch, tokenizer=None, dataset_type="gsm8k"):
    """Custom collate function for math problems"""
    if tokenizer is None:
        config = ConceptualTokenizerConfig(vocab_size=32000, max_length=512)
        tokenizer = ConceptualTokenizer(config)
    
    if dataset_type == "gsm8k":
        questions = []
        answers = []
        for item in batch:
            # Format GSM8K problems with question and answer
            question = f"Question: {item['question']}\nLet's solve this step by step:"
            answer = f"Answer: {item['answer']}"
            questions.append(question)
            answers.append(answer)
    else:
        questions = [item['question'] for item in batch]
        answers = None
    
    # First encode all questions
    encoded_batch = []
    for question in questions:
        encoded = tokenizer.encode_with_concepts(
            text=question,
            max_length=512,
            max_concepts=10
        )
        encoded_batch.append(encoded)
    
    # Pad sequences
    max_len = max(len(item['input_ids']) for item in encoded_batch)
    
    # Create padded tensors and attention masks
    input_ids_list = []
    attention_mask_list = []
    
    for item in encoded_batch:
        padding_length = max_len - len(item['input_ids'])
        input_ids = item['input_ids'] + [tokenizer.config.pad_id] * padding_length
        attention_mask = [1] * len(item['input_ids']) + [0] * padding_length
        
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
    
    input_ids = torch.tensor(input_ids_list)
    attention_mask = torch.tensor(attention_mask_list)
    
    concept_ids = torch.tensor([
        item['concept_ids'] + [0] * (10 - len(item['concept_ids']))  # Pad concepts to max 10
        for item in encoded_batch
    ])
    
    # Encode answers for training if needed
    if answers:
        encoded_answers = []
        for answer in answers:
            encoded = tokenizer.encode_with_concepts(
                text=answer,
                max_length=512,
                max_concepts=10
            )
            encoded_answers.append(encoded)
        
        # Get max length for answer padding
        max_answer_len = max(len(item['input_ids']) for item in encoded_answers)
        answer_tensor = torch.tensor([
            item['input_ids'] + [tokenizer.config.pad_id] * (max_answer_len - len(item['input_ids']))
            for item in encoded_answers
        ])
    else:
        answer_tensor = None

    # Create return dictionary with required fields
    return_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'concept_ids': concept_ids,
    }
    
    # Only add labels if we have answer_tensor
    if answer_tensor is not None:
        return_dict['labels'] = answer_tensor
        # Create labels attention mask (same as input attention mask for now)
        return_dict['labels_attention_mask'] = attention_mask
    
    return return_dict

def train_math_model(
    output_dir: str = "math_models",
    model_size: str = "2b",
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 5e-5,
    use_wandb: bool = True,
    dataset_type: str = "gsm8k"
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
    train_loader, val_loader = create_math_dataloaders(batch_size, dataset_type)
    
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
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest at the farmers market daily for $2 per egg. How much money does she make every day at the farmers market?",
        "A robe takes 2 blue pieces of cloth and 5 white pieces of cloth. If I have 18 blue pieces and 45 white pieces, how many complete robes can I make?",
        "John has 5 times as many marbles as Peter. If Peter has 8 marbles, how many marbles does John have?"
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
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "deepmind"], default="gsm8k")
    
    args = parser.parse_args()
    
    train_math_model(
        output_dir=args.output_dir,
        model_size=args.model_size,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_wandb=not args.no_wandb,
        dataset_type=args.dataset
    )
