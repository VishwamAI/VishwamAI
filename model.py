import json
import torch
from vishwamai import VishwamaiConfig, init_model
from vishwamai.dataprocessing import DataCollatorForLanguageModeling, VishwamaiDataset
from vishwamai.training import VishwamaiTrainer, GenerationConfig
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

def train_with_sample_data(json_path="data/sample.json"):
    # Load sample data
    with open(json_path, "r") as f:
        raw_data = json.load(f)

    # Prepare dataset
    dataset = VishwamaiDataset(raw_data)

    # Initialize tokenizer
    tokenizer_config = ConceptualTokenizerConfig()
    tokenizer = ConceptualTokenizer(tokenizer_config)

    # Model initialization
    model_config = VishwamaiConfig()
    model = init_model(model_config)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    # Trainer setup
    trainer = VishwamaiTrainer(
        model=model,
        config=GenerationConfig(),
        dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Training
    trainer.train()
    print("Training complete. You can now interact with the model.")
