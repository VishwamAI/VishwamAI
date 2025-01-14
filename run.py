from vishwamai.parquet_handling import ParquetDataset, ParquetConfig
from vishwamai.training import VishwamaiTrainer
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
from vishwamai.architecture import VishwamaiV1

# Initialize tokenizer
tokenizer_config = ConceptualTokenizerConfig()
tokenizer = ConceptualTokenizer(tokenizer_config)

# Initialize model
model = VishwamaiV1()

# Create datasets
train_dataset = ParquetDataset("train.parquet", tokenizer, ParquetConfig())
eval_dataset = ParquetDataset("eval.parquet", tokenizer, ParquetConfig())

# Initialize trainer with Parquet support
trainer = VishwamaiTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    use_parquet=True,
    parquet_config=ParquetConfig(
        chunk_size=10000,
        batch_size=32,
        num_workers=4
    )
)

# Train as usual
trainer.train()