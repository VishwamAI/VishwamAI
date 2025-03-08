# VishwamAI/configs/tokenizer_config.py
"""
Settings for configuring and training the VishwamAI tokenizer,
based on Hugging Face's tokenizer framework.
"""

class TokenizerConfig:
    # General tokenizer settings
    TOKENIZER_TYPE = "BPE"  # Options: "BPE", "WordPiece", "Unigram"
    VOCAB_SIZE = 50000  # Vocabulary size for the tokenizer
    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    MAX_LENGTH = 512  # Maximum sequence length

    # Pretrained tokenizer settings
    USE_PRETRAINED = True
    PRETRAINED_MODEL = "bert-base-uncased"  # Base model for tokenizer
    PRETRAINED_PATH = "huggingface.co/bert-base-uncased"

    # Custom training settings
    TRAIN_CUSTOM = False
    TRAINING_DATA_PATH = "data/train_corpus.txt"  # Path to training data
    MIN_FREQUENCY = 2  # Minimum token frequency for inclusion
    SAVE_PATH = "tokenizers/vishwamai_tokenizer"  # Save directory

    @staticmethod
    def load_tokenizer():
        """Load or configure the tokenizer based on settings."""
        from transformers import AutoTokenizer
        if TokenizerConfig.USE_PRETRAINED and not TokenizerConfig.TRAIN_CUSTOM:
            return AutoTokenizer.from_pretrained(TokenizerConfig.PRETRAINED_MODEL)
        elif TokenizerConfig.TRAIN_CUSTOM:
            raise NotImplementedError("Custom tokenizer training not implemented yet.")
        else:
            raise ValueError("Must specify either pretrained or custom tokenizer.")

if __name__ == "__main__":
    # Test the configuration
    tokenizer = TokenizerConfig.load_tokenizer()
    print("Tokenizer Loaded:", tokenizer)
    print("Vocab Size:", len(tokenizer))