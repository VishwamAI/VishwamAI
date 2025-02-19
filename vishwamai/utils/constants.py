"""Constants for Transformer configuration."""

# Distributed training
WORLD_SIZE = 1
RANK = 0

# Attention implementation
ATTN_IMPL = "naive"  # "naive" or "flash"

# Block size for weight quantization
BLOCK_SIZE = 64

# Default tensor type
DEFAULT_DTYPE = "bfloat16"

# Model capacities
MAX_SEQUENCE_LENGTH = 32768
DEFAULT_VOCAB_SIZE = 102400
