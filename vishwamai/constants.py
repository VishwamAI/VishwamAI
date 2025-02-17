from typing import Literal

# Global configuration variables
BLOCK_SIZE = 128
GEMM_IMPL: Literal["bf16", "fp8"] = "bf16"
ATTN_IMPL: Literal["naive", "absorb"] = "absorb"
WORLD_SIZE = 1
RANK = 0
