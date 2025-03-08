"""
GPU optimization modules for VishwamAI
"""

from .optimizations.deepgemm import (
    gemm_fp8_fp8_bf16_nt,
    get_best_configs,
    get_num_sms,
    layernorm
)

from .optimizations.deep_ep import (
    Buffer,
    get_buffer,
    set_num_sms
)

from .optimizations.flash_mla import (
    Flash_fwd_kernel_traits_mla,
    Flash_fwd_mla_params,
    run_mha_fwd_splitkv_mla,
    flash_mla_with_kvcache,
    get_mla_metadata
)