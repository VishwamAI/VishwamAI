"""
GPU optimization modules for VishwamAI with smallpond integration
"""

from .optimizations.deepgemm import (
    gemm_fp8_fp8_bf16_nt,
    get_best_configs,
    get_num_sms,
    layernorm,
    init_deepgemm_kernels,
    set_warps
)

from .optimizations.deep_ep import (
    Buffer,
    get_buffer,
    set_num_sms,
    init_expert_parallel,
    get_optimal_dispatch_config
)

from .optimizations.flash_mla import (
    Flash_fwd_kernel_traits_mla,
    Flash_fwd_mla_params,
    run_mha_fwd_splitkv_mla,
    flash_mla_with_kvcache,
    get_mla_metadata,
    init_flash_kernels
)

from .optimizations.eplb import (
    EPLB,
    init_load_balancer,
    get_load_stats
)

# Initialize kernels on import
init_deepgemm_kernels()
init_flash_kernels()
init_expert_parallel()
init_load_balancer()

# Enable smallpond integration by default
import smallpond
try:
    sp = smallpond.init()
except:
    sp = None