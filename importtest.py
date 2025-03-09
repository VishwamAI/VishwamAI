#!/usr/bin/env python3
"""
VishwamAI GPU Model Import Test
Runs continuous testing of GPU-optimized model imports and basic functionality
"""

import os
import sys
import time
import torch
import traceback
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
TEST_INTERVAL = 30  # seconds between test runs

def test_gpu_imports() -> Tuple[bool, Dict[str, List[str]]]:
    """Test imports of GPU-optimized models and components"""
    results = {
        "success": [],
        "failed": []
    }
    
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        # Basic ML imports
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        results["success"].append("PyTorch imports successful")
        
        # CUDA checks
        if torch.cuda.is_available():
            results["success"].append(f"CUDA available - version {torch.version.cuda}")
            results["success"].append(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                results["success"].append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            results["failed"].append("CUDA not available")

        # DeepGEMM imports
        from vishwamai.models.gpu.optimizations.deepgemm import (
            DistributedGEMM, GEMMConfig, get_best_configs,
            get_num_sms, layernorm
        )
        results["success"].append("DeepGEMM imports successful")

        # Kernel layer imports
        from vishwamai.models.gpu.kernel_layers import (
            DeepGEMMLinear,
            DeepGEMMLayerNorm,
            DeepGEMMGroupedLinear
        )
        results["success"].append("GPU kernel layers imports successful")

        # Transformer components
        from vishwamai.models.gpu.transformer import (
            TransformerComputeLayer,
            TransformerMemoryLayer, 
            HybridThoughtAwareAttention
        )
        results["success"].append("GPU transformer components imports successful")

        # MoE components
        from vishwamai.models.gpu.moe import OptimizedMoE
        results["success"].append("GPU MoE imports successful")

        # Test class instantiations if CUDA available
        if torch.cuda.is_available():
            test_configs = [
                ("DeepGEMMLinear", lambda: DeepGEMMLinear(128, 256)),
                ("DeepGEMMLayerNorm", lambda: DeepGEMMLayerNorm(128)),
                ("DeepGEMMGroupedLinear", lambda: DeepGEMMGroupedLinear(128, 256, 4)),
                ("TransformerComputeLayer", lambda: TransformerComputeLayer(512, 8)),
                ("TransformerMemoryLayer", lambda: TransformerMemoryLayer(512, 8)),
                ("HybridThoughtAwareAttention", lambda: HybridThoughtAwareAttention(512, 8)),
                ("OptimizedMoE", lambda: OptimizedMoE(512, 8))
            ]

            for name, instantiation_func in test_configs:
                try:
                    model = instantiation_func()
                    # Test with dummy input
                    with torch.no_grad():
                        x = torch.randn(2, 16, 128 if '128' in str(instantiation_func) else 512).cuda()
                        _ = model(x)
                    results["success"].append(f"{name} instantiation and forward pass successful")
                except Exception as e:
                    results["failed"].append(f"{name} test failed: {str(e)}")
                    traceback.print_exc()

    except ImportError as e:
        results["failed"].append(f"Import Error: {str(e)}")
        traceback.print_exc()
        return False, results
    except Exception as e:
        results["failed"].append(f"Unexpected Error: {str(e)}")
        traceback.print_exc()
        return False, results

    return len(results["failed"]) == 0, results

def print_results(attempt: int, results: Dict[str, List[str]], is_retry: bool = False) -> None:
    """Print test results in a formatted way"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if is_retry:
        print(f"\n=== Retry Attempt {attempt + 1} Results ({timestamp}) ===")
    else:
        print(f"\n=== Test Run Results ({timestamp}) ===")
    
    print("\nSuccesses:")
    for success in results["success"]:
        print(f"✓ {success}")
    
    if results["failed"]:
        print("\nFailures:")
        for failure in results["failed"]:
            print(f"✗ {failure}")

def main():
    """Main test loop"""
    print("Starting VishwamAI GPU model import test...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            attempt = 0
            success = False
            
            while attempt < MAX_RETRIES and not success:
                if attempt > 0:
                    print(f"\nRetrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                    
                print(f"\nStarting test run {attempt + 1}/{MAX_RETRIES}")
                success, results = test_gpu_imports()
                print_results(attempt, results, is_retry=(attempt > 0))
                
                if success:
                    print("\n✓ All tests passed successfully!")
                    break
                    
                attempt += 1
            
            if not success:
                print("\n✗ Maximum retry attempts reached. Some tests are still failing.")
            
            print(f"\nWaiting {TEST_INTERVAL} seconds before next test run...")
            time.sleep(TEST_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nTest loop stopped by user")
    except Exception as e:
        print(f"\nTest loop stopped due to error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()

