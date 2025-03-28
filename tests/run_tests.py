#!/usr/bin/env python3
"""Test runner for VishwamAI kernel tests."""

import os
import sys
import time
import argparse
import subprocess
from typing import List, Optional, Dict, Any
from pathlib import Path

# Import platform detection
try:
    from tests.conftest import PLATFORMS
except ImportError:
    PLATFORMS = {'cpu': True, 'gpu': False, 'tpu': False}
    try:
        import torch
        PLATFORMS['gpu'] = torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import jax
        PLATFORMS['tpu'] = bool(jax.devices('tpu'))
    except:
        pass

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VishwamAI kernel tests")
    parser.add_argument(
        "--platform",
        choices=["all"] + list(PLATFORMS.keys()),
        default="all",
        help="Platform to test on (default: all available platforms)"
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run (default: all)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--failfast",
        action="store_true",
        help="Stop on first failure"
    )
    return parser.parse_args()

def get_available_platforms() -> List[str]:
    """Get list of available platforms."""
    return [p for p, available in PLATFORMS.items() if available]

def validate_platform(platform: str) -> bool:
    """Validate if platform is available."""
    if platform == "all":
        return bool(get_available_platforms())
    return PLATFORMS.get(platform, False)

def get_test_command(args) -> List[str]:
    """Build pytest command based on arguments."""
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add fail-fast
    if args.failfast:
        cmd.append("--exitfirst")
    
    # Add test selection
    if args.platform != "all":
        cmd.append(f"-m {args.platform}")
    
    if args.test_type == "unit":
        cmd.append("not integration")
    elif args.test_type == "integration":
        cmd.append("integration")
    
    # Add benchmark settings
    if args.benchmark:
        cmd.extend([
            "--benchmark-only",
            "--benchmark-autosave",
            f"--benchmark-storage={Path.home()}/.vishwamai/benchmarks"
        ])
    
    # Add test discovery path
    cmd.append("tests/test_kernels.py")
    
    return cmd

def print_environment_info():
    """Print information about the test environment."""
    print("\nEnvironment Information:")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"Available Platforms: {get_available_platforms()}")
    
    if PLATFORMS['gpu']:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    if PLATFORMS['tpu']:
        import jax
        print(f"JAX: {jax.__version__}")
        print(f"TPU Devices: {len(jax.devices('tpu'))}")
    
    print()

def run_tests(args) -> int:
    """Run the test suite."""
    # Validate platform
    if not validate_platform(args.platform):
        platforms = get_available_platforms()
        if not platforms:
            print("Error: No supported platforms available")
            return 1
        if args.platform != "all":
            print(f"Error: Platform '{args.platform}' not available")
            print(f"Available platforms: {platforms}")
            return 1
    
    # Build and run test command
    cmd = get_test_command(args)
    env = os.environ.copy()
    
    # Set environment variables
    if args.platform == "tpu":
        env["JAX_PLATFORM_NAME"] = "tpu"
    elif args.platform == "gpu":
        env["JAX_PLATFORM_NAME"] = "gpu"
        if "CUDA_VISIBLE_DEVICES" not in env:
            env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Run tests
    try:
        print(f"Running tests with command: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(
            cmd,
            env=env,
            check=False,  # Don't raise on test failures
            capture_output=not args.verbose
        )
        
        duration = time.time() - start_time
        
        # Print output if not verbose
        if not args.verbose and result.stdout:
            print(result.stdout.decode())
        if result.stderr:
            print(result.stderr.decode(), file=sys.stderr)
        
        print(f"\nTest execution completed in {duration:.2f} seconds")
        return result.returncode
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}", file=sys.stderr)
        return 1

def main():
    """Main entry point."""
    args = parse_args()
    
    print("VishwamAI Kernel Test Runner")
    print("===========================")
    
    print_environment_info()
    
    # Print test configuration
    print("Test Configuration:")
    print(f"Platform: {args.platform}")
    print(f"Test Type: {args.test_type}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Verbose: {args.verbose}")
    print("\nStarting tests...\n")
    
    try:
        # Create benchmark storage directory
        if args.benchmark:
            os.makedirs(f"{Path.home()}/.vishwamai/benchmarks", exist_ok=True)
        
        # Run tests
        result = run_tests(args)
        
        if result == 0:
            print("\nAll tests passed successfully!")
        else:
            print(f"\nTests failed with exit code: {result}")
        
        return result
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
