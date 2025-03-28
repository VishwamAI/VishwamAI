#!/usr/bin/env python3
"""Test runner for VishwamAI kernel tests."""

import os
import sys
import argparse
import subprocess
from typing import List, Optional

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run VishwamAI kernel tests")
    parser.add_argument(
        "--platform",
        choices=["all", "tpu", "gpu", "cpu"],
        default="all",
        help="Platform to test on"
    )
    parser.add_argument(
        "--test-type",
        choices=["unit", "integration", "all"],
        default="all",
        help="Type of tests to run"
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
    return parser.parse_args()

def get_test_command(args) -> List[str]:
    """Build pytest command based on arguments."""
    cmd = ["pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Add test selection
    if args.platform != "all":
        cmd.append(f"-m {args.platform}")
    
    if args.test_type == "unit":
        cmd.append("not integration")
    elif args.test_type == "integration":
        cmd.append("integration")
    
    # Add benchmark settings
    if args.benchmark:
        cmd.append("--benchmark-only")
        cmd.append("--benchmark-autosave")
    
    # Add test discovery path
    cmd.append("tests/test_kernels.py")
    
    return cmd

def verify_environment() -> Optional[str]:
    """Verify test environment is properly configured."""
    missing = []
    
    try:
        import jax
    except ImportError:
        missing.append("JAX")
    
    try:
        import torch
    except ImportError:
        missing.append("PyTorch")
    
    try:
        import pytest
    except ImportError:
        missing.append("pytest")
    
    if missing:
        return f"Missing required packages: {', '.join(missing)}"
    
    return None

def check_device_availability(platform: str) -> bool:
    """Check if requested platform is available."""
    if platform == "gpu":
        import torch
        return torch.cuda.is_available()
    elif platform == "tpu":
        import jax
        return bool(jax.devices('tpu'))
    return True  # CPU is always available

def run_tests(args) -> int:
    """Run the test suite."""
    # Verify environment
    error = verify_environment()
    if error:
        print(f"Environment Error: {error}")
        return 1
    
    # Check platform availability
    if args.platform != "all" and not check_device_availability(args.platform):
        print(f"Error: {args.platform.upper()} is not available")
        return 1
    
    # Build and run test command
    cmd = get_test_command(args)
    env = os.environ.copy()
    
    # Set environment variables
    if args.platform == "tpu":
        env["JAX_PLATFORM_NAME"] = "tpu"
    elif args.platform == "gpu":
        env["JAX_PLATFORM_NAME"] = "gpu"
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU by default
    
    # Run tests
    try:
        print(f"Running tests with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=args.verbose
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Test execution failed: {e}")
        if args.verbose and e.output:
            print(e.output.decode())
        return e.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def main():
    """Main entry point."""
    args = parse_args()
    
    print("VishwamAI Kernel Test Runner")
    print("===========================")
    
    # Print test configuration
    print("\nTest Configuration:")
    print(f"Platform: {args.platform}")
    print(f"Test Type: {args.test_type}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Verbose: {args.verbose}")
    print("\nStarting tests...\n")
    
    # Run tests
    result = run_tests(args)
    
    if result == 0:
        print("\nAll tests passed successfully!")
    else:
        print(f"\nTests failed with exit code: {result}")
    
    return result

if __name__ == "__main__":
    sys.exit(main())
