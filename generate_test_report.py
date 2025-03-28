#!/usr/bin/env python3
"""Script to generate test reports from VishwamAI test results."""

import os
import sys
import json
import datetime
import argparse
from typing import Dict, Any, Optional
import subprocess
import uuid

import numpy as np
import jax
import torch

def get_platform_status() -> Dict[str, str]:
    """Get status of each platform."""
    status = {
        "TPU": "Not Available",
        "GPU": "Not Available",
        "CPU": "Available"
    }
    
    try:
        if jax.devices('tpu'):
            status["TPU"] = "Available"
    except:
        pass
        
    try:
        if torch.cuda.is_available():
            status["GPU"] = "Available"
    except:
        pass
    
    return status

def get_software_versions() -> Dict[str, str]:
    """Get versions of key software packages."""
    versions = {
        "PYTHON_VERSION": ".".join(map(str, sys.version_info[:3])),
        "NUMPY_VERSION": np.__version__,
        "JAX_VERSION": "Not Installed",
        "TORCH_VERSION": "Not Installed"
    }
    
    try:
        versions["JAX_VERSION"] = jax.__version__
    except:
        pass
    
    try:
        versions["TORCH_VERSION"] = torch.__version__
    except:
        pass
    
    return versions

def read_benchmark_results(benchmark_file: str) -> Dict[str, Any]:
    """Read benchmark results from JSON file."""
    try:
        with open(benchmark_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Benchmark file {benchmark_file} not found")
        return {}

def get_test_coverage() -> Dict[str, int]:
    """Get test coverage statistics."""
    coverage = {
        "TOTAL_LINES": 0,
        "COVERED_LINES": 0,
        "COVERAGE_PERCENT": 0,
        "TOTAL_TESTS": 0,
        "PASSED_TESTS": 0,
        "FAILED_TESTS": 0,
        "SKIPPED_TESTS": 0
    }
    
    try:
        # Run pytest with coverage
        result = subprocess.run(
            ["pytest", "--cov=vishwamai.kernels", "--cov-report=term-missing"],
            capture_output=True,
            text=True
        )
        
        # Parse coverage output
        for line in result.stdout.split('\n'):
            if "TOTAL" in line:
                parts = line.split()
                coverage["TOTAL_LINES"] = int(parts[1])
                coverage["COVERED_LINES"] = int(parts[2])
                coverage["COVERAGE_PERCENT"] = int(float(parts[3].strip('%')))
            
            if "=" in line and "test session" in line:
                parts = line.split()
                for part in parts:
                    if "passed" in part:
                        coverage["PASSED_TESTS"] = int(part.split("passed")[0])
                    elif "failed" in part:
                        coverage["FAILED_TESTS"] = int(part.split("failed")[0])
                    elif "skipped" in part:
                        coverage["SKIPPED_TESTS"] = int(part.split("skipped")[0])
        
        coverage["TOTAL_TESTS"] = (
            coverage["PASSED_TESTS"] +
            coverage["FAILED_TESTS"] +
            coverage["SKIPPED_TESTS"]
        )
    except Exception as e:
        print(f"Error getting coverage: {e}")
    
    return coverage

def get_performance_metrics(benchmark_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Extract performance metrics from benchmark results."""
    metrics = {
        "TPU": {},
        "GPU": {},
        "CPU": {}
    }
    
    operations = [
        "matrix_multiplication",
        "flash_attention",
        "layer_normalization",
        "sparse_operations"
    ]
    
    for platform in metrics:
        for op in operations:
            metrics[platform][op] = {
                "time_ms": 0.0,
                "memory_gb": 0.0,
                "throughput": 0.0
            }
    
    # Parse benchmark results
    if benchmark_results:
        for benchmark in benchmark_results.get("benchmarks", []):
            name = benchmark["name"]
            platform = None
            operation = None
            
            # Determine platform and operation from benchmark name
            if "tpu" in name.lower():
                platform = "TPU"
            elif "gpu" in name.lower():
                platform = "GPU"
            else:
                platform = "CPU"
                
            for op in operations:
                if op.replace("_", "") in name.lower():
                    operation = op
                    break
            
            if platform and operation:
                metrics[platform][operation]["time_ms"] = benchmark["stats"]["mean"] * 1000
                metrics[platform][operation]["throughput"] = 1 / benchmark["stats"]["mean"]
                
                # Memory is tracked separately in benchmark extras
                if "mem_usage" in benchmark.get("extras", {}):
                    metrics[platform][operation]["memory_gb"] = benchmark["extras"]["mem_usage"] / 1024
    
    return metrics

def calculate_cost_metrics(test_duration: float) -> Dict[str, Dict[str, float]]:
    """Calculate cost-related metrics."""
    costs = {
        "TPU": {"cost_per_hour": 4.50},
        "GPU": {"cost_per_hour": 2.48},
        "CPU": {"cost_per_hour": 0.76}
    }
    
    for platform in costs:
        platform_cost = costs[platform]["cost_per_hour"]
        costs[platform].update({
            "total_cost": platform_cost * test_duration,
            "flops_per_dollar": 0.0,  # Would need actual FLOPS measurements
            "performance_per_dollar": 0.0  # Would need performance metrics
        })
    
    return costs

def generate_report(
    benchmark_file: Optional[str] = None,
    test_duration: float = 1.0,
    output_file: Optional[str] = None
) -> None:
    """Generate test report from results."""
    # Read report template
    with open("docs/test_report_template.md", 'r') as f:
        template = f.read()
    
    # Get test environment info
    platform_status = get_platform_status()
    versions = get_software_versions()
    coverage = get_test_coverage()
    
    # Read benchmark results if available
    benchmark_results = {}
    if benchmark_file:
        benchmark_results = read_benchmark_results(benchmark_file)
    
    # Get performance metrics
    metrics = get_performance_metrics(benchmark_results)
    
    # Calculate costs
    costs = calculate_cost_metrics(test_duration)
    
    # Prepare template variables
    variables = {
        "DATE": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "RUN_ID": str(uuid.uuid4()),
        "TPU_STATUS": platform_status["TPU"],
        "GPU_STATUS": platform_status["GPU"],
        "CPU_STATUS": platform_status["CPU"],
        "JAX_VERSION": versions["JAX_VERSION"],
        "TORCH_VERSION": versions["TORCH_VERSION"],
        "NUMPY_VERSION": versions["NUMPY_VERSION"],
        "PYTHON_VERSION": versions["PYTHON_VERSION"],
        "TOTAL_LINES": coverage["TOTAL_LINES"],
        "COVERED_LINES": coverage["COVERED_LINES"],
        "COVERAGE_PERCENT": coverage["COVERAGE_PERCENT"],
        "TOTAL_TESTS": coverage["TOTAL_TESTS"],
        "PASSED_TESTS": coverage["PASSED_TESTS"],
        "FAILED_TESTS": coverage["FAILED_TESTS"],
        "SKIPPED_TESTS": coverage["SKIPPED_TESTS"],
        "TEST_DURATION": test_duration,
        "TPU_COST": f"{costs['TPU']['total_cost']:.2f}",
        "GPU_COST": f"{costs['GPU']['total_cost']:.2f}",
        "CPU_COST": f"{costs['CPU']['total_cost']:.2f}",
        "TOTAL_COST": f"{sum(p['total_cost'] for p in costs.values()):.2f}",
        "TESTER_NAME": os.environ.get("USER", "Unknown"),
        "GENERATION_DATE": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "REPORT_VERSION": "1.0.0"
    }
    
    # Fill in performance metrics
    for platform in metrics:
        for op, values in metrics[platform].items():
            template = template.replace(
                f"{platform}_{op}_time",
                f"{values['time_ms']:.2f}"
            )
            template = template.replace(
                f"{platform}_{op}_memory",
                f"{values['memory_gb']:.2f}"
            )
            template = template.replace(
                f"{platform}_{op}_throughput",
                f"{values['throughput']:.2f}"
            )
    
    # Fill in template variables
    for key, value in variables.items():
        template = template.replace(f"{{{key}}}", str(value))
    
    # Write report
    if output_file:
        with open(output_file, 'w') as f:
            f.write(template)
    else:
        print(template)

def main():
    parser = argparse.ArgumentParser(description="Generate VishwamAI test report")
    parser.add_argument(
        "--benchmark-file",
        help="JSON file containing benchmark results"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Test duration in hours"
    )
    parser.add_argument(
        "--output",
        help="Output file for report (default: print to stdout)"
    )
    
    args = parser.parse_args()
    generate_report(args.benchmark_file, args.duration, args.output)

if __name__ == "__main__":
    main()
