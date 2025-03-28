#!/usr/bin/env python3
"""Script to visualize VishwamAI test results and performance metrics."""

import os
import json
import argparse
from typing import Dict, Any, List, Tuple
import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def load_test_results(results_file: str) -> Dict[str, Any]:
    """Load test results from JSON file."""
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Results file {results_file} not found")
        return {}

def create_performance_plot(
    metrics: Dict[str, Dict[str, float]],
    output_dir: str
) -> None:
    """Create performance comparison plots."""
    platforms = ["TPU", "GPU", "CPU"]
    operations = [
        "matrix_multiplication",
        "flash_attention",
        "layer_normalization",
        "sparse_operations"
    ]
    
    # Set style
    plt.style.use('seaborn')
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Time plot
    ax1 = fig.add_subplot(gs[0, 0])
    data = [[metrics[p][op]["time_ms"] for p in platforms] for op in operations]
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        xticklabels=platforms,
        yticklabels=operations,
        cmap="YlOrRd",
        ax=ax1
    )
    ax1.set_title("Operation Time (ms)")
    
    # Memory plot
    ax2 = fig.add_subplot(gs[0, 1])
    data = [[metrics[p][op]["memory_gb"] for p in platforms] for op in operations]
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        xticklabels=platforms,
        yticklabels=operations,
        cmap="YlOrRd",
        ax=ax2
    )
    ax2.set_title("Memory Usage (GB)")
    
    # Throughput plot
    ax3 = fig.add_subplot(gs[1, 0])
    data = [[metrics[p][op]["throughput"] for p in platforms] for op in operations]
    sns.heatmap(
        data,
        annot=True,
        fmt=".2f",
        xticklabels=platforms,
        yticklabels=operations,
        cmap="YlOrRd",
        ax=ax3
    )
    ax3.set_title("Throughput (ops/s)")
    
    # Platform comparison
    ax4 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(operations))
    width = 0.25
    
    for i, platform in enumerate(platforms):
        times = [metrics[platform][op]["time_ms"] for op in operations]
        ax4.bar(x + i*width, times, width, label=platform)
    
    ax4.set_ylabel('Time (ms)')
    ax4.set_title('Operation Time by Platform')
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(operations, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'))
    plt.close()

def create_cost_analysis_plot(
    costs: Dict[str, Dict[str, float]],
    output_dir: str
) -> None:
    """Create cost analysis plots."""
    platforms = list(costs.keys())
    metrics = ["cost_per_hour", "total_cost", "flops_per_dollar"]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Cost comparison
    costs_per_hour = [costs[p]["cost_per_hour"] for p in platforms]
    total_costs = [costs[p]["total_cost"] for p in platforms]
    
    x = np.arange(len(platforms))
    width = 0.35
    
    ax1.bar(x - width/2, costs_per_hour, width, label='Cost per Hour')
    ax1.bar(x + width/2, total_costs, width, label='Total Cost')
    
    ax1.set_ylabel('Cost ($)')
    ax1.set_title('Cost Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms)
    ax1.legend()
    
    # Cost efficiency
    efficiency = [costs[p]["flops_per_dollar"] for p in platforms]
    ax2.bar(platforms, efficiency)
    ax2.set_ylabel('FLOPS/$')
    ax2.set_title('Cost Efficiency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_analysis.png'))
    plt.close()

def create_test_coverage_plot(
    coverage: Dict[str, int],
    output_dir: str
) -> None:
    """Create test coverage visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Coverage percentage
    covered = coverage["COVERED_LINES"]
    total = coverage["TOTAL_LINES"]
    uncovered = total - covered
    
    ax1.pie(
        [covered, uncovered],
        labels=['Covered', 'Uncovered'],
        autopct='%1.1f%%',
        colors=['#2ecc71', '#e74c3c']
    )
    ax1.set_title('Code Coverage')
    
    # Test results
    results = ['Passed', 'Failed', 'Skipped']
    values = [
        coverage["PASSED_TESTS"],
        coverage["FAILED_TESTS"],
        coverage["SKIPPED_TESTS"]
    ]
    
    ax2.bar(results, values, color=['#2ecc71', '#e74c3c', '#f1c40f'])
    ax2.set_title('Test Results')
    ax2.set_ylabel('Number of Tests')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_coverage.png'))
    plt.close()

def create_history_plot(
    history_file: str,
    output_dir: str
) -> None:
    """Create historical performance trend plot."""
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"History file {history_file} not found")
        return
    
    dates = [h['date'] for h in history]
    tpu_times = [h['tpu_time'] for h in history]
    gpu_times = [h['gpu_time'] for h in history]
    cpu_times = [h['cpu_time'] for h in history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, tpu_times, label='TPU', marker='o')
    plt.plot(dates, gpu_times, label='GPU', marker='s')
    plt.plot(dates, cpu_times, label='CPU', marker='^')
    
    plt.xlabel('Date')
    plt.ylabel('Time (ms)')
    plt.title('Performance Trends')
    plt.xticks(rotation=45)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_history.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize VishwamAI test results")
    parser.add_argument(
        "--results",
        required=True,
        help="JSON file containing test results"
    )
    parser.add_argument(
        "--history",
        help="JSON file containing historical results"
    )
    parser.add_argument(
        "--output-dir",
        default="visualizations",
        help="Directory for output plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_test_results(args.results)
    
    if results:
        # Create visualizations
        create_performance_plot(results.get("metrics", {}), args.output_dir)
        create_cost_analysis_plot(results.get("costs", {}), args.output_dir)
        create_test_coverage_plot(results.get("coverage", {}), args.output_dir)
        
        if args.history:
            create_history_plot(args.history, args.output_dir)
        
        print(f"Visualizations created in {args.output_dir}")
    else:
        print("No results to visualize")

if __name__ == "__main__":
    main()
