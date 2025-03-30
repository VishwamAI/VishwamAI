#!/usr/bin/env python3
"""
Visualization tool for CUDA profiling results from knowledge distillation kernels.
"""

import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def parse_args():
    parser = argparse.ArgumentParser(description='CUDA Profile Visualization Tool')
    parser.add_argument('--input', type=str, required=True,
                       help='Input profile results file (JSON or CSV)')
    parser.add_argument('--output-dir', type=str, default='profile_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--format', type=str, default='pdf',
                       choices=['pdf', 'png', 'svg'],
                       help='Output format for plots')
    return parser.parse_args()

class ProfileVisualizer:
    """Generate visualizations from profiling data."""
    
    def __init__(self, input_path: str, output_dir: str):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.data = self._load_data()
        
        # Set style
        plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
    
    def _load_data(self) -> Dict:
        """Load profiling data from file."""
        if self.input_path.suffix == '.json':
            with open(self.input_path) as f:
                return json.load(f)
        elif self.input_path.suffix == '.csv':
            data = {'metrics': {}, 'kernel_stats': [], 'memory_stats': {}}
            with open(self.input_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'kernel' in row:
                        data['kernel_stats'].append(row)
                    elif 'metric' in row:
                        data['metrics'][row['metric']] = float(row['value'])
            return data
        else:
            raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
    
    def plot_kernel_performance(self, output_format: str = 'pdf'):
        """Plot kernel performance metrics."""
        kernels = self.data.get('kernel_stats', {}).get('kernels', [])
        if not kernels:
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Execution time by kernel
        ax1 = fig.add_subplot(gs[0, 0])
        times = [float(k['Duration']) for k in kernels]
        names = [k['Name'] for k in kernels]
        sns.barplot(x=times, y=names, ax=ax1)
        ax1.set_title('Kernel Execution Times')
        ax1.set_xlabel('Duration (ms)')
        
        # Occupancy
        ax2 = fig.add_subplot(gs[0, 1])
        occupancy = [float(k.get('Occupancy', 0)) for k in kernels]
        sns.barplot(x=occupancy, y=names, ax=ax2)
        ax2.set_title('Kernel Occupancy')
        ax2.set_xlabel('Occupancy (%)')
        
        # Memory throughput
        ax3 = fig.add_subplot(gs[1, 0])
        throughput = [float(k.get('Memory Throughput', 0)) for k in kernels]
        sns.barplot(x=throughput, y=names, ax=ax3)
        ax3.set_title('Memory Throughput')
        ax3.set_xlabel('GB/s')
        
        # Register usage
        ax4 = fig.add_subplot(gs[1, 1])
        registers = [int(k.get('Registers Per Thread', 0)) for k in kernels]
        sns.barplot(x=registers, y=names, ax=ax4)
        ax4.set_title('Register Usage')
        ax4.set_xlabel('Registers/Thread')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'kernel_performance.{output_format}')
        plt.close()
    
    def plot_memory_analysis(self, output_format: str = 'pdf'):
        """Plot memory usage and patterns."""
        memory_stats = self.data.get('memory_stats', {})
        if not memory_stats:
            return
        
        fig = plt.figure(figsize=(15, 5))
        gs = GridSpec(1, 2)
        
        # Memory allocation pattern
        ax1 = fig.add_subplot(gs[0, 0])
        allocations = memory_stats.get('stats', {})
        sizes = [allocations.get('total_allocated', 0),
                allocations.get('total_freed', 0),
                allocations.get('peak_usage', 0)]
        labels = ['Total Allocated', 'Total Freed', 'Peak Usage']
        sns.barplot(x=labels, y=sizes, ax=ax1)
        ax1.set_title('Memory Usage Statistics')
        ax1.set_ylabel('Bytes')
        
        # Memory fragmentation
        ax2 = fig.add_subplot(gs[0, 1])
        blocks = memory_stats.get('block_sizes', [])
        if blocks:
            sns.histplot(data=blocks, ax=ax2)
            ax2.set_title('Memory Block Size Distribution')
            ax2.set_xlabel('Block Size (bytes)')
            ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'memory_analysis.{output_format}')
        plt.close()
    
    def plot_metrics_summary(self, output_format: str = 'pdf'):
        """Plot summary of performance metrics."""
        metrics = self.data.get('metrics', {}).get('mean', {})
        if not metrics:
            return
        
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # complete the circle
        angles = np.concatenate((angles, [angles[0]]))  # complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Performance Metrics Summary')
        
        plt.savefig(self.output_dir / f'metrics_summary.{output_format}')
        plt.close()
    
    def generate_report(self):
        """Generate HTML report with visualizations and analysis."""
        report_path = self.output_dir / 'profile_report.html'
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CUDA Profile Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 30px; }
                .metric { margin: 10px 0; }
                img { max-width: 100%; }
            </style>
        </head>
        <body>
            <h1>CUDA Profile Analysis Report</h1>
        """
        
        # Add metadata
        metadata = self.data.get('metadata', {})
        html_content += """
            <div class="section">
                <h2>Metadata</h2>
                <div class="metric">Timestamp: {}</div>
                <div class="metric">GPU: {}</div>
                <div class="metric">Driver Version: {}</div>
            </div>
        """.format(
            metadata.get('timestamp', 'N/A'),
            metadata.get('gpu_info', {}).get('name', 'N/A'),
            metadata.get('gpu_info', {}).get('driver_version', 'N/A')
        )
        
        # Add plots
        html_content += """
            <div class="section">
                <h2>Performance Analysis</h2>
                <img src="kernel_performance.png" alt="Kernel Performance">
                <img src="memory_analysis.png" alt="Memory Analysis">
                <img src="metrics_summary.png" alt="Metrics Summary">
            </div>
        """
        
        # Add summary statistics
        metrics = self.data.get('metrics', {}).get('mean', {})
        if metrics:
            html_content += """
                <div class="section">
                    <h2>Performance Metrics</h2>
            """
            for metric, value in metrics.items():
                html_content += f'<div class="metric">{metric}: {value:.2f}</div>'
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)

def main():
    args = parse_args()
    
    try:
        visualizer = ProfileVisualizer(args.input, args.output_dir)
        
        # Generate visualizations
        visualizer.plot_kernel_performance(args.format)
        visualizer.plot_memory_analysis(args.format)
        visualizer.plot_metrics_summary(args.format)
        
        # Generate HTML report
        visualizer.generate_report()
        
        print(f"Visualizations and report generated in {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
