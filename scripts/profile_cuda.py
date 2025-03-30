#!/usr/bin/env python3
"""
CUDA Profiling and Analysis Tool for Gemma 3 Knowledge Distillation Kernels
"""

import sys
import os
import subprocess
import argparse
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='CUDA Kernel Profiling Tool')
    parser.add_argument('--binary', type=str, required=True,
                       help='Path to test binary')
    parser.add_argument('--output-dir', type=str, default='profile_results',
                       help='Directory for output files')
    parser.add_argument('--metrics', type=str, default='all',
                       choices=['all', 'memory', 'compute', 'throughput'],
                       help='Metrics to collect')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations for each test')
    parser.add_argument('--nvprof', action='store_true',
                       help='Use nvprof instead of NSight')
    parser.add_argument('--export-format', type=str, default='json',
                       choices=['json', 'csv'], help='Export format')
    return parser.parse_args()

class CUDAProfiler:
    """CUDA profiling and analysis tools."""
    
    def __init__(self, binary_path: str, output_dir: str):
        self.binary_path = Path(binary_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability
        self._check_gpu()
        
        # Initialize results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'binary': str(binary_path),
                'gpu_info': self._get_gpu_info()
            },
            'metrics': {},
            'kernel_stats': {},
            'memory_stats': {},
            'errors': []
        }

    def _check_gpu(self):
        """Check GPU availability and CUDA version."""
        try:
            nvidia_smi = subprocess.run(['nvidia-smi'], 
                                     capture_output=True, text=True)
            if nvidia_smi.returncode != 0:
                raise RuntimeError("No CUDA GPU available")
            
            nvcc = subprocess.run(['nvcc', '--version'],
                                capture_output=True, text=True)
            if nvcc.returncode != 0:
                raise RuntimeError("CUDA toolkit not found")
        except FileNotFoundError:
            raise RuntimeError("Required CUDA tools not found")

    def _get_gpu_info(self) -> Dict[str, str]:
        """Get GPU device information."""
        cmd = ['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
               '--format=csv,noheader,nounits']
        result = subprocess.run(cmd, capture_output=True, text=True)
        name, driver, memory = result.stdout.strip().split(',')
        return {
            'name': name.strip(),
            'driver_version': driver.strip(),
            'memory_total': memory.strip()
        }

    def run_kernel_analysis(self, iterations: int = 10) -> Dict:
        """Run kernel analysis with multiple iterations."""
        metrics = []
        for i in range(iterations):
            env = os.environ.copy()
            env['CUDA_LAUNCH_BLOCKING'] = '1'
            
            result = subprocess.run([str(self.binary_path), '--analyze-kernels'],
                                 env=env, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.results['errors'].append(f"Iteration {i} failed")
                continue
            
            try:
                metrics.append(json.loads(result.stdout))
            except json.JSONDecodeError:
                self.results['errors'].append(f"Invalid JSON in iteration {i}")
        
        return self._aggregate_metrics(metrics)

    def run_memory_analysis(self) -> Dict:
        """Analyze memory usage patterns."""
        env = os.environ.copy()
        env['CUDA_LAUNCH_BLOCKING'] = '1'
        
        cmd = ['cuda-memcheck', '--leak-check', 'full',
               '--track-memory', 'yes', str(self.binary_path)]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return self._parse_memcheck_output(result.stdout)

    def run_profiler(self, use_nvprof: bool = False) -> Dict:
        """Run CUDA profiler (nvprof or NSight)."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f'profile_{timestamp}.{"csv" if use_nvprof else "nsight-cuprof"}'
        
        if use_nvprof:
            cmd = ['nvprof', '--csv', '--log-file', str(output_file),
                  str(self.binary_path)]
        else:
            cmd = ['ncu', '--csv', '--target-processes', 'all',
                  '--output', str(output_file), str(self.binary_path)]
        
        subprocess.run(cmd)
        return self._parse_profile_data(output_file, use_nvprof)

    def _aggregate_metrics(self, metrics: List[Dict]) -> Dict:
        """Aggregate metrics from multiple iterations."""
        if not metrics:
            return {}
        
        aggregated = {
            'mean': {},
            'min': {},
            'max': {},
            'std': {}
        }
        
        # Calculate statistics for each metric
        for key in metrics[0].keys():
            values = [m[key] for m in metrics]
            aggregated['mean'][key] = sum(values) / len(values)
            aggregated['min'][key] = min(values)
            aggregated['max'][key] = max(values)
            
            # Standard deviation
            mean = aggregated['mean'][key]
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            aggregated['std'][key] = variance ** 0.5
        
        return aggregated

    def _parse_memcheck_output(self, output: str) -> Dict:
        """Parse cuda-memcheck output."""
        results = {
            'leaks': [],
            'errors': [],
            'stats': {
                'total_allocations': 0,
                'total_leaks': 0,
                'total_errors': 0
            }
        }
        
        for line in output.splitlines():
            if 'Memory leak' in line:
                results['leaks'].append(line)
                results['stats']['total_leaks'] += 1
            elif 'ERROR SUMMARY' in line:
                results['stats']['total_errors'] = int(line.split(':')[1])
            elif 'HEAP SUMMARY' in line:
                # Parse heap summary information
                continue
        
        return results

    def _parse_profile_data(self, file_path: Path, is_nvprof: bool) -> Dict:
        """Parse profiler output."""
        results = {
            'kernels': [],
            'memory_ops': [],
            'summary': {}
        }
        
        with open(file_path) as f:
            if is_nvprof:
                reader = csv.DictReader(f)
            else:
                # NSight data format
                reader = csv.DictReader(f, delimiter=',')
            
            for row in reader:
                if 'kernel' in row.get('Type', '').lower():
                    results['kernels'].append(row)
                elif 'memory' in row.get('Type', '').lower():
                    results['memory_ops'].append(row)
        
        return results

    def export_results(self, format: str = 'json'):
        """Export profiling results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if format == 'json':
            output_file = self.output_dir / f'results_{timestamp}.json'
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        else:
            output_file = self.output_dir / f'results_{timestamp}.csv'
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write metadata
                writer.writerow(['Metadata'])
                for key, value in self.results['metadata'].items():
                    writer.writerow([key, value])
                
                # Write metrics
                writer.writerow([])
                writer.writerow(['Metrics'])
                for key, value in self.results['metrics'].items():
                    writer.writerow([key, value])

def main():
    args = parse_args()
    
    try:
        profiler = CUDAProfiler(args.binary, args.output_dir)
        
        # Run analysis based on selected metrics
        if args.metrics in ['all', 'compute']:
            profiler.results['metrics'] = profiler.run_kernel_analysis(args.iterations)
        
        if args.metrics in ['all', 'memory']:
            profiler.results['memory_stats'] = profiler.run_memory_analysis()
        
        if args.metrics in ['all', 'throughput']:
            profiler.results['kernel_stats'] = profiler.run_profiler(args.nvprof)
        
        # Export results
        profiler.export_results(args.export_format)
        
        print(f"Profiling completed. Results saved in {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
