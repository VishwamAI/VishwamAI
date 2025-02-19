import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
import os

class TrainingVisualizer:
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
        
    def plot_training_progress(
        self,
        metrics: Dict[str, List[float]],
        title: str = "Training Progress"
    ):
        """Plot training metrics over time."""
        plt.figure(figsize=(12, 6))
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        plt.title(title)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.save_dir}/training_progress_{timestamp}.png")
        plt.close()
        
    def plot_memory_usage(self, memory_stats: List[Dict[str, float]]):
        """Plot GPU memory usage over time."""
        df = pd.DataFrame(memory_stats)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['gpu_memory_used'], label='Used Memory (GB)')
        plt.plot(df['gpu_memory_cached'], label='Cached Memory (GB)')
        plt.axhline(y=38, color='r', linestyle='--', label='Memory Threshold')
        
        plt.title("GPU Memory Usage Over Time")
        plt.xlabel("Steps")
        plt.ylabel("Memory (GB)")
        plt.legend()
        plt.grid(True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.save_dir}/memory_usage_{timestamp}.png")
        plt.close()
        
    def plot_attention_patterns(self, attention_weights: np.ndarray):
        """Plot attention patterns."""
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            attention_weights,
            cmap='viridis',
            xticklabels=False,
            yticklabels=False
        )
        plt.title("Attention Patterns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.save_dir}/attention_patterns_{timestamp}.png")
        plt.close()
        
    def plot_model_architecture(self, model_stats: Dict[str, Any]):
        """Visualize model architecture and parameters."""
        layers = list(range(model_stats['num_layers']))
        params_per_layer = [model_stats['params_per_layer']] * len(layers)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Layer parameters
        ax1.bar(layers, params_per_layer)
        ax1.set_title("Parameters per Layer")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Parameters")
        
        # Memory usage per component
        components = list(model_stats['memory_per_component'].keys())
        memory = list(model_stats['memory_per_component'].values())
        ax2.pie(memory, labels=components, autopct='%1.1f%%')
        ax2.set_title("Memory Usage by Component")
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.save_dir}/model_architecture_{timestamp}.png")
        plt.close()
        
    def plot_training_efficiency(
        self,
        throughput: List[float],
        batch_sizes: List[int],
        learning_rates: List[float]
    ):
        """Plot training efficiency metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Throughput vs Batch Size
        ax1.plot(batch_sizes, throughput)
        ax1.set_title("Training Throughput vs Batch Size")
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Samples/Second")
        
        # Learning Rate Schedule
        ax2.plot(learning_rates)
        ax2.set_title("Learning Rate Schedule")
        ax2.set_xlabel("Steps")
        ax2.set_ylabel("Learning Rate")
        ax2.set_yscale('log')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.save_dir}/training_efficiency_{timestamp}.png")
        plt.close()

    def create_training_report(
        self,
        metrics: Dict[str, Any],
        memory_stats: List[Dict[str, float]],
        model_stats: Dict[str, Any]
    ):
        """Create comprehensive training report with visualizations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = f"{self.save_dir}/report_{timestamp}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate all plots
        self.plot_training_progress(metrics)
        self.plot_memory_usage(memory_stats)
        self.plot_model_architecture(model_stats)
        
        # Create HTML report
        with open(f"{report_dir}/report.html", "w") as f:
            f.write("<html><body>")
            f.write("<h1>Training Report</h1>")
            
            # Add metrics summary
            f.write("<h2>Training Metrics</h2>")
            f.write("<table border='1'>")
            for metric, value in metrics.items():
                f.write(f"<tr><td>{metric}</td><td>{value}</td></tr>")
            f.write("</table>")
            
            # Add plots
            f.write("<h2>Visualizations</h2>")
            for img in os.listdir(report_dir):
                if img.endswith(".png"):
                    f.write(f"<img src='{img}'><br>")
                    
            f.write("</body></html>")
            
        return report_dir

    def plot_evaluation_results(
        self,
        metrics: Dict[str, Dict[str, float]],
        save_path: str
    ):
        """Plot detailed evaluation results."""
        num_datasets = len(metrics)
        num_metrics = len(next(iter(metrics.values())))
        
        # Create subplot grid
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 3, figure=fig)
        
        # Accuracy comparison
        ax1 = fig.add_subplot(gs[0, :2])
        datasets = list(metrics.keys())
        accuracies = [m['accuracy'] for m in metrics.values()]
        ax1.bar(datasets, accuracies)
        ax1.set_title('Accuracy by Dataset')
        ax1.set_xticklabels(datasets, rotation=45)
        ax1.set_ylabel('Accuracy')
        
        # Perplexity comparison
        ax2 = fig.add_subplot(gs[0, 2])
        perplexities = [m['perplexity'] for m in metrics.values()]
        ax2.boxplot(perplexities)
        ax2.set_title('Perplexity Distribution')
        ax2.set_ylabel('Perplexity')
        
        # Reasoning score scatter
        ax3 = fig.add_subplot(gs[1, 0])
        reasoning_scores = [m['reasoning_score'] for m in metrics.values()]
        ax3.scatter(accuracies, reasoning_scores)
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('Reasoning Score')
        ax3.set_title('Accuracy vs Reasoning')
        
        # Calibration error heatmap
        ax4 = fig.add_subplot(gs[1, 1:])
        calibration_errors = [m['calibration_error'] for m in metrics.values()]
        calibration_matrix = np.array([accuracies, calibration_errors]).T
        sns.heatmap(
            calibration_matrix,
            xticklabels=['Accuracy', 'Calibration Error'],
            yticklabels=datasets,
            ax=ax4,
            cmap='RdYlBu_r',
            annot=True
        )
        ax4.set_title('Accuracy and Calibration Error Heatmap')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
