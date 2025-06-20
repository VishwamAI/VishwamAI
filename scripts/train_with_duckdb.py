#!/usr/bin/env python3
"""
VishwamAI Training with DuckDB Integration

A comprehensive training script that replaces Weights & Biases with DuckDB
for experiment tracking, metrics logging, and data analysis.

Usage Examples:
    # Basic distillation training with DuckDB
    python train_with_duckdb.py --config configs/distillation_config.json
    
    # Training with custom experiment name
    python train_with_duckdb.py --experiment-name "multimodal_v1" --preset multimodal
    
    # Generate training report from database
    python train_with_duckdb.py --generate-report --db-path experiments.duckdb
    
    # Progressive training with synthetic data
    python train_with_duckdb.py --progressive --use-synthetic-data --dataset wikitext-2
"""

import argparse
import logging
import json
import torch
import duckdb
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns

from vishwamai.distillation import (
    DistillationConfig,
    train_with_distillation,
    evaluate_distilled_model,
    create_distillation_dataset,
    DistillationDataset
)
from vishwamai.huggingface_integration import (
    VishwamAIForCausalLM,
    VishwamAIConfig,
    VishwamAITokenizer,
    save_vishwamai_model
)
from vishwamai import get_hardware_info


class VishwamAIDuckDBTracker:
    """Advanced DuckDB tracker for VishwamAI experiments."""
    
    def __init__(self, db_path: str = "vishwamai_experiments.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._setup_schema()
        
    def _setup_schema(self):
        """Setup comprehensive database schema for VishwamAI experiments."""
        
        # Main experiments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_name TEXT,
                model_type TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                config JSON,
                hardware_info JSON,
                dataset_info JSON,
                final_metrics JSON,
                notes TEXT
            )
        """)
        
        # Training metrics with detailed tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_metrics (
                experiment_id TEXT,
                step INTEGER,
                epoch INTEGER,
                metric_name TEXT,
                metric_value DOUBLE,
                metric_type TEXT,
                phase TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Knowledge distillation specific metrics
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS distillation_metrics (
                experiment_id TEXT,
                step INTEGER,
                epoch INTEGER,
                teacher_model TEXT,
                distillation_loss DOUBLE,
                student_loss DOUBLE,
                combined_loss DOUBLE,
                temperature DOUBLE,
                alpha DOUBLE,
                kl_divergence DOUBLE,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Model architecture and performance tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                experiment_id TEXT,
                checkpoint_name TEXT,
                model_size_mb DOUBLE,
                parameters_count BIGINT,
                inference_time_ms DOUBLE,
                memory_usage_mb DOUBLE,
                perplexity DOUBLE,
                bleu_score DOUBLE,
                rouge_scores JSON,
                custom_metrics JSON,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Progressive training stages
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS progressive_stages (
                experiment_id TEXT,
                stage_name TEXT,
                stage_order INTEGER,
                max_seq_length INTEGER,
                learning_rate DOUBLE,
                batch_size INTEGER,
                start_step INTEGER,
                end_step INTEGER,
                stage_loss DOUBLE,
                stage_perplexity DOUBLE,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Synthetic data generation tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS synthetic_data_tracking (
                experiment_id TEXT,
                generation_method TEXT,
                prompt_template TEXT,
                generated_samples INTEGER,
                quality_threshold DOUBLE,
                avg_quality_score DOUBLE,
                generation_time_seconds DOUBLE,
                model_used TEXT,
                parameters JSON,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Hyperparameter optimization tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hyperparameter_trials (
                experiment_id TEXT,
                trial_id TEXT,
                hyperparameters JSON,
                objective_value DOUBLE,
                status TEXT,
                trial_duration_seconds DOUBLE,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        logging.info(f"DuckDB schema initialized at {self.db_path}")
    
    def start_experiment(self, experiment_name: str, config: Dict, hardware_info: Dict, dataset_info: Dict = None) -> str:
        """Start a new experiment and return the experiment ID."""
        experiment_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.conn.execute("""
            INSERT INTO experiments 
            (experiment_id, experiment_name, model_type, start_time, status, config, hardware_info, dataset_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            experiment_id,
            experiment_name,
            config.get('model_type', 'VishwamAI'),
            datetime.now(),
            'running',
            json.dumps(config),
            json.dumps(hardware_info),
            json.dumps(dataset_info or {})
        ])
        
        logging.info(f"Started experiment: {experiment_id}")
        return experiment_id
    
    def log_training_metrics(self, experiment_id: str, metrics: Dict, step: int, epoch: int, phase: str = "train"):
        """Log training metrics."""
        timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.conn.execute("""
                    INSERT INTO training_metrics 
                    (experiment_id, step, epoch, metric_name, metric_value, metric_type, phase, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, [experiment_id, step, epoch, metric_name, float(value), "scalar", phase, timestamp])
    
    def log_distillation_metrics(self, experiment_id: str, metrics: Dict, step: int, epoch: int):
        """Log distillation-specific metrics."""
        self.conn.execute("""
            INSERT INTO distillation_metrics 
            (experiment_id, step, epoch, teacher_model, distillation_loss, student_loss, 
             combined_loss, temperature, alpha, kl_divergence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            experiment_id, step, epoch,
            metrics.get('teacher_model', 'unknown'),
            metrics.get('distillation_loss', 0.0),
            metrics.get('student_loss', 0.0),
            metrics.get('combined_loss', 0.0),
            metrics.get('temperature', 4.0),
            metrics.get('alpha', 0.7),
            metrics.get('kl_divergence', 0.0),
            datetime.now()
        ])
    
    def log_model_performance(self, experiment_id: str, performance_data: Dict):
        """Log model performance metrics."""
        self.conn.execute("""
            INSERT INTO model_performance 
            (experiment_id, checkpoint_name, model_size_mb, parameters_count, 
             inference_time_ms, memory_usage_mb, perplexity, bleu_score, 
             rouge_scores, custom_metrics, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            experiment_id,
            performance_data.get('checkpoint_name', 'final'),
            performance_data.get('model_size_mb', 0.0),
            performance_data.get('parameters_count', 0),
            performance_data.get('inference_time_ms', 0.0),
            performance_data.get('memory_usage_mb', 0.0),
            performance_data.get('perplexity', 0.0),
            performance_data.get('bleu_score', 0.0),
            json.dumps(performance_data.get('rouge_scores', {})),
            json.dumps(performance_data.get('custom_metrics', {})),
            datetime.now()
        ])
    
    def finish_experiment(self, experiment_id: str, final_metrics: Dict, status: str = "completed"):
        """Mark experiment as finished."""
        self.conn.execute("""
            UPDATE experiments 
            SET end_time = ?, status = ?, final_metrics = ?
            WHERE experiment_id = ?
        """, [datetime.now(), status, json.dumps(final_metrics), experiment_id])
    
    def generate_experiment_report(self, experiment_id: str = None, output_dir: str = "./reports"):
        """Generate comprehensive experiment report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if experiment_id:
            experiments = [experiment_id]
        else:
            # Get all experiments
            exp_df = self.conn.execute("SELECT experiment_id FROM experiments").df()
            experiments = exp_df['experiment_id'].tolist()
        
        for exp_id in experiments:
            self._generate_single_experiment_report(exp_id, output_path)
    
    def _generate_single_experiment_report(self, experiment_id: str, output_path: Path):
        """Generate report for a single experiment."""
        
        # Get experiment info
        exp_info = self.conn.execute("""
            SELECT * FROM experiments WHERE experiment_id = ?
        """, [experiment_id]).df()
        
        if exp_info.empty:
            logging.warning(f"No experiment found with ID: {experiment_id}")
            return
        
        exp_path = output_path / experiment_id
        exp_path.mkdir(exist_ok=True)
        
        # Export all data to CSV
        tables = [
            "training_metrics", "distillation_metrics", "model_performance",
            "progressive_stages", "synthetic_data_tracking"
        ]
        
        for table in tables:
            df = self.conn.execute(f"""
                SELECT * FROM {table} WHERE experiment_id = ?
            """, [experiment_id]).df()
            
            if not df.empty:
                df.to_csv(exp_path / f"{table}.csv", index=False)
        
        # Generate visualizations
        self._create_training_plots(experiment_id, exp_path)
        
        # Generate summary report
        summary = self._create_experiment_summary(experiment_id)
        with open(exp_path / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        logging.info(f"Report generated for experiment {experiment_id} at {exp_path}")
    
    def _create_training_plots(self, experiment_id: str, output_path: Path):
        """Create training visualization plots."""
        try:
            # Training metrics over time
            metrics_df = self.conn.execute("""
                SELECT step, metric_name, metric_value, phase 
                FROM training_metrics 
                WHERE experiment_id = ? AND metric_name IN ('loss', 'perplexity', 'learning_rate')
            """, [experiment_id]).df()
            
            if not metrics_df.empty:
                plt.figure(figsize=(15, 10))
                
                unique_metrics = metrics_df['metric_name'].unique()
                for i, metric in enumerate(unique_metrics):
                    plt.subplot(2, 2, i + 1)
                    metric_data = metrics_df[metrics_df['metric_name'] == metric]
                    
                    for phase in metric_data['phase'].unique():
                        phase_data = metric_data[metric_data['phase'] == phase]
                        plt.plot(phase_data['step'], phase_data['metric_value'], 
                               label=f"{phase}", marker='o', markersize=2)
                    
                    plt.title(f"{metric.capitalize()} over Training Steps")
                    plt.xlabel("Step")
                    plt.ylabel(metric.capitalize())
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_path / "training_metrics.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Distillation metrics
            distill_df = self.conn.execute("""
                SELECT step, distillation_loss, student_loss, combined_loss 
                FROM distillation_metrics 
                WHERE experiment_id = ?
            """, [experiment_id]).df()
            
            if not distill_df.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(distill_df['step'], distill_df['distillation_loss'], 
                        label='Distillation Loss', marker='o', markersize=2)
                plt.plot(distill_df['step'], distill_df['student_loss'], 
                        label='Student Loss', marker='s', markersize=2)
                plt.plot(distill_df['step'], distill_df['combined_loss'], 
                        label='Combined Loss', marker='^', markersize=2)
                
                plt.title("Knowledge Distillation Loss Progression")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(output_path / "distillation_losses.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logging.warning(f"Failed to create plots: {e}")
    
    def _create_experiment_summary(self, experiment_id: str) -> Dict:
        """Create experiment summary."""
        # Get basic experiment info
        exp_info = self.conn.execute("""
            SELECT * FROM experiments WHERE experiment_id = ?
        """, [experiment_id]).fetchone()
        
        # Get training summary
        training_summary = self.conn.execute("""
            SELECT 
                metric_name,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                AVG(metric_value) as avg_value,
                COUNT(*) as count
            FROM training_metrics 
            WHERE experiment_id = ?
            GROUP BY metric_name
        """, [experiment_id]).fetchall()
        
        # Get final performance
        final_performance = self.conn.execute("""
            SELECT * FROM model_performance 
            WHERE experiment_id = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, [experiment_id]).fetchone()
        
        return {
            "experiment_info": exp_info,
            "training_summary": training_summary,
            "final_performance": final_performance,
            "generated_at": datetime.now().isoformat()
        }
    
    def get_experiment_comparison(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments."""
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_info = self.conn.execute("""
                SELECT experiment_name, config, final_metrics 
                FROM experiments WHERE experiment_id = ?
            """, [exp_id]).fetchone()
            
            if exp_info:
                comparison_data.append({
                    'experiment_id': exp_id,
                    'experiment_name': exp_info[0],
                    'config': json.loads(exp_info[1]) if exp_info[1] else {},
                    'final_metrics': json.loads(exp_info[2]) if exp_info[2] else {}
                })
        
        return pd.DataFrame(comparison_data)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="VishwamAI Training with DuckDB Integration")
    
    # Configuration
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--experiment-name", type=str, default="vishwamai_training",
                       help="Name for the experiment")
    parser.add_argument("--db-path", type=str, default="./vishwamai_experiments.duckdb",
                       help="Path to DuckDB database")
    
    # Training options
    parser.add_argument("--preset", type=str, choices=["small", "medium", "large", "multimodal"],
                       help="Use preset configuration")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1",
                       help="Dataset name")
    parser.add_argument("--progressive", action="store_true",
                       help="Use progressive training")
    parser.add_argument("--use-synthetic-data", action="store_true",
                       help="Generate synthetic training data")
    
    # Report generation
    parser.add_argument("--generate-report", action="store_true",
                       help="Generate experiment report")
    parser.add_argument("--experiment-id", type=str,
                       help="Specific experiment ID for report generation")
    parser.add_argument("--compare-experiments", nargs="+",
                       help="Compare multiple experiments")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Initialize DuckDB tracker
    tracker = VishwamAIDuckDBTracker(args.db_path)
    
    try:
        if args.generate_report:
            logger.info("Generating experiment report...")
            tracker.generate_experiment_report(args.experiment_id)
            return
        
        if args.compare_experiments:
            logger.info("Comparing experiments...")
            comparison_df = tracker.get_experiment_comparison(args.compare_experiments)
            print(comparison_df)
            return
        
        # Load configuration
        if args.config:
            with open(args.config, 'r') as f:
                config_dict = json.load(f)
            config = DistillationConfig(**config_dict)
        else:
            config = DistillationConfig()
        
        # Override with preset if specified
        if args.preset:
            if args.preset == "small":
                config.student_config.dim = 768
                config.student_config.depth = 6
            elif args.preset == "medium":
                config.student_config.dim = 1024
                config.student_config.depth = 12
            elif args.preset == "large":
                config.student_config.dim = 2048
                config.student_config.depth = 24
        
        # Get hardware info
        hardware_info = get_hardware_info()
        
        # Start experiment
        experiment_id = tracker.start_experiment(
            args.experiment_name,
            config.__dict__,
            hardware_info,
            {"dataset": args.dataset}
        )
        
        logger.info(f"🚀 Starting VishwamAI training with DuckDB tracking")
        logger.info(f"📊 Experiment ID: {experiment_id}")
        logger.info(f"🗄️  Database: {args.db_path}")
        
        # Create dataset
        texts, synthetic_data = create_distillation_dataset(
            args.dataset,
            max_samples=10000,
            config=config if args.use_synthetic_data else None
        )
        
        # Log dataset info
        tracker.log_training_metrics(experiment_id, {
            "dataset_size": len(texts),
            "synthetic_samples": len(synthetic_data) if synthetic_data else 0
        }, 0, 0, "data")
        
        # Create tokenizer and datasets
        tokenizer = VishwamAITokenizer()
        split_idx = int(len(texts) * 0.9)
        
        from vishwamai.distillation import DistillationDataset
        train_dataset = DistillationDataset(texts[:split_idx], tokenizer, config.max_seq_length)
        eval_dataset = DistillationDataset(texts[split_idx:], tokenizer, config.max_seq_length)
        
        # Train model with tracking
        logger.info("🏋️  Starting distillation training...")
        trained_model = train_with_distillation(config, train_dataset, eval_dataset, tokenizer)
        
        # Final evaluation
        logger.info("📊 Running final evaluation...")
        eval_results = evaluate_distilled_model(trained_model, tokenizer, eval_dataset)
        
        # Log final metrics
        tracker.log_model_performance(experiment_id, {
            "checkpoint_name": "final",
            "perplexity": eval_results.get("perplexity", 0.0),
            "custom_metrics": eval_results
        })
        
        # Finish experiment
        tracker.finish_experiment(experiment_id, eval_results, "completed")
        
        # Generate report
        tracker.generate_experiment_report(experiment_id)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"📁 Results saved to database: {args.db_path}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if 'experiment_id' in locals():
            tracker.finish_experiment(experiment_id, {}, "failed")
        raise
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
