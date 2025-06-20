#!/usr/bin/env python3
"""
VishwamAI Knowledge Distillation Training Script

This script provides a complete workflow for training VishwamAI models
using knowledge distillation with multiple teacher models, synthetic
data generation, and progressive training strategies.

Example usage:
    python train_distillation.py --config configs/distillation_config.json
    python train_distillation.py --preset small --dataset wikitext
    python train_distillation.py --preset multimodal --use-synthetic-data
"""

import argparse
import logging
import json
import torch
import jax
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Optional imports
try:
    import duckdb
    import pandas as pd
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

from vishwamai.distillation import (
    DistillationConfig,
    train_with_distillation,
    evaluate_distilled_model,
    create_distillation_dataset,
    DistillationDataset,
    SMALL_DISTILLATION_CONFIG,
    MEDIUM_DISTILLATION_CONFIG,
    MULTIMODAL_DISTILLATION_CONFIG
)
from vishwamai.huggingface_integration import (
    VishwamAIForCausalLM,
    VishwamAIConfig,
    VishwamAITokenizer,
    save_vishwamai_model
)
from vishwamai import get_hardware_info, setup_mixed_precision


class DuckDBTracker:
    """DuckDB-based experiment tracking system for VishwamAI distillation training."""
    
    def __init__(self, db_path: str = "experiments.duckdb", experiment_name: str = None):
        """Initialize DuckDB tracker with database and experiment tracking."""
        self.db_path = db_path
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conn = duckdb.connect(db_path)
        self.run_id = self._create_run()
        self._setup_tables()
        
    def _setup_tables(self):
        """Create necessary tables for experiment tracking."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                experiment_name TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                config JSON,
                hardware_info JSON,
                final_metrics JSON
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT,
                step INTEGER,
                epoch INTEGER,
                metric_name TEXT,
                metric_value DOUBLE,
                metric_type TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                run_id TEXT,
                checkpoint_name TEXT,
                step INTEGER,
                epoch INTEGER,
                model_path TEXT,
                metrics JSON,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS training_logs (
                run_id TEXT,
                step INTEGER,
                epoch INTEGER,
                log_level TEXT,
                message TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS synthetic_data (
                run_id TEXT,
                sample_id TEXT,
                prompt TEXT,
                generated_text TEXT,
                teacher_model TEXT,
                generation_params JSON,
                quality_score DOUBLE,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES experiments(run_id)
            )
        """)
    
    def _create_run(self) -> str:
        """Create a new experiment run."""
        run_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return run_id
    
    def start_experiment(self, config: dict, hardware_info: dict):
        """Start a new experiment run."""
        self.conn.execute("""
            INSERT INTO experiments (run_id, experiment_name, start_time, status, config, hardware_info)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            self.run_id,
            self.experiment_name,
            datetime.now(),
            "running",
            json.dumps(config),
            json.dumps(hardware_info)
        ])
    
    def log_metrics(self, metrics: dict, step: int = None, epoch: int = None, metric_type: str = "training"):
        """Log metrics to DuckDB."""
        timestamp = datetime.now()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.conn.execute("""
                    INSERT INTO metrics (run_id, step, epoch, metric_name, metric_value, metric_type, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [self.run_id, step, epoch, name, float(value), metric_type, timestamp])
    
    def log_checkpoint(self, checkpoint_name: str, step: int, epoch: int, model_path: str, metrics: dict):
        """Log model checkpoint information."""
        self.conn.execute("""
            INSERT INTO model_checkpoints (run_id, checkpoint_name, step, epoch, model_path, metrics, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            self.run_id,
            checkpoint_name,
            step,
            epoch,
            model_path,
            json.dumps(metrics),
            datetime.now()
        ])
    
    def log_training_event(self, message: str, level: str = "INFO", step: int = None, epoch: int = None):
        """Log training events and messages."""
        self.conn.execute("""
            INSERT INTO training_logs (run_id, step, epoch, log_level, message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [self.run_id, step, epoch, level, message, datetime.now()])
    
    def log_synthetic_data(self, samples: list):
        """Log synthetic data generation information."""
        timestamp = datetime.now()
        for i, sample in enumerate(samples):
            sample_id = f"{self.run_id}_synthetic_{i}"
            self.conn.execute("""
                INSERT INTO synthetic_data (run_id, sample_id, prompt, generated_text, teacher_model, 
                                          generation_params, quality_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                self.run_id,
                sample_id,
                sample.get("prompt", ""),
                sample.get("generated_text", ""),
                sample.get("teacher", "unknown"),
                json.dumps(sample.get("params", {})),
                sample.get("quality_score", 0.0),
                timestamp
            ])
    
    def finish_experiment(self, final_metrics: dict, status: str = "completed"):
        """Mark experiment as finished."""
        self.conn.execute("""
            UPDATE experiments 
            SET end_time = ?, status = ?, final_metrics = ?
            WHERE run_id = ?
        """, [datetime.now(), status, json.dumps(final_metrics), self.run_id])
    
    def get_experiment_summary(self) -> dict:
        """Get a summary of the current experiment."""
        exp_data = self.conn.execute("""
            SELECT * FROM experiments WHERE run_id = ?
        """, [self.run_id]).fetchone()
        
        metrics_data = self.conn.execute("""
            SELECT metric_name, metric_value, metric_type, step, epoch
            FROM metrics WHERE run_id = ? 
            ORDER BY step, epoch
        """, [self.run_id]).fetchall()
        
        return {
            "experiment": exp_data,
            "metrics": metrics_data
        }
    
    def export_to_csv(self, output_dir: str):
        """Export experiment data to CSV files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export metrics
        metrics_df = self.conn.execute("""
            SELECT * FROM metrics WHERE run_id = ?
        """, [self.run_id]).df()
        metrics_df.to_csv(f"{output_dir}/metrics.csv", index=False)
        
        # Export checkpoints
        checkpoints_df = self.conn.execute("""
            SELECT * FROM model_checkpoints WHERE run_id = ?
        """, [self.run_id]).df()
        checkpoints_df.to_csv(f"{output_dir}/checkpoints.csv", index=False)
        
        # Export training logs
        logs_df = self.conn.execute("""
            SELECT * FROM training_logs WHERE run_id = ?
        """, [self.run_id]).df()
        logs_df.to_csv(f"{output_dir}/training_logs.csv", index=False)
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("distillation_training.log")
        ]
    )


def load_config(config_path: str) -> DistillationConfig:
    """Load distillation configuration from file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert nested student_config
    if 'student_config' in config_dict:
        student_config_dict = config_dict.pop('student_config')
        student_config = VishwamAIConfig(**student_config_dict)
        config_dict['student_config'] = student_config
    
    return DistillationConfig(**config_dict)


def save_config(config: DistillationConfig, output_dir: str) -> None:
    """Save distillation configuration to file."""
    config_dict = config.__dict__.copy()
    
    # Convert student_config to dict
    if hasattr(config_dict['student_config'], '__dict__'):
        config_dict['student_config'] = config_dict['student_config'].__dict__
    
    output_path = Path(output_dir) / "distillation_config.json"
    with open(output_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


def create_synthetic_data_with_api(
    prompts: list,
    api_model: str = "gpt-3.5-turbo",
    max_samples: int = 1000
) -> list:
    """
    Generate synthetic data using API-based models (GPT, Claude, etc.).
    This can be used as an alternative teacher model for distillation.
    """
    synthetic_data = []
    
    try:
        import openai
        
        for i, prompt in enumerate(prompts[:max_samples]):
            try:
                response = openai.ChatCompletion.create(
                    model=api_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates high-quality educational content."},
                        {"role": "user", "content": f"Continue this text in a natural and informative way: {prompt}"}
                    ],
                    max_tokens=256,
                    temperature=0.7
                )
                
                generated_text = response.choices[0].message.content
                synthetic_data.append({
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "teacher": api_model,
                    "type": "synthetic_api"
                })
                
                if i % 10 == 0:
                    logging.info(f"Generated {i+1}/{len(prompts)} synthetic samples")
                    
            except Exception as e:
                logging.warning(f"Failed to generate with API for prompt {i}: {e}")
                continue
    
    except ImportError:
        logging.warning("OpenAI package not installed. Skipping API-based synthetic data generation.")
    
    return synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Train VishwamAI with Knowledge Distillation")
    
    # Configuration options
    parser.add_argument("--config", type=str, help="Path to distillation config JSON file")
    parser.add_argument("--preset", type=str, choices=["small", "medium", "multimodal"], 
                       help="Use predefined configuration preset")
    
    # Data options
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1",
                       help="Dataset name from Hugging Face")
    parser.add_argument("--max-samples", type=int, default=10000,
                       help="Maximum number of training samples")
    parser.add_argument("--use-synthetic-data", action="store_true",
                       help="Generate synthetic training data")
    parser.add_argument("--synthetic-ratio", type=float, default=0.3,
                       help="Ratio of synthetic to real data")
    
    # Model options
    parser.add_argument("--student-size", type=str, choices=["small", "medium", "large"], 
                       default="medium", help="Student model size")
    parser.add_argument("--teacher-models", nargs="+", 
                       default=["microsoft/DialoGPT-medium"],
                       help="Teacher model names")
    
    # Training options
    parser.add_argument("--output-dir", type=str, default="./distillation_outputs",
                       help="Output directory for model and logs")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512,
                       help="Maximum sequence length")
    
    # Distillation options
    parser.add_argument("--temperature", type=float, default=4.0,
                       help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7,
                       help="Distillation loss weight")
    parser.add_argument("--use-attention-distillation", action="store_true",
                       help="Enable attention transfer")
    parser.add_argument("--use-hidden-distillation", action="store_true",
                       help="Enable hidden state distillation")
    
    # Progressive training
    parser.add_argument("--progressive", action="store_true",
                       help="Use progressive distillation")
    parser.add_argument("--progressive-stages", type=str,
                       help="JSON string defining progressive stages")
    
    # Monitoring options
    parser.add_argument("--use-duckdb", action="store_true", default=True,
                       help="Use DuckDB for experiment tracking (default: True)")
    parser.add_argument("--db-path", type=str, default="./experiments.duckdb",
                       help="Path to DuckDB database file")
    parser.add_argument("--experiment-name", type=str,
                       help="Experiment name for tracking")
    
    # Evaluation options
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate existing model")
    parser.add_argument("--model-path", type=str,
                       help="Path to pre-trained model for evaluation")
    
    # Advanced options
    parser.add_argument("--use-api-teacher", action="store_true",
                       help="Use API-based teacher models (GPT, Claude)")
    parser.add_argument("--api-model", type=str, default="gpt-3.5-turbo",
                       help="API model name for synthetic data generation")
    parser.add_argument("--mixed-precision", action="store_true",
                       help="Use mixed precision training")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Use gradient checkpointing to save memory")
    
    # Utility options
    parser.add_argument("--dry-run", action="store_true",
                       help="Validate configuration without training")
    parser.add_argument("--save-config", action="store_true",
                       help="Save final configuration to output directory")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 VishwamAI Knowledge Distillation Training")
    logger.info("=" * 60)
    
    # Load or create configuration
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
    elif args.preset:
        logger.info(f"Using preset configuration: {args.preset}")
        if args.preset == "small":
            config = SMALL_DISTILLATION_CONFIG
        elif args.preset == "medium":
            config = MEDIUM_DISTILLATION_CONFIG
        elif args.preset == "multimodal":
            config = MULTIMODAL_DISTILLATION_CONFIG
    else:
        # Create default configuration
        logger.info("Creating default configuration")
        config = DistillationConfig()
    
    # Override configuration with command line arguments
    if args.teacher_models != ["microsoft/DialoGPT-medium"]:
        config.teacher_models = args.teacher_models
        config.use_multiple_teachers = len(args.teacher_models) > 1
    
    config.output_dir = args.output_dir
    config.num_train_epochs = args.epochs
    config.per_device_train_batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.max_seq_length = args.max_seq_length
    config.temperature = args.temperature
    config.distillation_alpha = args.alpha
    config.use_attention_distillation = args.use_attention_distillation
    config.use_hidden_state_distillation = args.use_hidden_distillation
    config.use_progressive_distillation = args.progressive
    config.use_synthetic_data = args.use_synthetic_data
    config.synthetic_data_ratio = args.synthetic_ratio
    config.use_duckdb = args.use_duckdb
    config.db_path = args.db_path
    config.experiment_name = args.experiment_name
    
    # Handle progressive stages
    if args.progressive_stages:
        try:
            config.progressive_stages = json.loads(args.progressive_stages)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid progressive stages JSON: {e}")
            return
    
    # Student model size configuration
    if args.student_size == "small":
        config.student_config.dim = 768
        config.student_config.depth = 6
        config.student_config.heads = 12
    elif args.student_size == "medium":
        config.student_config.dim = 1536
        config.student_config.depth = 12
        config.student_config.heads = 16
    elif args.student_size == "large":
        config.student_config.dim = 2048
        config.student_config.depth = 24
        config.student_config.heads = 32
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration if requested
    if args.save_config:
        save_config(config, config.output_dir)
        logger.info(f"Configuration saved to {config.output_dir}/distillation_config.json")
    
    # Print configuration summary
    logger.info("\n📋 Configuration Summary:")
    logger.info(f"  Student Model: {config.student_config.dim}d, {config.student_config.depth} layers")
    logger.info(f"  Teacher Models: {config.teacher_models if config.use_multiple_teachers else [config.teacher_model_name]}")
    logger.info(f"  Dataset: {args.dataset}")
    logger.info(f"  Training Epochs: {config.num_train_epochs}")
    logger.info(f"  Batch Size: {config.per_device_train_batch_size}")
    logger.info(f"  Learning Rate: {config.learning_rate}")
    logger.info(f"  Distillation Temperature: {config.temperature}")
    logger.info(f"  Progressive Training: {config.use_progressive_distillation}")
    logger.info(f"  Synthetic Data: {config.use_synthetic_data}")
    
    # Initialize DuckDB tracker
    db_tracker = None
    if config.use_duckdb and HAS_DUCKDB:
        logger.info(f"🗄️  Initializing DuckDB experiment tracking...")
        db_tracker = DuckDBTracker(
            db_path=config.db_path,
            experiment_name=config.experiment_name
        )
        
        # Start experiment tracking
        config_dict = {
            'student_model_dim': config.student_config.dim,
            'student_model_depth': config.student_config.depth,
            'teacher_models': config.teacher_models if config.use_multiple_teachers else [config.teacher_model_name],
            'dataset': args.dataset,
            'training_epochs': config.num_train_epochs,
            'batch_size': config.per_device_train_batch_size,
            'learning_rate': config.learning_rate,
            'distillation_temperature': config.temperature,
            'progressive_training': config.use_progressive_distillation,
            'synthetic_data': config.use_synthetic_data
        }
        
        db_tracker.start_experiment(config_dict, hardware_info)
        db_tracker.log_training_event("Experiment started with DuckDB tracking")
        logger.info(f"  Database: {config.db_path}")
        logger.info(f"  Experiment: {db_tracker.experiment_name}")
        logger.info(f"  Run ID: {db_tracker.run_id}")
    elif not HAS_DUCKDB:
        logger.warning("⚠️  DuckDB not available. Install with: pip install duckdb pandas")
    
    # Hardware information
    hardware_info = get_hardware_info()
    logger.info(f"\n🖥️  Hardware Information:")
    logger.info(f"  Devices: {hardware_info['num_devices']} ({', '.join(hardware_info['device_types'])})")
    logger.info(f"  JAX Backend: {jax.default_backend()}")
    
    if hardware_info['has_gpu']:
        for i, gpu in enumerate(hardware_info.get('gpu_details', [])):
            logger.info(f"  GPU {i}: {gpu['name']} ({gpu['memory_total']}GB)")
    
    # Setup mixed precision if requested
    if args.mixed_precision:
        mp_config = setup_mixed_precision()
        logger.info(f"  Mixed Precision: {mp_config['use_mixed_precision']} ({mp_config['dtype']})")
    
    if args.dry_run:
        logger.info("✅ Configuration validation completed (dry run)")
        return
    
    try:
        # Evaluation only mode
        if args.eval_only:
            if not args.model_path:
                logger.error("--model-path required for evaluation mode")
                return
            
            logger.info(f"📊 Evaluating model from {args.model_path}")
            
            # Load model
            model = VishwamAIForCausalLM.from_pretrained(args.model_path)
            tokenizer = VishwamAITokenizer()
            
            # Create evaluation dataset
            texts, _ = create_distillation_dataset(
                args.dataset, split="test", max_samples=1000
            )
            eval_dataset = DistillationDataset(texts, tokenizer, config.max_seq_length)
            
            # Evaluate
            results = evaluate_distilled_model(model, tokenizer, eval_dataset)
            
            logger.info("📈 Evaluation Results:")
            for key, value in results.items():
                if key != "sample_generations":
                    logger.info(f"  {key}: {value}")
            
            # Log sample generations
            logger.info("\n📝 Sample Generations:")
            for sample in results["sample_generations"]:
                logger.info(f"  Input: {sample['input']}")
                logger.info(f"  Output: {sample['output']}")
                logger.info("")
            
            return
        
        # Create training dataset
        logger.info(f"📚 Creating training dataset from {args.dataset}...")
        texts, synthetic_data = create_distillation_dataset(
            args.dataset,
            max_samples=args.max_samples,
            config=config if config.use_synthetic_data else None
        )
        
        # Add API-generated synthetic data if requested
        if args.use_api_teacher and config.use_synthetic_data:
            logger.info("🤖 Generating synthetic data using API teacher...")
            api_synthetic = create_synthetic_data_with_api(
                texts[:100],  # Use first 100 texts as prompts
                args.api_model,
                max_samples=500
            )
            if synthetic_data:
                synthetic_data.extend(api_synthetic)
            else:
                synthetic_data = api_synthetic
            
            # Log synthetic data to DuckDB
            if db_tracker and api_synthetic:
                db_tracker.log_synthetic_data(api_synthetic)
                db_tracker.log_training_event(f"Generated {len(api_synthetic)} API synthetic samples")
                logger.info(f"  Logged {len(api_synthetic)} synthetic samples to database")
        
        # Create datasets
        split_idx = int(len(texts) * 0.9)  # 90% train, 10% eval
        
        tokenizer = VishwamAITokenizer(config.student_config.vocab_size)
        
        train_dataset = DistillationDataset(
            texts[:split_idx],
            tokenizer,
            config.max_seq_length,
            synthetic_data
        )
        
        eval_dataset = DistillationDataset(
            texts[split_idx:],
            tokenizer,
            config.max_seq_length
        )
        
        logger.info(f"  Training samples: {len(train_dataset)}")
        logger.info(f"  Evaluation samples: {len(eval_dataset)}")
        if synthetic_data:
            logger.info(f"  Synthetic samples: {len(synthetic_data)}")
        
        # Log dataset information to DuckDB
        if db_tracker:
            dataset_metrics = {
                'train_samples': len(train_dataset),
                'eval_samples': len(eval_dataset),
                'synthetic_samples': len(synthetic_data) if synthetic_data else 0,
                'total_samples': len(train_dataset) + len(eval_dataset)
            }
            db_tracker.log_metrics(dataset_metrics, step=0, metric_type="dataset")
            db_tracker.log_training_event(f"Dataset created: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
        
        # Train model
        logger.info("🏋️  Starting distillation training...")
        if db_tracker:
            db_tracker.log_training_event("Training started", "INFO", step=0, epoch=0)
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
        
        trained_model = train_with_distillation(
            config,
            train_dataset,
            eval_dataset,
            tokenizer
        )
        
        training_time = None
        if start_time:
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
            logger.info(f"⏱️  Training completed in {training_time:.2f} seconds")
            
            if db_tracker:
                db_tracker.log_metrics(
                    {'training_time_seconds': training_time}, 
                    metric_type="performance"
                )
                db_tracker.log_training_event(f"Training completed in {training_time:.2f} seconds")
        
        # Save model in Hugging Face format
        logger.info("💾 Saving trained model...")
        save_vishwamai_model(
            trained_model,
            config.output_dir,
            push_to_hub=False
        )
        
        # Final evaluation
        logger.info("📊 Running final evaluation...")
        eval_results = evaluate_distilled_model(
            trained_model,
            tokenizer,
            eval_dataset
        )
        
        # Save evaluation results
        with open(Path(config.output_dir) / "evaluation_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        logger.info("📈 Final Evaluation Results:")
        for key, value in eval_results.items():
            if key != "sample_generations":
                logger.info(f"  {key}: {value}")
        
        # Log final results to DuckDB
        if db_tracker:
            final_metrics = {k: v for k, v in eval_results.items() 
                           if k != "sample_generations" and isinstance(v, (int, float))}
            db_tracker.log_metrics(final_metrics, metric_type="final_evaluation")
            db_tracker.finish_experiment(final_metrics, "completed")
            
            # Export experiment data to CSV
            output_dir = Path(config.output_dir) / "experiment_data"
            db_tracker.export_to_csv(str(output_dir))
            logger.info(f"📊 Experiment data exported to CSV: {output_dir}")
            
            # Close the database connection
            db_tracker.close()
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📁 Model saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️  Training interrupted by user")
        if db_tracker:
            db_tracker.log_training_event("Training interrupted by user", "WARNING")
            db_tracker.finish_experiment({}, "interrupted")
            db_tracker.close()
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if db_tracker:
            db_tracker.log_training_event(f"Training failed: {e}", "ERROR")
            db_tracker.finish_experiment({}, "failed")
            db_tracker.close()
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
