#!/usr/bin/env python3
"""
Simple example of VishwamAI training with DuckDB tracking.

This example demonstrates the basic usage of DuckDB instead of wandb
for experiment tracking and metrics logging.
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def basic_duckdb_training():
    """Basic training example with DuckDB tracking."""
    
    logger.info("🚀 VishwamAI Training with DuckDB - Basic Example")
    
    try:
        from scripts.train_with_duckdb import VishwamAIDuckDBTracker
        from vishwamai.distillation import DistillationConfig
        from vishwamai import get_hardware_info
        
        # Initialize DuckDB tracker
        tracker = VishwamAIDuckDBTracker("example_experiments.duckdb")
        
        # Create configuration
        config = DistillationConfig(
            student_config={
                "dim": 768,
                "depth": 6,
                "heads": 12,
                "vocab_size": 50304,
                "max_seq_len": 512
            },
            num_train_epochs=1,
            per_device_train_batch_size=4,
            learning_rate=5e-5,
            max_seq_length=256,
            use_duckdb=True,
            db_path="example_experiments.duckdb"
        )
        
        # Get hardware info
        hardware_info = get_hardware_info()
        
        # Start experiment
        experiment_id = tracker.start_experiment(
            "basic_example",
            config.__dict__,
            hardware_info,
            {"dataset": "dummy", "description": "Basic training example"}
        )
        
        logger.info(f"📊 Started experiment: {experiment_id}")
        
        # Simulate training metrics
        for epoch in range(3):
            for step in range(10):
                # Simulate decreasing loss
                loss = 5.0 - (epoch * 10 + step) * 0.1
                perplexity = 2 ** loss
                
                # Log training metrics
                tracker.log_training_metrics(experiment_id, {
                    "loss": loss,
                    "perplexity": perplexity,
                    "learning_rate": 5e-5 * (0.95 ** step)
                }, step + epoch * 10, epoch, "train")
                
                # Log distillation metrics every few steps
                if step % 3 == 0:
                    tracker.log_distillation_metrics(experiment_id, {
                        "teacher_model": "microsoft/DialoGPT-medium",
                        "distillation_loss": loss * 0.7,
                        "student_loss": loss * 0.3,
                        "combined_loss": loss,
                        "temperature": 4.0,
                        "alpha": 0.7,
                        "kl_divergence": 0.5 + step * 0.01
                    }, step + epoch * 10, epoch)
                
                if step % 5 == 0:
                    logger.info(f"  Step {step + epoch * 10}: loss={loss:.3f}, perplexity={perplexity:.3f}")
        
        # Log final model performance
        tracker.log_model_performance(experiment_id, {
            "checkpoint_name": "final",
            "model_size_mb": 256.5,
            "parameters_count": 85000000,
            "inference_time_ms": 15.2,
            "memory_usage_mb": 512.0,
            "perplexity": 2.5,
            "bleu_score": 0.85,
            "custom_metrics": {
                "accuracy": 0.92,
                "f1_score": 0.89
            }
        })
        
        # Finish experiment
        final_metrics = {
            "final_loss": 2.0,
            "final_perplexity": 4.0,
            "training_time_seconds": 3600,
            "best_checkpoint": "epoch_2_step_25"
        }
        
        tracker.finish_experiment(experiment_id, final_metrics, "completed")
        
        # Generate report
        logger.info("📈 Generating experiment report...")
        tracker.generate_experiment_report(experiment_id, "./example_reports")
        
        logger.info("✅ Basic training example completed!")
        logger.info(f"📁 Database: example_experiments.duckdb")
        logger.info(f"📊 Report: ./example_reports/{experiment_id}/")
        
        # Show some analytics
        show_experiment_analytics(tracker, experiment_id)
        
    except ImportError as e:
        logger.error(f"❌ Missing dependencies: {e}")
        logger.info("💡 Install with: pip install duckdb pandas matplotlib seaborn")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'tracker' in locals():
            tracker.close()


def show_experiment_analytics(tracker, experiment_id):
    """Show some basic analytics from the experiment."""
    
    logger.info("📊 Experiment Analytics:")
    
    # Get training progression
    training_data = tracker.conn.execute("""
        SELECT step, metric_name, metric_value
        FROM training_metrics
        WHERE experiment_id = ? AND metric_name IN ('loss', 'perplexity')
        ORDER BY step
    """, [experiment_id]).fetchall()
    
    if training_data:
        logger.info("  Training Progression:")
        for step, metric, value in training_data[-5:]:  # Last 5 entries
            logger.info(f"    Step {step}: {metric} = {value:.3f}")
    
    # Get distillation summary
    distill_summary = tracker.conn.execute("""
        SELECT 
            AVG(distillation_loss) as avg_distill_loss,
            AVG(student_loss) as avg_student_loss,
            AVG(kl_divergence) as avg_kl_div
        FROM distillation_metrics
        WHERE experiment_id = ?
    """, [experiment_id]).fetchone()
    
    if distill_summary:
        logger.info("  Distillation Summary:")
        logger.info(f"    Avg Distillation Loss: {distill_summary[0]:.3f}")
        logger.info(f"    Avg Student Loss: {distill_summary[1]:.3f}")
        logger.info(f"    Avg KL Divergence: {distill_summary[2]:.3f}")
    
    # Get experiment comparison data
    all_experiments = tracker.conn.execute("""
        SELECT experiment_id, experiment_name, 
               JSON_EXTRACT(final_metrics, '$.final_perplexity') as final_perplexity
        FROM experiments
        ORDER BY start_time DESC
        LIMIT 5
    """).fetchall()
    
    if all_experiments:
        logger.info("  Recent Experiments Comparison:")
        for exp_id, exp_name, perplexity in all_experiments:
            logger.info(f"    {exp_name}: perplexity = {perplexity}")


def advanced_analytics_example():
    """Example of advanced analytics with DuckDB."""
    
    logger.info("📊 Advanced Analytics Example")
    
    try:
        import duckdb
        import pandas as pd
        
        # Connect to database
        conn = duckdb.connect("example_experiments.duckdb")
        
        # Complex analytics query
        results = conn.execute("""
            WITH training_stats AS (
                SELECT 
                    e.experiment_name,
                    e.experiment_id,
                    COUNT(tm.step) as total_steps,
                    MIN(tm.metric_value) as min_loss,
                    MAX(tm.metric_value) as max_loss,
                    AVG(tm.metric_value) as avg_loss
                FROM experiments e
                JOIN training_metrics tm ON e.experiment_id = tm.experiment_id
                WHERE tm.metric_name = 'loss'
                GROUP BY e.experiment_name, e.experiment_id
            ),
            distill_stats AS (
                SELECT 
                    experiment_id,
                    AVG(kl_divergence) as avg_kl_div,
                    AVG(distillation_loss / (distillation_loss + student_loss)) as distill_ratio
                FROM distillation_metrics
                GROUP BY experiment_id
            )
            SELECT 
                ts.experiment_name,
                ts.total_steps,
                ts.min_loss,
                ts.avg_loss,
                ds.avg_kl_div,
                ds.distill_ratio
            FROM training_stats ts
            LEFT JOIN distill_stats ds ON ts.experiment_id = ds.experiment_id
            ORDER BY ts.avg_loss
        """).df()
        
        logger.info("📈 Advanced Analytics Results:")
        print(results)
        
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Analytics failed: {e}")


if __name__ == "__main__":
    # Run basic example
    basic_duckdb_training()
    
    # Run advanced analytics
    advanced_analytics_example()
    
    logger.info("🎉 All examples completed!")
