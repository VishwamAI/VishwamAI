# VishwamAI Training with DuckDB Integration

This documentation explains how to use DuckDB instead of Weights & Biases (wandb) for experiment tracking, metrics logging, and data analysis in VishwamAI training workflows.

## 🎯 Overview

DuckDB provides a lightweight, serverless alternative to cloud-based experiment tracking systems. It offers:

- **Local Data Control**: All experiment data stored locally in a single file
- **SQL Analytics**: Powerful SQL queries for data analysis
- **CSV Export**: Easy data export for external analysis
- **Visualization**: Built-in plotting capabilities
- **Performance**: Fast local database with excellent analytics performance

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install duckdb pandas matplotlib seaborn
```

### 2. Basic Training with DuckDB

```bash
# Run distillation training with DuckDB tracking
python scripts/train_with_duckdb.py --experiment-name "my_first_experiment"

# Use predefined configuration
python scripts/train_with_duckdb.py --config configs/distillation_config.json

# Progressive training with synthetic data
python scripts/train_with_duckdb.py --progressive --use-synthetic-data --preset medium
```

### 3. Generate Experiment Reports

```bash
# Generate report for latest experiment
python scripts/train_with_duckdb.py --generate-report

# Generate report for specific experiment
python scripts/train_with_duckdb.py --generate-report --experiment-id "vishwamai_training_20250619_143022"

# Compare multiple experiments
python scripts/train_with_duckdb.py --compare-experiments exp1 exp2 exp3
```

## 📊 Database Schema

The DuckDB integration uses a comprehensive schema designed for ML experiment tracking:

### Core Tables

1. **experiments**: Main experiment metadata
2. **training_metrics**: Step-by-step training metrics
3. **distillation_metrics**: Knowledge distillation specific metrics
4. **model_performance**: Model performance and evaluation metrics
5. **progressive_stages**: Progressive training stage tracking
6. **synthetic_data_tracking**: Synthetic data generation metrics

### Example Queries

```sql
-- Get all experiments with their final performance
SELECT 
    experiment_name,
    start_time,
    status,
    JSON_EXTRACT(final_metrics, '$.perplexity') as final_perplexity
FROM experiments
ORDER BY start_time DESC;

-- Training loss progression for an experiment
SELECT step, metric_value as loss
FROM training_metrics 
WHERE experiment_id = 'your_experiment_id' 
  AND metric_name = 'loss'
ORDER BY step;

-- Compare distillation vs student loss
SELECT 
    step,
    distillation_loss,
    student_loss,
    combined_loss
FROM distillation_metrics
WHERE experiment_id = 'your_experiment_id'
ORDER BY step;
```

## 🔧 Configuration

### Complete Configuration Example

```json
{
  "model_config": {
    "student_config": {
      "dim": 1024,
      "depth": 12,
      "heads": 16,
      "vocab_size": 50304,
      "max_seq_len": 1024
    },
    "teacher_model_name": "microsoft/DialoGPT-medium"
  },
  "training_config": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "learning_rate": 5e-5,
    "max_seq_length": 512
  },
  "distillation_config": {
    "temperature": 4.0,
    "distillation_alpha": 0.7,
    "use_progressive_distillation": true
  },
  "tracking_config": {
    "use_duckdb": true,
    "db_path": "./vishwamai_experiments.duckdb",
    "experiment_name": "vishwamai_distillation",
    "export_to_csv": true,
    "generate_plots": true
  }
}
```

## 📈 Analytics and Reporting

### Built-in Analytics

The DuckDB tracker provides several built-in analytics:

1. **Training Progression**: Loss curves, learning rate schedules
2. **Distillation Analysis**: Teacher-student knowledge transfer metrics
3. **Model Performance**: Inference time, memory usage, accuracy metrics
4. **Data Quality**: Synthetic data quality scores and distribution

### Custom Analytics

Access the database directly for custom analysis:

```python
import duckdb

# Connect to your experiment database
conn = duckdb.connect("vishwamai_experiments.duckdb")

# Custom query example
results = conn.execute("""
    SELECT 
        experiment_name,
        AVG(metric_value) as avg_loss,
        MIN(metric_value) as best_loss
    FROM experiments e
    JOIN training_metrics tm ON e.experiment_id = tm.experiment_id
    WHERE tm.metric_name = 'loss'
    GROUP BY experiment_name
""").df()

print(results)
```

### Export Options

```python
from scripts.train_with_duckdb import VishwamAIDuckDBTracker

tracker = VishwamAIDuckDBTracker("vishwamai_experiments.duckdb")

# Export specific experiment
tracker.generate_experiment_report("experiment_id", "./reports")

# Export all experiments
tracker.generate_experiment_report(output_dir="./all_reports")

# Get comparison data
comparison = tracker.get_experiment_comparison(["exp1", "exp2", "exp3"])
print(comparison)
```

## 🎮 Advanced Usage

### 1. Progressive Training with Tracking

```python
from scripts.train_with_duckdb import VishwamAIDuckDBTracker

tracker = VishwamAIDuckDBTracker()
experiment_id = tracker.start_experiment("progressive_training", config, hardware_info)

# Log each stage
for stage_idx, stage in enumerate(progressive_stages):
    # Training for this stage...
    
    tracker.log_training_metrics(experiment_id, {
        "stage": stage_idx,
        "stage_loss": stage_loss,
        "stage_perplexity": stage_perplexity
    }, step, epoch, f"stage_{stage_idx}")
```

### 2. Hyperparameter Optimization Integration

```python
def objective(trial):
    # Get hyperparameters from trial
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    
    # Create experiment
    experiment_id = tracker.start_experiment(f"hpo_trial_{trial.number}", config, hardware_info)
    
    # Train model...
    final_loss = train_model(lr, batch_size)
    
    # Log trial results
    tracker.conn.execute("""
        INSERT INTO hyperparameter_trials 
        (experiment_id, trial_id, hyperparameters, objective_value, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [experiment_id, str(trial.number), json.dumps(trial.params), final_loss, "completed", datetime.now()])
    
    return final_loss
```

### 3. Multi-Model Comparison

```python
# Compare different model architectures
models = ["small", "medium", "large"]
experiment_ids = []

for model_size in models:
    config.student_config = get_model_config(model_size)
    exp_id = train_with_tracking(f"comparison_{model_size}", config)
    experiment_ids.append(exp_id)

# Generate comparison report
comparison_df = tracker.get_experiment_comparison(experiment_ids)
comparison_df.to_csv("model_comparison.csv")
```

## 🔍 Monitoring and Debugging

### Real-time Monitoring

```python
# Monitor training progress
def monitor_experiment(experiment_id):
    while True:
        latest_metrics = tracker.conn.execute("""
            SELECT metric_name, metric_value, timestamp
            FROM training_metrics
            WHERE experiment_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        """, [experiment_id]).df()
        
        print(f"Latest metrics:\n{latest_metrics}")
        time.sleep(60)  # Check every minute
```

### Debug Training Issues

```python
# Find experiments with issues
problematic = tracker.conn.execute("""
    SELECT experiment_id, experiment_name, config
    FROM experiments
    WHERE status = 'failed' OR 
          JSON_EXTRACT(final_metrics, '$.perplexity') > 1000
""").df()

print("Experiments with issues:", problematic)
```

## 🌐 Integration with Hugging Face

The DuckDB tracking integrates seamlessly with Hugging Face workflows:

```python
from vishwamai.huggingface_integration import VishwamAIForCausalLM

# Train with DuckDB tracking
trained_model = train_with_distillation(config, train_dataset, eval_dataset, tokenizer)

# Save to Hugging Face format
trained_model.save_pretrained("./models/vishwamai-distilled")

# Log model info
tracker.log_model_performance(experiment_id, {
    "model_path": "./models/vishwamai-distilled",
    "model_size_mb": get_model_size("./models/vishwamai-distilled"),
    "huggingface_compatible": True
})
```

## 🚀 Performance Tips

1. **Batch Inserts**: Use batch inserts for high-frequency logging
2. **Indexing**: Create indexes on frequently queried columns
3. **Partitioning**: Consider partitioning large tables by experiment_id
4. **Cleanup**: Regularly archive old experiments

```sql
-- Create useful indexes
CREATE INDEX idx_training_metrics_exp_step ON training_metrics(experiment_id, step);
CREATE INDEX idx_experiments_name_time ON experiments(experiment_name, start_time);

-- Archive old experiments
CREATE TABLE archived_experiments AS SELECT * FROM experiments WHERE start_time < '2024-01-01';
DELETE FROM experiments WHERE start_time < '2024-01-01';
```

## 🔧 Troubleshooting

### Common Issues

1. **Database Lock**: Ensure only one process writes to the database
2. **Memory Usage**: Monitor memory for large experiments
3. **Disk Space**: DuckDB files can grow large with extensive logging

### Solutions

```python
# Handle database locks gracefully
try:
    tracker.log_training_metrics(experiment_id, metrics, step, epoch)
except Exception as e:
    logger.warning(f"Failed to log metrics: {e}")
    # Continue training without logging
```

## 📚 Examples

Complete examples are available in the `examples/` directory:

- `basic_duckdb_training.py`: Simple training with DuckDB
- `progressive_distillation.py`: Progressive training with stage tracking  
- `hyperparameter_optimization.py`: HPO with DuckDB logging
- `model_comparison.py`: Multi-model comparison workflow

## 🤝 Contributing

To contribute to the DuckDB integration:

1. Follow the existing schema conventions
2. Add proper error handling
3. Include tests for new functionality
4. Update documentation

## 📄 License

This DuckDB integration follows the same license as the main VishwamAI project.
