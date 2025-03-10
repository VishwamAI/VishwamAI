"""
Logger module for VishwamAI using DuckDB for efficient storage and querying of training metrics.
"""

import duckdb
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
import os

class DuckDBLogger:
    """DuckDB-based logging system for training metrics and experiment tracking."""
    
    def __init__(
        self,
        db_path: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the DuckDB logger.
        
        Args:
            db_path: Path to the DuckDB database file
            experiment_name: Name of the current experiment
            config: Configuration dictionary for the experiment
        """
        self.db_path = db_path
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = datetime.now()
        
        # Connect to DuckDB
        self.conn = duckdb.connect(db_path)
        
        # Create necessary tables if they don't exist
        self._create_tables()
        
        # Log experiment metadata
        self._log_experiment(config or {})
    
    def _create_tables(self):
        """Create the necessary database tables if they don't exist."""
        # Experiments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                config JSON,
                status VARCHAR
            )
        """)
        
        # Metrics table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                experiment_id VARCHAR,
                step INTEGER,
                timestamp TIMESTAMP,
                metric_name VARCHAR,
                metric_value DOUBLE,
                metric_type VARCHAR,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
    
    def _log_experiment(self, config: Dict[str, Any]):
        """Log experiment metadata to the database."""
        self.conn.execute("""
            INSERT INTO experiments (experiment_id, start_time, config, status)
            VALUES (?, ?, ?, ?)
        """, [
            self.experiment_name,
            self.start_time,
            json.dumps(config),
            'running'
        ])
        
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = 'train'):
        """Log training metrics to the database.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
            prefix: Metric prefix (e.g., 'train' or 'eval')
        """
        timestamp = datetime.now()
        
        # Prepare data for insertion
        data = []
        for name, value in metrics.items():
            metric_name = f"{prefix}/{name}"
            data.append({
                'experiment_id': self.experiment_name,
                'step': step,
                'timestamp': timestamp,
                'metric_name': metric_name,
                'metric_value': float(value),
                'metric_type': prefix
            })
        
        # Convert to DataFrame and insert
        df = pd.DataFrame(data)
        self.conn.execute("""
            INSERT INTO metrics
            SELECT * FROM df
        """)
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment metrics.
        
        Returns:
            Dictionary containing experiment summary statistics
        """
        # Query metrics summary
        metrics_df = self.conn.execute("""
            SELECT 
                metric_name,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                AVG(metric_value) as mean_value,
                STDDEV(metric_value) as std_value
            FROM metrics
            WHERE experiment_id = ?
            GROUP BY metric_name
        """, [self.experiment_name]).fetchdf()
        
        # Convert to dictionary format
        metrics_summary = {}
        for _, row in metrics_df.iterrows():
            metrics_summary[row['metric_name']] = {
                'min': row['min_value'],
                'max': row['max_value'],
                'mean': row['mean_value'],
                'std': row['std_value']
            }
        
        # Get step count
        total_steps = self.conn.execute("""
            SELECT MAX(step) as max_step
            FROM metrics
            WHERE experiment_id = ?
        """, [self.experiment_name]).fetchone()[0]
        
        return {
            'experiment_id': self.experiment_name,
            'start_time': self.start_time,
            'end_time': datetime.now(),
            'total_steps': total_steps,
            'metrics_summary': metrics_summary
        }
    
    def export_to_csv(self, output_dir: str = 'logs'):
        """Export metrics to CSV files.
        
        Args:
            output_dir: Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Export metrics
        metrics_df = self.conn.execute("""
            SELECT * FROM metrics
            WHERE experiment_id = ?
            ORDER BY step, metric_name
        """, [self.experiment_name]).fetchdf()
        
        metrics_df.to_csv(
            f"{output_dir}/{self.experiment_name}_metrics.csv",
            index=False
        )
        
        # Export experiment metadata
        exp_df = self.conn.execute("""
            SELECT * FROM experiments
            WHERE experiment_id = ?
        """, [self.experiment_name]).fetchdf()
        
        exp_df.to_csv(
            f"{output_dir}/{self.experiment_name}_metadata.csv",
            index=False
        )
    
    def close(self):
        """Close the database connection and update experiment status."""
        # Update experiment end time and status
        self.conn.execute("""
            UPDATE experiments
            SET end_time = ?, status = 'completed'
            WHERE experiment_id = ?
        """, [datetime.now(), self.experiment_name])
        
        # Close connection
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()