"""
Logger module for VishwamAI using DuckDB for efficient storage and querying of training metrics.
"""

import duckdb
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional, Union
import numpy as np
import os
import logging
from pathlib import Path

class DuckDBLogger:
    """DuckDB-based logging system for training metrics."""
    
    def __init__(
        self,
        db_path: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize DuckDB logger.
        
        Args:
            db_path: Path to DuckDB database file
            experiment_name: Name of the experiment for grouping metrics
            config: Configuration dictionary for the experiment
        """
        self.db_path = db_path
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = config or {}
        
        # Initialize database connection
        self.conn = duckdb.connect(db_path)
        self._init_tables()
        
        # Store experiment config
        if config:
            self.log_config(config)
    
    def _init_tables(self):
        """Initialize database tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                start_time TIMESTAMP,
                config JSON
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                experiment_id VARCHAR,
                step INTEGER,
                metric_name VARCHAR,
                metric_value DOUBLE,
                timestamp TIMESTAMP,
                PRIMARY KEY (experiment_id, step, metric_name)
            )
        """)
        
        # Insert experiment record
        self.conn.execute("""
            INSERT INTO experiments (experiment_id, start_time, config)
            VALUES (?, ?, ?)
            ON CONFLICT DO NOTHING
        """, [self.experiment_name, datetime.now(), json.dumps(self.config)])
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int, prefix: str = "train/"):
        """Log metrics for current step.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Training step number
            prefix: Prefix for metric names (default: "train/")
        """
        timestamp = datetime.now()
        
        # Prepare data for insertion
        rows = []
        for name, value in metrics.items():
            if isinstance(value, (int, float, np.number)):
                metric_name = f"{prefix}{name}" if not name.startswith(prefix) else name
                rows.append({
                    'experiment_id': self.experiment_name,
                    'step': step,
                    'metric_name': metric_name,
                    'metric_value': float(value),
                    'timestamp': timestamp
                })
        
        if rows:
            df = pd.DataFrame(rows)
            self.conn.execute("""
                INSERT INTO metrics 
                SELECT * FROM df
                ON CONFLICT DO UPDATE SET
                metric_value = EXCLUDED.metric_value,
                timestamp = EXCLUDED.timestamp
            """)
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.conn.execute("""
            UPDATE experiments 
            SET config = ?
            WHERE experiment_id = ?
        """, [json.dumps(config), self.experiment_name])
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of current experiment."""
        metrics_df = self.conn.execute("""
            SELECT 
                metric_name,
                MIN(metric_value) as min_value,
                MAX(metric_value) as max_value,
                AVG(metric_value) as mean_value,
                COUNT(*) as count
            FROM metrics
            WHERE experiment_id = ?
            GROUP BY metric_name
        """, [self.experiment_name]).fetchdf()
        
        summary = {
            'experiment_id': self.experiment_name,
            'total_steps': self.conn.execute("""
                SELECT MAX(step) FROM metrics WHERE experiment_id = ?
            """, [self.experiment_name]).fetchone()[0],
            'metrics_summary': {
                row['metric_name']: {
                    'min': row['min_value'],
                    'max': row['max_value'],
                    'mean': row['mean_value'],
                    'count': row['count']
                }
                for _, row in metrics_df.iterrows()
            }
        }
        return summary
    
    def export_to_csv(self, output_dir: str):
        """Export metrics to CSV file."""
        output_path = Path(output_dir) / f"{self.experiment_name}_metrics.csv"
        
        metrics_df = self.conn.execute("""
            SELECT step, metric_name, metric_value, timestamp
            FROM metrics
            WHERE experiment_id = ?
            ORDER BY step, metric_name
        """, [self.experiment_name]).fetchdf()
        
        metrics_df.to_csv(output_path, index=False)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

def setup_logging(
    name: str,
    db_path: Optional[str] = None,
    log_dir: str = "logs",
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    log_level: int = logging.INFO
) -> DuckDBLogger:
    """Set up logging with DuckDB for metrics tracking."""
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up file logging
    log_file = log_dir / f"{name}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Set up console logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatters and add them to the handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove any existing handlers
    logger.handlers.clear()
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Initialize DuckDB logger
    if db_path is None:
        db_path = str(log_dir / "training_metrics.duckdb")
    
    duckdb_logger = DuckDBLogger(
        db_path=db_path,
        experiment_name=experiment_name,
        config=config
    )
    
    logger.info(f"Initialized logging to {log_file}")
    logger.info(f"DuckDB metrics database: {db_path}")
    
    return duckdb_logger