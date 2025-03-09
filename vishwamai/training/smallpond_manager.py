"""
Data management utilities using smallpond for distributed data processing.
"""

import os
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import numpy as np
import smallpond
from pathlib import Path

class SmallpondManager:
    """Manages distributed data processing using smallpond"""
    
    def __init__(self, 
                config: Dict[str, Any],
                cache_dir: Optional[str] = None):
        """
        Initialize smallpond manager.
        
        Args:
            config: Smallpond configuration
            cache_dir: Directory for caching intermediate results
        """
        self.config = config
        self.cache_dir = cache_dir or "/tmp/vishwamai/smallpond"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize smallpond session
        self.session = smallpond.init(
            num_executors=config.get('num_executors', 1),
            ray_address=config.get('ray_address'),
            bind_numa_node=config.get('bind_numa_node', True),
            data_root=self.cache_dir,
            platform=config.get('platform')
        )

    def load_dataset(self, 
                    data_path: Union[str, Path],
                    num_partitions: Optional[int] = None,
                    partition_col: Optional[str] = None) -> Any:
        """
        Load and partition dataset using smallpond.
        
        Args:
            data_path: Path to data file/directory
            num_partitions: Number of partitions (default: number of executors)
            partition_col: Column to partition by
            
        Returns:
            Partitioned smallpond DataFrame
        """
        data_path = str(Path(data_path).absolute())
        
        # Read data based on file format
        if data_path.endswith('.parquet'):
            df = self.session.read_parquet(data_path)
        elif data_path.endswith('.csv'):
            df = self.session.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
            
        # Partition data
        if partition_col:
            df = df.repartition(num_partitions or self.config.get('num_executors', 1), 
                              hash_by=partition_col)
        else:
            df = df.repartition(num_partitions or self.config.get('num_executors', 1))
            
        return df

    def save_dataset(self,
                    df: Any,
                    output_path: Union[str, Path],
                    write_mode: str = 'overwrite'):
        """
        Save dataset using smallpond.
        
        Args:
            df: Smallpond DataFrame
            output_path: Output path
            write_mode: Write mode ('overwrite' or 'append')
        """
        output_path = str(Path(output_path).absolute())
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save based on file format
        if output_path.endswith('.parquet'):
            df.write_parquet(output_path, mode=write_mode)
        elif output_path.endswith('.csv'):
            df.write_csv(output_path, mode=write_mode)
        else:
            raise ValueError(f"Unsupported output format: {output_path}")

    def process_dataset(self,
                       df: Any,
                       processing_fn: callable,
                       batch_size: Optional[int] = None) -> Any:
        """
        Process dataset using smallpond's distributed computing.
        
        Args:
            df: Input smallpond DataFrame
            processing_fn: Processing function to apply
            batch_size: Batch size for processing
            
        Returns:
            Processed smallpond DataFrame
        """
        if batch_size:
            return df.map_partitions(
                lambda partition: processing_fn(partition), 
                batch_size=batch_size
            )
        return df.map_partitions(processing_fn)

    def merge_datasets(self,
                      dfs: List[Any],
                      merge_cols: Optional[List[str]] = None) -> Any:
        """
        Merge multiple datasets using smallpond.
        
        Args:
            dfs: List of smallpond DataFrames
            merge_cols: Columns to merge on
            
        Returns:
            Merged smallpond DataFrame
        """
        if not dfs:
            raise ValueError("No datasets provided for merging")
            
        result = dfs[0]
        for df in dfs[1:]:
            if merge_cols:
                result = result.merge(df, on=merge_cols)
            else:
                result = result.concat(df)
                
        return result

    def cleanup(self):
        """Cleanup smallpond session and resources"""
        if hasattr(self, 'session'):
            self.session.shutdown()