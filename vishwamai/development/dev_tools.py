# VishwamAI/development/dev_tools.py
"""
Development and debugging tools for VishwamAI.
Includes utilities for debugging, logging, profiling, and automation.
"""

import torch
import jax.numpy as jnp
import logging
from datetime import datetime
import time
import psutil
import pandas as pd
import subprocess
import os

# --- Debugging Utilities ---

def check_tensor_shape(tensor, expected_shape, name="Tensor"):
    """
    Check if the tensor has the expected shape.
    
    Args:
        tensor: Input tensor (PyTorch or JAX).
        expected_shape (tuple): Expected shape of the tensor.
        name (str): Name of the tensor for error messages (default: "Tensor").
    
    Raises:
        TypeError: If tensor type is unsupported.
        ValueError: If shapes do not match.
    """
    if isinstance(tensor, torch.Tensor):
        actual_shape = tensor.shape
    elif isinstance(tensor, jnp.ndarray):
        actual_shape = tensor.shape
    else:
        raise TypeError(f"Unsupported tensor type for {name}")

    if actual_shape != expected_shape:
        raise ValueError(f"{name} expected shape {expected_shape}, but got {actual_shape}")

def check_data_type(tensor, expected_dtype, name="Tensor"):
    """
    Check if the tensor has the expected data type.
    
    Args:
        tensor: Input tensor (PyTorch or JAX).
        expected_dtype: Expected data type (e.g., torch.float32, jnp.float32).
        name (str): Name of the tensor for error messages (default: "Tensor").
    
    Raises:
        TypeError: If tensor type is unsupported.
        ValueError: If data types do not match.
    """
    if isinstance(tensor, torch.Tensor):
        actual_dtype = tensor.dtype
    elif isinstance(tensor, jnp.ndarray):
        actual_dtype = tensor.dtype
    else:
        raise TypeError(f"Unsupported tensor type for {name}")

    if actual_dtype != expected_dtype:
        raise ValueError(f"{name} expected dtype {expected_dtype}, but got {actual_dtype}")

def visualize_tensor(tensor, name="Tensor"):
    """
    Visualize the tensor by printing its shape, dtype, and sample values.
    
    Args:
        tensor: Input tensor (PyTorch or JAX).
        name (str): Name of the tensor for display (default: "Tensor").
    
    Raises:
        TypeError: If tensor type is unsupported.
    """
    if isinstance(tensor, torch.Tensor):
        print(f"{name} (Torch Tensor): shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"Sample values: {tensor.flatten()[:5]}...")
    elif isinstance(tensor, jnp.ndarray):
        print(f"{name} (JAX Array): shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"Sample values: {tensor.flatten()[:5]}...")
    else:
        raise TypeError(f"Unsupported tensor type for {name}")

# --- Logging Utilities ---

class Logger:
    """
    Logger class for logging messages with timestamps.
    Supports console output and optional file logging.
    
    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to log file. If None, logs to console only.
        level: Logging level (default: logging.INFO).
    """
    def __init__(self, name, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def info(self, message):
        """Log an info-level message."""
        self.logger.info(message)

    def debug(self, message):
        """Log a debug-level message."""
        self.logger.debug(message)

    def error(self, message):
        """Log an error-level message."""
        self.logger.error(message)

    def log_tensor_stats(self, tensor, name):
        """
        Log statistics of the tensor (mean, std, min, max).
        
        Args:
            tensor: Input tensor (PyTorch or JAX).
            name (str): Name of the tensor for logging.
        
        Raises:
            TypeError: If tensor type is unsupported.
        """
        if isinstance(tensor, torch.Tensor):
            mean = tensor.mean().item()
            std = tensor.std().item()
            min_val = tensor.min().item()
            max_val = tensor.max().item()
        elif isinstance(tensor, jnp.ndarray):
            mean = float(jnp.mean(tensor))
            std = float(jnp.std(tensor))
            min_val = float(jnp.min(tensor))
            max_val = float(jnp.max(tensor))
        else:
            raise TypeError(f"Unsupported tensor type for {name}")

        self.info(f"Tensor {name}: mean={mean:.4f}, std={std:.4f}, min={min_val:.4f}, max={max_val:.4f}")

# --- Profiling Utilities ---

class Timer:
    """
    Timer class to measure execution time of code blocks.
    
    Usage:
        with Timer("my_task"):
            # code to time
    
    Args:
        task_name (str): Name of the task being timed (default: "Task").
    """
    def __init__(self, task_name="Task"):
        self.task_name = task_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        print(f"{self.task_name} took {elapsed:.4f} seconds")

def profile_memory():
    """
    Profile current memory usage in MB.
    
    Returns:
        float: Memory usage in MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024**2
    print(f"Current memory usage: {memory_mb:.2f} MB")
    return memory_mb

# --- Automation Scripts ---

def preprocess_data(input_path, output_path, drop_na=True):
    """
    Preprocess data by loading from input_path, optionally dropping missing values,
    and saving to output_path.
    
    Args:
        input_path (str): Path to input CSV file.
        output_path (str): Path to save processed CSV file.
        drop_na (bool): Whether to drop rows with missing values (default: True).
    
    Raises:
        FileNotFoundError: If input file does not exist.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if drop_na:
        df = df.dropna()
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

def save_checkpoint(model, path, optimizer=None):
    """
    Save model (and optionally optimizer) checkpoint to path.
    
    Args:
        model: PyTorch model to save.
        path (str): Path to save checkpoint.
        optimizer: Optional optimizer to save (default: None).
    """
    checkpoint = {'model_state_dict': model.state_dict()}
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model, path, optimizer=None):
    """
    Load model (and optionally optimizer) checkpoint from path.
    
    Args:
        model: PyTorch model to load state into.
        path (str): Path to checkpoint file.
        optimizer: Optional optimizer to load state into (default: None).
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")

def setup_environment(requirements_file="requirements.txt"):
    """
    Set up the development environment by installing dependencies from requirements_file.
    
    Args:
        requirements_file (str): Path to requirements.txt (default: "requirements.txt").
    
    Raises:
        FileNotFoundError: If requirements file does not exist.
    """
    if not os.path.exists(requirements_file):
        raise FileNotFoundError(f"Requirements file not found: {requirements_file}")

    subprocess.run(["pip", "install", "-r", requirements_file], check=True)
    print("Development environment set up successfully")

# --- Example Usage ---

if __name__ == "__main__":
    # Initialize logger
    logger = Logger("dev_tools", log_file="dev_tools.log")

    # Test tensor utilities
    sample_tensor = torch.randn(5, 5)
    check_tensor_shape(sample_tensor, (5, 5), "sample_tensor")
    check_data_type(sample_tensor, torch.float32, "sample_tensor")
    visualize_tensor(sample_tensor, "sample_tensor")
    logger.log_tensor_stats(sample_tensor, "sample_tensor")

    # Test profiling
    with Timer("Sleep Task"):
        time.sleep(1)
    profile_memory()