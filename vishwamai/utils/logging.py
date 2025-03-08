"""
Logging utilities for VishwamAI using Aim for experiment tracking.
Provides a centralized interface for logging metrics, artifacts, and experiment metadata.
"""

import os
import torch
import logging
from typing import Dict, Any, Optional, Union, List
from aim import Run
from aim.sdk.objects.image import Image as AimImage
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class AimLogger:
    """
    Wrapper for Aim experiment tracking.
    Handles metrics logging, artifact storage, and experiment metadata.
    """
    
    def __init__(
        self,
        experiment_name: str,
        repo: Optional[str] = None,
        hparams: Optional[Dict[str, Any]] = None,
        log_system_params: bool = True
    ):
        """
        Initialize Aim logger.
        
        Args:
            experiment_name: Name of the experiment
            repo: Path to Aim repository. If None, uses default
            hparams: Hyperparameters to log
            log_system_params: Whether to log system parameters
        """
        self.run = Run(
            experiment=experiment_name,
            repo=repo,
            run_hash=datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        
        # Log hyperparameters
        if hparams:
            self.run["hparams"] = hparams
            
        # Log system info if requested
        if log_system_params:
            self._log_system_info()
            
    def _log_system_info(self):
        """Log system information and hardware details"""
        system_info = {
            "python_version": os.sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0)
            })
            
        self.run["system"] = system_info
        
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        context: str = "training"
    ):
        """
        Log metrics to Aim.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current training step
            epoch: Current epoch
            context: Context for the metrics (e.g., "training", "validation")
        """
        for name, value in metrics.items():
            self.run.track(
                value,
                name=name,
                step=step,
                epoch=epoch,
                context={"subset": context}
            )
            
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple):
        """
        Log model architecture graph.
        
        Args:
            model: PyTorch model
            input_shape: Shape of input tensor for graph visualization
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            import tempfile
            import shutil
            
            # Create temporary directory for graph
            tmp_dir = tempfile.mkdtemp()
            writer = SummaryWriter(tmp_dir)
            
            # Create dummy input
            device = next(model.parameters()).device
            dummy_input = torch.zeros(input_shape, device=device)
            
            # Add graph to tensorboard
            writer.add_graph(model, dummy_input)
            writer.close()
            
            # Log the graph file as an artifact
            self.run["model/graph"] = os.path.join(tmp_dir, "events.out.tfevents.*")
            
            # Cleanup
            shutil.rmtree(tmp_dir)
        except Exception as e:
            logger.warning(f"Failed to log model graph: {str(e)}")
            
    def log_model_weights(self, model: torch.nn.Module, step: Optional[int] = None):
        """
        Log model weight distributions and gradients.
        
        Args:
            model: PyTorch model
            step: Current training step
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log weight distributions
                self.run.track(
                    param.data.cpu().numpy(),
                    name=f"weights/{name}",
                    step=step
                )
                
                # Log gradients if available
                if param.grad is not None:
                    self.run.track(
                        param.grad.cpu().numpy(),
                        name=f"gradients/{name}",
                        step=step
                    )
                    
    def log_image(
        self,
        image: Union[np.ndarray, torch.Tensor],
        name: str,
        step: Optional[int] = None,
        caption: Optional[str] = None
    ):
        """
        Log images to Aim.
        
        Args:
            image: Image data as numpy array or PyTorch tensor
            name: Name for the image
            step: Current training step
            caption: Optional caption for the image
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
            
        aim_image = AimImage(image, caption=caption)
        self.run.track(aim_image, name=name, step=step)
        
    def log_text(
        self,
        text: str,
        name: str,
        step: Optional[int] = None,
        context: Optional[str] = None
    ):
        """
        Log text data to Aim.
        
        Args:
            text: Text to log
            name: Name for the text entry
            step: Current training step
            context: Optional context string
        """
        self.run.track(
            text,
            name=name,
            step=step,
            context={"subset": context} if context else None
        )
        
    def log_confusion_matrix(
        self,
        matrix: np.ndarray,
        classes: List[str],
        name: str,
        step: Optional[int] = None
    ):
        """
        Log confusion matrix.
        
        Args:
            matrix: Confusion matrix as numpy array
            classes: List of class names
            name: Name for the confusion matrix
            step: Current training step
        """
        self.run.track(
            {
                "matrix": matrix,
                "classes": classes
            },
            name=name,
            step=step
        )
        
    def set_tags(self, tags: List[str]):
        """
        Set tags for the current run.
        
        Args:
            tags: List of tags to add
        """
        self.run.add_tags(tags)
        
    def log_artifact(self, local_path: str, name: str):
        """
        Log an artifact file.
        
        Args:
            local_path: Path to the artifact file
            name: Name for the artifact
        """
        self.run["artifacts"][name] = local_path
        
    def close(self):
        """Close the Aim run"""
        self.run.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()