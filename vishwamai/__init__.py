"""Vishwamai: A Mixture-of-Experts Language Model with Multi-Level Attention."""

from importlib.metadata import version as _get_version

try:
    __version__ = _get_version(__name__)
except Exception:
    __version__ = "0.1.0"

# Import key components for easy access
from vishwamai.model.transformer.model import VishwamaiModel
from vishwamai.data.tokenization import SPTokenizer
from vishwamai.training import Trainer
from vishwamai.utils import (
    load_model,
    load_config,
    setup_logging,
    evaluate_model
)

# Import dataset utilities
from vishwamai.data.dataset import (
    BaseDataset,
    create_dataloaders
)

# Import model components
from vishwamai.model.moe import (
    ExpertLayer,
    Router,
    MoELayer
)

from vishwamai.model.mla import (
    MultiLevelAttention,
    MLABlock
)

# Import training utilities
from vishwamai.training.optimizer import (
    get_optimizer,
    get_scheduler
)

from vishwamai.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LRSchedulerCallback
)

# Set default logging level
from vishwamai.utils.logging import setup_default_logging
setup_default_logging()

__all__ = [
    # Version
    "__version__",
    
    # Core components
    "VishwamaiModel",
    "SPTokenizer",
    "Trainer",
    
    # Utils
    "load_model",
    "load_config",
    "setup_logging",
    "evaluate_model",
    
    # Dataset
    "BaseDataset",
    "create_dataloaders",
    
    # Model components
    "ExpertLayer",
    "Router",
    "MoELayer",
    "MultiLevelAttention",
    "MLABlock",
    
    # Training
    "get_optimizer",
    "get_scheduler",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LRSchedulerCallback",
]

# Package metadata
__author__ = "Vishwamai Research Team"
__email__ = "team@vishwamai.org"
__description__ = "A Mixture-of-Experts Language Model with Multi-Level Attention"
__url__ = "https://github.com/organization/vishwamai"
__license__ = "Apache License 2.0"

# Optional dependencies
OPTIONAL_DEPENDENCIES = {
    "tpu": [
        "torch-xla",
        "cloud-tpu-client",
    ],
    "training": [
        "wandb",
        "tensorboard",
        "accelerate",
    ],
    "serving": [
        "fastapi",
        "uvicorn",
        "pydantic",
    ],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
    ],
    "dev": [
        "pytest",
        "black",
        "isort",
        "flake8",
        "mypy",
    ],
}

def get_optional_dependencies():
    """Return dictionary of optional dependencies."""
    return OPTIONAL_DEPENDENCIES

def check_dependencies(group: str) -> bool:
    """Check if dependencies for a specific group are installed.
    
    Args:
        group: The dependency group to check (e.g., 'tpu', 'training')
        
    Returns:
        bool: True if all dependencies are installed
    """
    if group not in OPTIONAL_DEPENDENCIES:
        raise ValueError(f"Unknown dependency group: {group}")
        
    try:
        import pkg_resources
    except ImportError:
        return False
        
    deps = OPTIONAL_DEPENDENCIES[group]
    installed = {pkg.key for pkg in pkg_resources.working_set}
    required = {pkg.split('>=')[0] for pkg in deps}
    
    return all(pkg in installed for pkg in required)
