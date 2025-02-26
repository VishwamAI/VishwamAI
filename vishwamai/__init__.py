"""VishwamAI package."""

# First import loss functions to prevent circular imports
from .loss_functions import (
    cross_entropy_loss,
    kl_divergence_loss,
    tot_guided_loss,
    compute_metrics
)

# Core components
from .model import VishwamAIModel, ModelConfig
from .tokenizer import VishwamAITokenizer
from .transformer import VishwamAIModel, VisionTransformer10B

# Import error correction components - fixing import location
from .error_correction import (
    ErrorCorrectionModule,
    compute_error_metrics
)


# Break circular imports by carefully organizing the import order
from .distillation import VishwamaiGuruKnowledge, VishwamaiShaalaTrainer
from .tot import TreeOfThoughts, Thought, SearchState
from .data_utils import create_train_dataloader, create_val_dataloader
from .integration import ToTIntegrationLayer, MixtureDensityNetwork, MultiLevelToTAttention

# Import training components last to avoid circular imports
from .training import train, train_step, eval_step

__all__ = [
    # Loss functions
    "cross_entropy_loss",
    "kl_divergence_loss",
    "tot_guided_loss",
    "compute_metrics",
    
    # Core components
    "VishwamAIModel",
    "ModelConfig",
    "VishwamAITokenizer",
    "VisionTransformer10B",
    
    # Error correction components
    "ErrorCorrectionModule",
    "ErrorCorrectionTrainer",
    "compute_error_metrics",
    
    # Other modules
    "TreeOfThoughts",
    "Thought",
    "SearchState",
    "create_train_dataloader",
    "create_val_dataloader",
    "ToTIntegrationLayer",
    "MixtureDensityNetwork",
    "MultiLevelToTAttention",
    "VishwamaiGuruKnowledge",
    "VishwamaiShaalaTrainer",
    "train",
    "train_step",
    "eval_step"
]

__version__ = "0.1.0"
