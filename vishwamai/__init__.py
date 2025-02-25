"""VishwamAI package."""

# Core components
from .model import VishwamAIModel, ModelConfig
from .tokenizer import VishwamAITokenizer
from .transformer import VishwamAIModel, VisionTransformer10B

# Break circular imports by carefully organizing the import order
from .tot import TreeOfThoughts, Thought, SearchState
from .data_utils import create_train_dataloader, create_val_dataloader
from .integration import ToTIntegrationLayer, MixtureDensityNetwork, MultiLevelToTAttention
from .error_correction import ErrorCorrectionModule, compute_error_metrics
from .distillation import VishwamaiGuruKnowledge, VishwamaiShaalaTrainer
from .training import train, train_step, eval_step

# Import loss functions to make them available throughout the package
from .loss_functions import (
    cross_entropy_loss,
    kl_divergence_loss,
    tot_guided_loss,
    compute_metrics
)

__all__ = [
    "VishwamAIModel",
    "ModelConfig",
    "VishwamAITokenizer",
    "VisionTransformer10B",
    "TreeOfThoughts",
    "Thought",
    "SearchState",
    "create_train_dataloader",
    "create_val_dataloader",
    "ToTIntegrationLayer",
    "MixtureDensityNetwork",
    "MultiLevelToTAttention",
    "ErrorCorrectionModule",
    "compute_error_metrics",
    "VishwamaiGuruKnowledge",
    "VishwamaiShaalaTrainer",
    "train",
    "train_step",
    "eval_step"
]
__version__ = "0.1.0"
