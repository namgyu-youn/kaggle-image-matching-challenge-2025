"""Image Matching Challenge 2025 Solution Package"""

from .dataset import ImageMatchingDataset, DatasetPairs, get_dataloaders
from .models import (
    FeatureExtractor,
    SimilarityNetwork,
    PoseEstimator,
    ImageMatchingModel,
)
from .loss import (
    ContrastiveLoss,
    PoseLoss,
)
from .trainer import Trainer

__version__ = "0.1.0"

__all__ = [
    "ImageMatchingDataset",
    "DatasetPairs",
    "get_dataloaders",
    "FeatureExtractor",
    "SimilarityNetwork",
    "PoseEstimator",
    "ImageMatchingModel",
    "ContrastiveLoss",
    "PoseLoss",
    "Trainer",
]