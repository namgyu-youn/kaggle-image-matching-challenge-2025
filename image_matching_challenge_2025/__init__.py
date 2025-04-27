"""Image Matching Challenge 2025 Solution Package"""

from .dataset import ImageMatchingDataset, DatasetPairs, get_dataloaders
from .models import (
    FeatureExtractor,
    SimilarityNetwork,
    PoseEstimator,
    ImageMatchingModel,
    SuperPointDetector,
    AdvancedMatchingModel,
)
from .loss import (
    ContrastiveLoss,
    TripletLoss,
    PoseLoss,
    CombinedLoss,
    MetricLearningLoss,
    GeometricConsistencyLoss,
)
from .trainer import Trainer
from .inference import InferencePipeline

__version__ = "0.1.0"

__all__ = [
    "ImageMatchingDataset",
    "DatasetPairs",
    "get_dataloaders",
    "FeatureExtractor",
    "SimilarityNetwork",
    "PoseEstimator",
    "ImageMatchingModel",
    "SuperPointDetector",
    "AdvancedMatchingModel",
    "ContrastiveLoss",
    "TripletLoss",
    "PoseLoss",
    "CombinedLoss",
    "MetricLearningLoss",
    "GeometricConsistencyLoss",
    "Trainer",
    "InferencePipeline",
]