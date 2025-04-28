"""Image Matching Challenge 2025 Solution - Advanced Version (V2)"""

from .models import (
    DINOv2FeatureExtractor,
    LoFTRFeatureMatcher,
    SuperGlueMatchingModule,
    AttentionalGNN,
    PositionEncodingSine,
)
from .trainer import (
    MixedPrecisionTrainer,
    HardNegativeMining,
    CurriculumLearning,
    EMAModel,
    AdversarialTraining,
    LabelSmoothing,
    WarmupCosineScheduler,
    get_advanced_optimizer,
)
from .augmentation import (
    RandomPerspective,
    RandomOcclusion,
    RandomLightingCondition,
    AdvancedImageAugmentation,
    PairImageAugmentation,
    get_augmentation_transform,
)
from .ensemble import (
    ensemble_predictions,
    weighted_clustering,
    ensemble_submissions,
    merge_clusters,
    consensus_clustering,
)
from .metric_learning import (
    ArcFaceLoss,
    CircleLoss,
    MultiSimilarityLoss,
    SupConLoss,
    TripletBatchHardLoss,
    ProxyNCALoss,
    get_metric_learning_loss,
)

__version__ = "0.2.0"

__all__ = [
    # Advanced Models
    "DINOv2FeatureExtractor",
    "LoFTRFeatureMatcher",
    "SuperGlueMatchingModule",
    "AttentionalGNN",
    "PositionEncodingSine",

    # Advanced Training
    "MixedPrecisionTrainer",
    "HardNegativeMining",
    "CurriculumLearning",
    "EMAModel",
    "AdversarialTraining",
    "LabelSmoothing",
    "WarmupCosineScheduler",
    "get_advanced_optimizer",

    # Augmentation
    "RandomPerspective",
    "RandomOcclusion",
    "RandomLightingCondition",
    "AdvancedImageAugmentation",
    "PairImageAugmentation",
    "get_augmentation_transform",

    # Ensemble
    "ensemble_predictions",
    "weighted_clustering",
    "ensemble_submissions",
    "merge_clusters",
    "consensus_clustering",

    # Metric Learning
    "ArcFaceLoss",
    "CircleLoss",
    "MultiSimilarityLoss",
    "SupConLoss",
    "TripletBatchHardLoss",
    "ProxyNCALoss",
    "get_metric_learning_loss",
]