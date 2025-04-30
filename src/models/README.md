# Model Implementations

This directory contains various model implementations for the Image Matching Challenge 2025.

## Models

- `dino.py`: DINOv2 feature extractor for image matching
- `loftr.py`: LoFTR-style feature matcher with transformer architecture
- `superglue.py`: SuperGlue-style graph neural network for feature matching
- `utils.py`: Utility classes for models, such as positional encoding

## Usage

These models can be used by importing them from the `src.models` module:

```python
from src.models import DINOv2FeatureExtractor, LoFTRFeatureMatcher, SuperGlueMatchingModule
```

You can select the model type when running `train.py` using the `--model_type` argument:

```bash
python src/train.py --model_type dino
python src/train.py --model_type loftr
python src/train.py --model_type superglue
```

## Model Details

### DINOv2FeatureExtractor

Uses Facebook's DINOv2 vision transformer as a backbone for feature extraction. It extracts features from image pairs and computes similarity scores.

### LoFTRFeatureMatcher

Implements a LoFTR-style feature matcher with self and cross-attention layers for coarse-to-fine matching.

### SuperGlueMatchingModule

Implements a SuperGlue-style graph neural network for feature matching using attentional graph neural networks.