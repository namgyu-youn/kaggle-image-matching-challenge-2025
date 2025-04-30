# Using Image Matching Models in RunPod

This guide explains how to use the image matching models in a RunPod environment.

## Prerequisites

1. Make sure you have a RunPod instance with PyTorch and CUDA installed.
2. Clone this repository to your RunPod instance.
3. Install `uv` (a faster alternative to pip):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to your PATH if needed
export PATH="$HOME/.cargo/bin:$PATH"
```

4. Create a virtual environment and install dependencies:

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the package and its dependencies
uv pip install -e ".[dev,v2]"
```

## Data Preparation

download kaggle dataset using the following command:

```bash
kaggle competitions download -c image-matching-challenge-2025
```

## Training a Model

You can train different types of models using the `train.py` script:

### Basic Usage

```bash
uv run python src/train.py --data_dir /path/to/data --model_type dino
```

### Available Model Types

- `dino`: DINOv2 feature extractor (default)
- `loftr`: LoFTR-style feature matcher
- `superglue`: SuperGlue-style graph neural network
- `advanced`: Base ImageMatchingModel

### Training Configuration

You can customize the training process with various arguments:

```bash
python src/train.py \
  --data_dir /path/to/data \
  --model_type dino \
  --feature_dim 512 \
  --batch_size 8 \
  --epochs 100 \
  --learning_rate 1e-4 \
  --warmup_epochs 10 \
  --checkpoint_dir checkpoints \
  --log_dir logs
```

### Using Mixed Precision Training

To enable mixed precision training for faster training:

```bash
python src/train.py --data_dir /path/to/data --use_mixed_precision true
```

### Using EMA (Exponential Moving Average)

To enable EMA for more stable training:

```bash
python src/train.py --data_dir /path/to/data --use_ema true --ema_decay 0.999
```

## Resuming Training

To resume training from a checkpoint:

```bash
python src/train.py --data_dir /path/to/data --resume checkpoints/checkpoint_latest.pth
```

## Evaluation

To evaluate a trained model:

```bash
python src/evaluation.py \
  --model_path checkpoints/checkpoint_best.pth \
  --model_type dino \
  --data_dir /path/to/data \
  --output_dir evaluation_results
```

## Inference

To run inference on new images:

```bash
python src/inference.py \
  --model_path checkpoints/checkpoint_best.pth \
  --model_type dino \
  --image1 /path/to/image1.png \
  --image2 /path/to/image2.png
```

To process a whole dataset:

```bash
python src/inference.py \
  --model_path checkpoints/checkpoint_best.pth \
  --model_type dino \
  --dataset_dir /path/to/dataset \
  --output_dir inference_results
```

## RunPod-Specific Tips

1. Use persistent storage to save your models and data.
2. For large datasets, consider mounting external storage.
3. Monitor GPU memory usage with `nvidia-smi` to optimize batch size.
4. Use the RunPod file browser to easily upload and download files.
5. Consider using RunPod templates with pre-installed dependencies for faster setup.

## Example RunPod Workflow

1. Start a RunPod instance with a GPU (e.g., RTX 3090, A100).
2. Clone the repository:

   ```bash
   git clone https://github.com/your-username/image-matching-challenge-2025.git
   cd image-matching-challenge-2025
   ```

3. Install `uv` and set up the environment:

   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.cargo/bin:$PATH"

   # Create and activate virtual environment
   uv venv
   source .venv/bin/activate

   # Install dependencies
   uv pip install -e ".[dev,v2]"
   ```

4. Upload your dataset to RunPod storage.

5. Train a model:

   ```bash
   uv run python src/train.py \
     --data_dir /path/to/data \
     --model_type dino \
     --batch_size 8 \
     --epochs 100 \
     --learning_rate 1e-4 \
     --use_mixed_precision true \
     --use_ema true
   ```

6. Evaluate the model:

   ```bash
   uv run python src/evaluation.py \
     --model_path checkpoints/checkpoint_best.pth \
     --model_type dino \
     --data_dir /path/to/data
   ```

7. Run inference on new images:

   ```bash
   uv run python src/inference.py \
     --model_path checkpoints/checkpoint_best.pth \
     --model_type dino \
     --image1 /path/to/image1.png \
     --image2 /path/to/image2.png
   ```

8. Download the trained model and results from RunPod storage.
