# Model Training Guide

This guide explains how to use the image matching models in a RunPod (SSH) environment.

## Prerequisites

1. Make sure you have a instance with PyTorch and CUDA installed.
2. Clone this repository and Install `uv` (a faster alternative to pip and poetry):

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to your PATH if needed
export PATH="$HOME/.cargo/bin:$PATH"
```

## Usage

```bash
chmod +x setup_dataset.sh
chmod +x train.sh

# Build dataset
./setup_dataset.sh

# Train the model
./train.sh
```

Note. parser related to model's performance are placed in `config.yaml` and customized to NVIDIA L40s GPU