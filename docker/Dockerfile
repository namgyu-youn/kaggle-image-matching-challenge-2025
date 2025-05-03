FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files (excluding docker directory)
COPY src/ /app/src/
COPY dataset/ /app/dataset/
COPY pyproject.toml README.md /app/

# Install Kaggle API (required by setup_dataset.sh)
RUN pip install kaggle

# Create directories needed by the scripts
RUN mkdir -p /app/data/train /app/data/test /app/checkpoints /app/logs

# Set up the environment (but don't download data)
# This runs the train.sh script up to the point of starting training
# to install all dependencies and set up the environment
RUN bash -c 'set -e; \
    # Install uv package manager (from train.sh) \
    export PATH="$HOME/.local/bin:$PATH"; \
    if ! command -v uv &> /dev/null; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh; \
        export PATH="$HOME/.cargo/bin:$PATH"; \
    fi; \
    # Create and activate virtual environment \
    uv venv .venv; \
    source .venv/bin/activate; \
    # Install dependencies \
    uv pip install -e ".[dev,v2]"; \
    # Set environment variables \
    export PYTHONWARNINGS="ignore::UserWarning"; \
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128";'

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:${PATH}"
ENV PYTHONWARNINGS="ignore::UserWarning"
ENV PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# Default command to run when container starts
# Note: This assumes data.sh and train.sh are mounted at runtime
CMD ["/bin/bash", "-c", "./data.sh && ./train.sh"]