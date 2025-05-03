#!/bin/bash

# Kaggle Image Matching Challenge 2025 - Training script

set -e  # Early stopping

# Color definitions for better readability
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No color

# Function definitions
print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default settings
DATA_DIR="./data"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs"
NUM_WORKERS=4
MODEL_TYPE="advanced"  # Used only for information display

# Install prerequisites
print_step "Installing prerequisites..."

# Check if uv is installed
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &> /dev/null; then
    print_warning "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "uv has been installed."
else
    print_success "uv is already installed."
fi

# Create and activate virtual environment
if [ ! -d ".venv" ]; then
    print_step "Creating virtual environment..."
    uv venv
    print_success "Virtual environment created."
else
    print_warning "Virtual environment already exists."
fi

# Install dependencies using uv
print_step "Activating virtual environment..."
source .venv/bin/activate
uv pip install -e ".[dev,v2]"
print_success "Dependencies installed."

# Check data directory
print_step "Working with data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"
if [ ! -d "$DATA_DIR/train" ]; then
    print_warning "$DATA_DIR/train directory not found. Please ensure dataset is properly organized."
fi

# Set environment variables
print_step "Configuring environment variables..."
export PYTHONWARNINGS="ignore::UserWarning"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
print_success "Environment configured."

# Build checkpoint and log directory
print_step "Creating checkpoint and log directories..."
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
print_success "Directories created."

# Construct training command
print_step "Constructing training command..."
TRAIN_CMD="uv run python src/train.py --data_dir $DATA_DIR --checkpoint_dir $CHECKPOINT_DIR --log_dir $LOG_DIR --num_workers $NUM_WORKERS"

# Display configuration summary
print_step "Configuration Summary:"
echo "  - Model Type: $MODEL_TYPE (defined in config.yml)"
# Start training
print_step "Starting training..."
echo -e "${BLUE}Execution Command:${NC} $TRAIN_CMD"
echo ""

eval $TRAIN_CMD

echo ""
print_success "Training completed!"