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
DATA_DIR=""
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs"
NUM_WORKERS=4
RESUME=""

# Additional information storage variable (not supported by train.py)
MODEL_TYPE="dino"  # For information display only

# Command line argument processing
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"  # Stored only for information display
            shift 2
            ;;
        --feature_dim)
            print_warning "--feature_dim option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --batch_size)
            print_warning "--batch_size option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --epochs)
            print_warning "--epochs option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --learning_rate)
            print_warning "--learning_rate option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --warmup_epochs)
            print_warning "--warmup_epochs option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --use_mixed_precision)
            print_warning "--use_mixed_precision option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --use_ema)
            print_warning "--use_ema option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --ema_decay)
            print_warning "--ema_decay option is not supported by train.py and will be ignored"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --help)
            echo -e "${BLUE}Image Matching Challenge 2025 - Training Script${NC}"
            echo "Usage: $0 [options]"
            echo -e "\n${YELLOW}Required Options:${NC}"
            echo "  --data_dir PATH         Path to dataset directory (required)"
            echo -e "\n${YELLOW}Supported Options:${NC}"
            echo "  --checkpoint_dir DIR    Checkpoint save directory [default: checkpoints]"
            echo "  --log_dir DIR           Log save directory [default: logs]"
            echo "  --num_workers NUM       Number of data loader workers [default: 4]"
            echo "  --resume PATH           Checkpoint path to resume training"
            echo "  --help                  Display this help message"
            echo -e "\n${YELLOW}Information Display Options (not supported by train.py):${NC}"
            echo "  --model_type TYPE       Model type (used only for information display)"
            echo -e "${RED}Note:${NC} Only the mentioned supported options will be passed to train.py."
            exit 0
            ;;
        *)
            print_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$DATA_DIR" ]; then
    print_error "--data_dir argument is required. See --help for usage."
    exit 1
fi

# Check working directory
if [ ! -f "src/train.py" ]; then
    print_error "src/train.py not found. Run this script from the project root directory."
    exit 1
fi

# 1. Install prerequisites
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

# 2. Check data directory
print_step "Checking data directory: $DATA_DIR"
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi
print_success "Data directory exists."

# Set environment variables to ignore xFormers warnings
print_step "Configuring environment variables..."
export PYTHONWARNINGS="ignore::UserWarning"
print_success "Environment configured to ignore xFormers warnings."

# Build checkpoint and log directory
print_step "Creating checkpoint and log directories..."
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
print_success "Directories created."

# 3. Check and analyze directory structure
print_step "Analyzing directory structure..."
if [ -d "$DATA_DIR/train" ]; then
    print_success "Confirmed: $DATA_DIR/train directory exists"
else
    print_warning "$DATA_DIR/train directory does not exist. Please verify data structure."
fi

# 4. Adjust data path
DATA_PATH="$DATA_DIR"
if [[ "${DATA_DIR##*/}" == "train" ]]; then
    # Use parent directory if ending with /train
    DATA_PATH="$(dirname "$DATA_DIR")"
    print_warning "Data path ends with '/train'. Using parent directory: $DATA_PATH"
fi

# 5. Construct training command using array (include only parameters supported by train.py)
print_step "Constructing training command..."
TRAIN_CMD=(
    uv run python src/train.py
    --data_dir "$DATA_PATH"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --log_dir "$LOG_DIR"
    --num_workers "$NUM_WORKERS"
)

# Resume for checkpoint
if [ ! -z "$RESUME" ]; then
    TRAIN_CMD+=(--resume "$RESUME")
fi

# 6. Display configuration summary
print_step "Configuration Summary:"
echo "  - Data Directory: $DATA_PATH"
echo "  - Model Type: $MODEL_TYPE (for information display, not passed to train.py)"
echo "  - Checkpoint Directory: $CHECKPOINT_DIR"
echo "  - Log Directory: $LOG_DIR"
echo "  - Number of Workers: $NUM_WORKERS"
if [ ! -z "$RESUME" ]; then
    echo "  - Resume Training from Checkpoint: $RESUME"
fi

# 7. Start training
print_step "Starting training..."
echo -e "${BLUE}Execution Command:${NC} ${TRAIN_CMD[*]}"
echo ""

# Set environment variable to prevent autocast errors
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

"${TRAIN_CMD[@]}"

# Script completion
echo ""
echo -e "${GREEN}[COMPLETED]${NC} Script execution finished."