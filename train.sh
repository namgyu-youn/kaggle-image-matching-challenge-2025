#!/bin/bash

# Kaggle Imagge Matching Challenge 2025 - Training script

set -e  # Early stopping

# Color definitions
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
MODEL_TYPE="dino"
FEATURE_DIM=512
BATCH_SIZE=2
EPOCHS=100
LEARNING_RATE=0.0001
WARMUP_EPOCHS=10
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs"
USE_MIXED_PRECISION="false"
USE_EMA="false"
EMA_DECAY=0.999
NUM_WORKERS=4
RESUME=""

# Command line argument processing
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --model_type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --feature_dim)
            FEATURE_DIM="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --warmup_epochs)
            WARMUP_EPOCHS="$2"
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
            USE_MIXED_PRECISION="$2"
            shift 2
            ;;
        --use_ema)
            USE_EMA="$2"
            shift 2
            ;;
        --ema_decay)
            EMA_DECAY="$2"
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
            echo "Image Matching Challenge 2025 - 훈련 스크립트"
            echo "사용법: $0 [옵션]"
            echo "Options:"
            echo "  --data_dir PATH       Path to dataset directory (required)"
            echo "  --model_type TYPE            Model type (dino, loftr, superglue, advanced) [기본값: dino]"
            echo "  --feature_dim DIM            Feature dimension [default: 512]"
            echo "  --batch_size SIZE            Batch size [default: 8]"
            echo "  --epochs NUM                 Epochs [default: 100]"
            echo "  --learning_rate RATE         Learning rate [default: 0.0001]"
            echo "  --warmup_epochs NUM          Warm-up epochs [default: 10]"
            echo "  --checkpoint_dir DIR  Directory to save checkpoints [default: checkpoints]"
            echo "  --log_dir DIR         Directory to save logs [default: logs]"
            echo "  --use_mixed_precision BOOL   Mixed precision [default: false]"
            echo "  --use_ema BOOL               EMA 사용 [default: false]"
            echo "  --ema_decay RATE             EMA 감쇠율 [dfeault: 0.999]"
            echo "  --num_workers NUM            Workers for data loader [default: 4]"
            echo "  --resume PATH         Path to checkpoint to resume from"
            echo "  --help                Display this help message"
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
    print_error "--data_dir argument is required. Use --help for usage."
    exit 1
fi

# Check working directory
if [ ! -f "src/train.py" ]; then
    print_error "src/train.py not found. Run this script from the project's root directory."
    exit 1
fi

# 1. Install prerequisites
print_step "Installing prerequisites..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    print_warning "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    print_success "uv installed."
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
print_step "Setting environment variables..."
export PYTHONWARNINGS="ignore::UserWarning"
print_success "Environment configured to ignore xFormers warnings."

# Build checkpoint and log directory
print_step "Building checkpoint and log directory..."
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
print_success "Directory is generated."

# 3. Check and analyze directory structure
print_step "Analyzing directory structure..."
if [ -d "$DATA_DIR/train" ]; then
    print_success "Confirmed: $DATA_DIR/train directory exists"
else
    print_warning "$DATA_DIR/train directory does not exist. Check your data structure."
fi

# 4. 훈련 명령어 구성
# 데이터 경로 확인 로직
# dataset.py에서는 data_dir/train 경로를 찾기 때문에
# 입력된 경로가 이미 /train으로 끝나는지 확인하고 조정
DATA_PATH=$DATA_DIR
if [[ "$DATA_DIR" == *"/train" ]]; then
    # /train으로 끝나면 상위 디렉토리 사용
    DATA_PATH=$(dirname "$DATA_DIR")
    print_warning "데이터 경로가 '/train'으로 끝납니다. 상위 디렉토리를 사용합니다: $DATA_PATH"
fi

TRAIN_CMD="uv run python src/train.py"
TRAIN_CMD+=" --data_dir $DATA_PATH"
TRAIN_CMD+=" --model_type $MODEL_TYPE"
TRAIN_CMD+=" --feature_dim $FEATURE_DIM"
TRAIN_CMD+=" --batch_size $BATCH_SIZE"
TRAIN_CMD+=" --epochs $EPOCHS"
TRAIN_CMD+=" --learning_rate $LEARNING_RATE"
TRAIN_CMD+=" --warmup_epochs $WARMUP_EPOCHS"
TRAIN_CMD+=" --checkpoint_dir $CHECKPOINT_DIR"
TRAIN_CMD+=" --log_dir $LOG_DIR"
TRAIN_CMD+=" --use_mixed_precision $USE_MIXED_PRECISION"
TRAIN_CMD+=" --use_ema $USE_EMA"
TRAIN_CMD+=" --ema_decay $EMA_DECAY"
TRAIN_CMD+=" --num_workers $NUM_WORKERS"

# Resume for checkpoint
if [ ! -z "$RESUME" ]; then
    TRAIN_CMD+=" --resume $RESUME"
fi

# 5. Model Training
print_step "Following is the config setting:"
echo "  - Data directory: $DATA_PATH"
echo "  - Model type: $MODEL_TYPE"
echo "  - Feature dimension: $FEATURE_DIM"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Epochs : $EPOCHS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Warm up epochs: $WARMUP_EPOCHS"
echo "  - Mixed precision: $USE_MIXED_PRECISION"
echo "  - EMA: $USE_EMA"
if [ ! -z "$RESUME" ]; then
    echo "  - Checkpoint for resume: $RESUME"
fi

# 훈련 명령 실행
echo -e "${BLUE}실행 명령어:${NC} $TRAIN_CMD"
echo ""
eval $TRAIN_CMD

# 6. 훈련 완료
print_success "훈련이 완료되었습니다!"
echo "체크포인트는 '$CHECKPOINT_DIR' 디렉토리에 저장되었습니다."
echo "로그는 '$LOG_DIR' 디렉토리에서 확인할 수 있습니다."

# 스크립트 종료
echo ""
echo -e "${GREEN}[완료]${NC} 스크립트가 성공적으로 실행되었습니다."