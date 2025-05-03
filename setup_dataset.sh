#!/usr/bin/env bash

set -e  # Exit on error

# Default settings
DATA_DIR="./data"
KAGGLE_COMPETITION="image-matching-challenge-2025"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE_JSON="$SCRIPT_DIR/kaggle.json"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --help)
      cat << EOF
Dataset Builder for Image Matching Challenge 2025

Usage: $0 --data_dir PATH

Options:
  --data_dir    Path to output dataset (default: ./data)
  --help        Show this help message
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

# 1. Check Kaggle CLI
echo "[STEP] Checking Kaggle CLI..."
if ! command -v kaggle &> /dev/null; then
  echo "[ERROR] Kaggle CLI not found. Install with: pip install kaggle" >&2
  exit 1
fi

# 2. Check authentication
echo "[STEP] Verifying Kaggle credentials..."
if [[ ! -f "$KAGGLE_JSON" ]]; then
  echo "[ERROR] kaggle.json not found in script directory. Place credentials file in the same directory as this script." >&2
  exit 1
fi

# Create temporary Kaggle config directory if not exists
if [[ ! -d ~/.kaggle ]]; then
  mkdir -p ~/.kaggle
fi

# Copy the kaggle.json to the standard location temporarily
cp "$KAGGLE_JSON" ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 3. Prepare directories
echo "[STEP] Creating directories..."
mkdir -p "$DATA_DIR/train" "$DATA_DIR/test"

# 4. Download dataset
echo "[STEP] Downloading dataset to current directory..."
kaggle competitions download -c "$KAGGLE_COMPETITION" -p .

# 5. Extract the ZIP file
echo "[STEP] Extracting dataset..."
zip_file=$(find . -maxdepth 1 -name '*.zip' | head -n1)
unzip -o -q "$zip_file"

# 6. Organize train/test folders
echo "[STEP] Organizing data..."
if [[ -d "train" ]]; then
  cp -rn train/* "$DATA_DIR/train/"
  rm -r train
fi
if [[ -d "test" ]]; then
  cp -rn test/* "$DATA_DIR/test/"
  rm -r test
fi

# 7. Summary
echo "[STEP] Generating summary..."
train_count=$(find "$DATA_DIR/train" -type f | wc -l)
test_count=$(find "$DATA_DIR/test" -type f | wc -l)
total=$((train_count + test_count))

echo "Dataset summary:"
echo "  Total images: $total"
echo "  Train images: $train_count"
echo "  Test images:  $test_count"

# Done
echo "[SUCCESS] Dataset is ready at $DATA_DIR"