#!/usr/bin/env bash

set -e  # Exit on error

# Default settings
DATA_DIR="./data"
KAGGLE_COMPETITION="kaggle"
KAGGLE_JSON="./kaggle.json"

# 1. Install and check Kaggle API
echo "[STEP] Installing/checking Kaggle API..."
pip install kaggle

if [[ ! -f "$KAGGLE_JSON" ]]; then
  echo "[ERROR] kaggle.json not found in script directory. Place credentials file in the same directory as this script." >&2
  exit 1
fi

# 3. Prepare directories
mkdir -p "$DATA_DIR/train" "$DATA_DIR/test"

# 4. Download dataset
echo "[STEP] Downloading dataset to current directory..."
kaggle competitions download -c "$KAGGLE_COMPETITION" -p .

# 5. Extract the ZIP file
echo "[STEP] Extracting dataset..."
zip_file=$(find . -maxdepth 1 -name '*.zip' | head -n1)
apt-get update
apt-get install zip unzip
unzip -o -q "$zip_file"

# 6. Organize train/test folders
if [[ -d "train" ]]; then
  cp -rn train/* "$DATA_DIR/train/"
  rm -r train
fi
if [[ -d "test" ]]; then
  cp -rn test/* "$DATA_DIR/test/"
  rm -r test
fi

# Done
echo "[SUCCESS] Dataset is ready at $DATA_DIR"