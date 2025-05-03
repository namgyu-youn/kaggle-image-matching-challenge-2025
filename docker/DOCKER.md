# Docker Setup for Image Matching Challenge 2025

This document explains how to use Docker to run the Image Matching Challenge 2025 code.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/) (optional but recommended)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for GPU support)

## Setup

1. **Kaggle API Credentials**

   Make sure you have a valid `kaggle.json` file in the `docker/` directory. You can download this from your Kaggle account settings.

2. **Build the Docker Image**

   ```bash
   # From the repository root directory
   docker-compose build
   ```

## Running the Container

### Using Docker Compose (Recommended)

```bash
# From the repository root directory
docker-compose up
```

This will:
- Build the Docker image if it doesn't exist
- Mount the necessary volumes (data, checkpoints, logs)
- Run both the dataset setup and training scripts

### Using Docker Directly

```bash
# From the repository root directory
docker run --gpus all \
  -v $(pwd)/docker/kaggle.json:/app/kaggle.json \
  -v $(pwd)/docker/data.sh:/app/data.sh \
  -v $(pwd)/docker/train.sh:/app/train.sh \
  -v $(pwd)/docker/config.yml:/app/config.yml \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  image-matching-challenge
```

## Running Individual Scripts

If you want to run only specific scripts inside the container:

```bash
# Setup dataset only
docker-compose run --rm image-matching ./data.sh

# Training only (assuming dataset is already set up)
docker-compose run --rm image-matching ./train.sh
```

## Accessing TensorBoard

If you want to monitor training with TensorBoard, you can expose port 6006:

```bash
docker-compose run --rm -p 6006:6006 image-matching tensorboard --logdir=/app/logs --host=0.0.0.0
```

Then access TensorBoard at http://localhost:6006 in your browser.

## Troubleshooting

- **GPU Issues**: Make sure the NVIDIA Container Toolkit is properly installed and your GPU drivers are up to date.
- **Memory Issues**: Adjust the batch size in `docker/config.yml` if you encounter out-of-memory errors.
- **Kaggle API**: Ensure your `docker/kaggle.json` file has the correct permissions (600) and valid credentials.