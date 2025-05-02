#!/usr/bin/env python
"""
Test script for the inference pipeline
"""

import os
import argparse
import torch
from src.inference import load_model_for_inference, create_inference_pipeline

def parse_args():
    parser = argparse.ArgumentParser(description='Test inference pipeline')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, default='advanced',
                        choices=['dino', 'loftr', 'superglue', 'advanced'],
                        help='Type of model')
    parser.add_argument('--image1', type=str, required=True, help='Path to first image')
    parser.add_argument('--image2', type=str, required=True, help='Path to second image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--output', type=str, default='visualization.png',
                        help='Path to save visualization')

    return parser.parse_args()

def main():
    args = parse_args()

    # Check if files exist
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return

    if not os.path.exists(args.image1):
        print(f"Error: Image 1 not found at {args.image1}")
        return

    if not os.path.exists(args.image2):
        print(f"Error: Image 2 not found at {args.image2}")
        return

    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Load model
        print(f"Loading model of type {args.model_type} from {args.model_path}...")
        model = load_model_for_inference(
            args.model_path,
            model_type=args.model_type,
            device=device
        )
        print("Model loaded successfully")

        # Create inference pipeline
        config = {
            'device': device,
            'batch_size': args.batch_size,
            'use_tta': False  # Set to True to enable test-time augmentation
        }

        pipeline = create_inference_pipeline(
            model_path=args.model_path,
            model_type=args.model_type,
            config=config
        )
        print("Inference pipeline created successfully")

        # Predict similarity
        similarity = pipeline.predict_similarity(args.image1, args.image2)
        print(f"Similarity between images: {similarity:.4f}")

        # Extract features
        features1 = pipeline.extract_features(args.image1)
        features2 = pipeline.extract_features(args.image2)
        print(f"Feature shapes: {features1.shape}, {features2.shape}")

        # Create visualization
        print(f"Creating visualization at {args.output}...")
        pipeline.visualize_similarity(args.image1, args.image2, save_path=args.output)
        print(f"Visualization saved to {args.output}")

        print("All inference operations completed successfully!")

    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
