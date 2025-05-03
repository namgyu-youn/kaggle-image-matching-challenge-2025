import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from collections import defaultdict


class ImageMatchingDataset(Dataset):
    """Dataset for Image Matching Challenge 2025"""

    def __init__(self, root_dir, mode='train', transform=None):
        self.root_dir = Path(root_dir).absolute()  # Make path absolute
        self.mode = mode
        self.transform = transform or self._get_default_transform()

        # Check if the directory exists
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        # Collect all images with their dataset and scene information
        self.images = []
        self.image_paths = []
        self.dataset_ids = []
        self.scene_ids = []

        if mode == 'train':
            self._load_training_data()
        else:
            self._load_test_data()

    def _get_default_transform(self):
        """Default transform pipeline"""
        return transforms.Compose([
            transforms.Resize((1024, 1024)),  # Standard resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_training_data(self):
        """Load training data with ground truth"""
        train_dir = self.root_dir / 'train'

        # Check if train directory exists
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")

        for dataset_dir in train_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_id = dataset_dir.name

            # Handle outliers
            outlier_dir = dataset_dir / 'outliers'
            if outlier_dir.exists():
                for img_path in outlier_dir.glob('*.png'):
                    self.image_paths.append(str(img_path))
                    self.dataset_ids.append(dataset_id)
                    self.scene_ids.append('outliers')

            # Handle scene directories (images directly in dataset directory)
            for img_path in dataset_dir.glob('*.png'):
                self.image_paths.append(str(img_path))
                self.dataset_ids.append(dataset_id)
                self.scene_ids.append('scene1')  # Default scene name

            # Handle subdirectories as scenes
            for scene_dir in dataset_dir.iterdir():
                if not scene_dir.is_dir() or scene_dir.name == 'outliers':
                    continue

                scene_id = scene_dir.name

                for img_path in scene_dir.glob('*.png'):
                    self.image_paths.append(str(img_path))
                    self.dataset_ids.append(dataset_id)
                    self.scene_ids.append(scene_id)

    def _load_test_data(self):
        """Load test data"""
        test_dir = self.root_dir / 'test'

        for dataset_dir in test_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_id = dataset_dir.name

            for img_path in dataset_dir.glob('*.png'):
                self.image_paths.append(str(img_path))
                self.dataset_ids.append(dataset_id)
                self.scene_ids.append(None)  # No scene info for test data

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Get a single item"""
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        item = {
            'image': image,
            'path': img_path,
            'filename': Path(img_path).name,
            'dataset_id': self.dataset_ids[idx],
            'scene_id': self.scene_ids[idx],
            'index': idx
        }

        return item


class DatasetPairs(Dataset):
    """Dataset for training on image pairs"""

    def __init__(self, base_dataset, negative_ratio=0.5):
        self.base_dataset = base_dataset
        self.negative_ratio = negative_ratio

        # Create positive and negative pairs
        self.pairs = []
        self.labels = []

        # Group images by dataset and scene
        self.dataset_scene_images = defaultdict(lambda: defaultdict(list))

        for idx in range(len(base_dataset)):
            item = base_dataset.dataset_ids[idx]
            scene = base_dataset.scene_ids[idx]

            if scene and scene != 'outliers':
                self.dataset_scene_images[item][scene].append(idx)

        self._create_pairs()

    def _create_pairs(self):
        """Create positive and negative pairs"""
        # Positive pairs (same scene)
        for dataset_id, scenes in self.dataset_scene_images.items():
            for scene_id, images in scenes.items():
                if len(images) > 1:
                    for i in range(len(images)):
                        for j in range(i + 1, len(images)):
                            self.pairs.append((images[i], images[j]))
                            self.labels.append(1)

        # Negative pairs (different scenes)
        num_positive = len(self.pairs)
        num_negative = int(num_positive * self.negative_ratio)

        all_dataset_ids = list(self.dataset_scene_images.keys())

        for _ in range(num_negative):
            dataset_id = np.random.choice(all_dataset_ids)
            scenes = list(self.dataset_scene_images[dataset_id].keys())

            if len(scenes) > 1:
                scene1, scene2 = np.random.choice(scenes, 2, replace=False)
                img1_idx = np.random.choice(self.dataset_scene_images[dataset_id][scene1])
                img2_idx = np.random.choice(self.dataset_scene_images[dataset_id][scene2])

                self.pairs.append((img1_idx, img2_idx))
                self.labels.append(0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """Get a pair of images"""
        img1_idx, img2_idx = self.pairs[idx]

        item1 = self.base_dataset[img1_idx]
        item2 = self.base_dataset[img2_idx]

        return {
            'image1': item1['image'],
            'image2': item2['image'],
            'path1': item1['path'],
            'path2': item2['path'],
            'label': self.labels[idx],
            'dataset_id': item1['dataset_id']
        }


def get_dataloaders(data_dir, batch_size=32, num_workers=4, transform=None):
    """Create dataloaders for training and validation"""
    # Training dataset
    train_dataset = ImageMatchingDataset(data_dir, mode='train', transform=transform)
    train_pairs = DatasetPairs(train_dataset)

    # Create split for validation
    total_size = len(train_pairs)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_pairs, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader