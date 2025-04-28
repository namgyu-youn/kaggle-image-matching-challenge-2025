import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
import cv2


class RandomPerspective:
    """Apply random perspective transformation"""

    def __init__(self, distortion_scale=0.2, p=0.5):
        self.distortion_scale = distortion_scale
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size

            # Random perspective points
            start_points = np.float32([[0, 0], [width - 1, 0],
                                     [width - 1, height - 1], [0, height - 1]])

            # Random distortion
            end_points = np.float32([
                [random.uniform(0, width * self.distortion_scale),
                 random.uniform(0, height * self.distortion_scale)],
                [random.uniform(width * (1 - self.distortion_scale), width - 1),
                 random.uniform(0, height * self.distortion_scale)],
                [random.uniform(width * (1 - self.distortion_scale), width - 1),
                 random.uniform(height * (1 - self.distortion_scale), height - 1)],
                [random.uniform(0, width * self.distortion_scale),
                 random.uniform(height * (1 - self.distortion_scale), height - 1)]
            ])

            matrix = cv2.getPerspectiveTransform(start_points, end_points)
            img_np = np.array(img)
            warped = cv2.warpPerspective(img_np, matrix, (width, height))

            return Image.fromarray(warped)
        return img


class RandomOcclusion:
    """Randomly occlude parts of the image"""

    def __init__(self, p=0.5, max_occlusion=0.3, min_occlusion=0.1):
        self.p = p
        self.max_occlusion = max_occlusion
        self.min_occlusion = min_occlusion

    def __call__(self, img):
        if random.random() < self.p:
            width, height = img.size

            # Random occlusion size
            occlusion_ratio = random.uniform(self.min_occlusion, self.max_occlusion)
            occ_width = int(width * occlusion_ratio)
            occ_height = int(height * occlusion_ratio)

            # Random position
            x = random.randint(0, width - occ_width)
            y = random.randint(0, height - occ_height)

            # Apply occlusion (black rectangle)
            img_occluded = img.copy()
            img_occluded = TF.erase(img_occluded, y, x, occ_height, occ_width, 0)

            return img_occluded
        return img


class RandomLightingCondition:
    """Simulate different lighting conditions"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            operation = random.choice(['brightness', 'contrast', 'saturation', 'gamma'])

            if operation == 'brightness':
                factor = random.uniform(0.5, 1.5)
                img = TF.adjust_brightness(img, factor)
            elif operation == 'contrast':
                factor = random.uniform(0.5, 1.5)
                img = TF.adjust_contrast(img, factor)
            elif operation == 'saturation':
                factor = random.uniform(0.5, 1.5)
                img = TF.adjust_saturation(img, factor)
            elif operation == 'gamma':
                gamma = random.uniform(0.7, 1.3)
                img = TF.adjust_gamma(img, gamma)

        return img


class AdvancedImageAugmentation:
    """Advanced image augmentation for training"""

    def __init__(self, image_size=(480, 640), strong_aug=False):
        self.image_size = image_size
        self.strong_aug = strong_aug

        # Basic augmentations
        basic_transforms = [
            T.Resize(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ]

        # Strong augmentations
        if strong_aug:
            strong_transforms = [
                RandomPerspective(distortion_scale=0.2, p=0.3),
                RandomOcclusion(p=0.2),
                RandomLightingCondition(p=0.3),
                T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.2),
                T.RandomApply([T.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.3)
            ]
            self.transform = T.Compose(basic_transforms + strong_transforms + [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose(basic_transforms + [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, img):
        return self.transform(img)


class PairImageAugmentation:
    """Augmentation for image pairs that maintains geometric consistency"""

    def __init__(self, image_size=(480, 640)):
        self.image_size = image_size
        self.resize = T.Resize(image_size)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    def __call__(self, img1, img2):
        # Resize both images
        img1 = self.resize(img1)
        img2 = self.resize(img2)

        # Apply same geometric transformations to both images
        if random.random() < 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        # Random rotation (same angle for both)
        angle = random.uniform(-10, 10)
        img1 = TF.rotate(img1, angle)
        img2 = TF.rotate(img2, angle)

        # Different color jitter for each image (more realistic)
        color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.1)
        img1 = color_jitter(img1)
        img2 = color_jitter(img2)

        # Convert to tensor and normalize
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)

        return img1, img2


def get_augmentation_transform(mode='train', strong_aug=False):
    """Get augmentation transform based on mode"""
    if mode == 'train':
        return AdvancedImageAugmentation(strong_aug=strong_aug)
    else:
        return T.Compose([
            T.Resize((480, 640)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])