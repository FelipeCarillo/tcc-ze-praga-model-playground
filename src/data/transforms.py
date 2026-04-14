"""Pipelines de augmentation com albumentations."""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int) -> A.Compose:
    """Pipeline de augmentation para o split de treino."""
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int) -> A.Compose:
    """Pipeline determinístico para val e test."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
