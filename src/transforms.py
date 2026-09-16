"""Albumentations pipelines (albumentations >= 2.0).

Images are resized first (512 -> 256) so the remaining augmentations run on
smaller arrays. Masks automatically use nearest-neighbour interpolation.
Augmentation is applied to the training split only.
"""
from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A  # noqa: E402
import cv2  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15),
                 border_mode=cv2.BORDER_CONSTANT, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.RandomGamma(gamma_limit=(85, 115), p=0.3),
        A.ElasticTransform(alpha=20, sigma=5, p=0.15),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_preview_transforms(size: int = 256) -> A.Compose:
    """Training augmentations without normalisation, for visual inspection."""
    return A.Compose(get_train_transforms(size).transforms[:-1])
