"""
Albumentations augmentation pipeline

Why Albumentations over Keras built-in layers:
Keras's augmentation layers cover the basics. Albumentations adds transforms that directly address specific data issues:
- CoarseDropout: Fixes subject bias by randomly erasing patches, forcing the model to attend to brushstroke texture.
- ElasticTransform: Creates subtle, organic warping that mimics natural variation in brush handling.
- CLAHE: Normalises local contrast — useful since some artists mix high-contrast etchings with softer paintings.
- HueSaturationValue: Fine-grained colour jitter within safe bounds — preserves palette signatures.

Integration pattern:
Albumentations operates on numpy arrays (HWC uint8). tf.data operates on tensors. 
The bridge is tf.numpy_function — it calls a Python function element-wise inside the TF graph.
"""

import numpy as np
import tensorflow as tf
import albumentations as A

# ── Training augmentation ─────────────────────────────────────────────────────
# Each transform has a probability p — tuned to be meaningful but not aggressive.
# The ordering matters: geometric transforms first, then colour, then dropout.
train_transform = A.Compose([
    # ── Geometric ─────────────────────────────────────────────────────────────
    A.HorizontalFlip(p=0.5),
    # ShiftScaleRotate: combines three transforms into one — efficient and prevents position-sensitivity
    A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.1, rotate_limit=10, border_mode=0, p=0.6
    ),
    # ElasticTransform: small organic deformations that mimic natural variation
    A.ElasticTransform(alpha=60, sigma=6, p=0.3),

    # ── Colour ────────────────────────────────────────────────────────────────
    # Fine-grained HSV jitter — intentionally mild to preserve artist palettes.
    A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4
    ),
    # CLAHE: Normalises local contrast tile-by-tile
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),

    # ── Dropout ───────────────────────────────────────────────────────────────
    # CoarseDropout (CutOut): erases rectangular regions with the dataset mean.
    A.CoarseDropout(
        max_holes=8, max_height=48, max_width=48,
        min_holes=2, min_height=16, min_width=16,
        fill_value=[0.485 * 255, 0.456 * 255, 0.406 * 255],  # ImageNet mean
        p=0.4
    ),
])

# Val/test: no augmentation at all
eval_transform = A.Compose([])

# ── Bridge function ───────────────────────────────────────────────────────────
def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply training augmentation to a single HWC uint8 image."""
    return train_transform(image=image)["image"]

def tf_augment(image, label):
    """Wrap augment_image for use inside tf.data.Dataset.map()."""
    aug_image = tf.numpy_function(
        func=augment_image,
        inp=[image],
        Tout=tf.uint8,
    )
    aug_image.set_shape(image.shape)
    return aug_image, label

# ── Full pipeline builder ─────────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE

def build_alb_train_ds_v2(raw_ds, batch_size, cache_path: str):
    """
    Recommended: cache RAW images, augment AFTER cache.
    Every epoch sees freshly randomised augmentations, while disk reads only
    happen once (raw pixels are cached to SSD on the first epoch).
    """
    ds = raw_ds.cache(cache_path)          
    ds = ds.shuffle(50_000, seed=None)     
    ds = ds.map(tf_augment, num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

def build_eval_ds(raw_ds, batch_size):
    """Val/test: no augmentation, no shuffle."""
    return raw_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)