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
    # Affine: combines three transforms into one — efficient and prevents position-sensitivity
    A.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        scale=(0.9, 1.1),
        rotate=(-10, 10),
        p=0.6
    ),
    # ElasticTransform: small organic deformations that mimic natural variation
    A.ElasticTransform(alpha=24, sigma=4, p=0.3),

    # ── Colour ────────────────────────────────────────────────────────────────
    # Fine-grained HSV jitter — intentionally mild to preserve artist palettes.
    A.HueSaturationValue(
        hue_shift_limit=7, sat_shift_limit=15, val_shift_limit=15, p=0.4
    ),

    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),

    # CLAHE: Normalises local contrast tile-by-tile
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

    # ── Dropout ───────────────────────────────────────────────────────────────
    # CoarseDropout (CutOut): erases rectangular regions with the dataset mean.
    A.CoarseDropout(
        num_holes_range=(2, 8), 
        hole_height_range=(16, 48), 
        hole_width_range=(16, 48),
        fill= "random",
        p=0.3
    ),
])

# Val/test: no augmentation at all
eval_transform = A.Compose([])

# ── Bridge function ───────────────────────────────────────────────────────────
def augment_image(image: np.ndarray) -> np.ndarray:
    """Apply training augmentation to a single HWC uint8 image."""
    # 1. Normalize to uint8 safely
    if image.dtype != np.uint8:
        # Check if the float image is [0, 1] or [0, 255]
        if np.max(image) <= 1.01: # Small epsilon for float precision
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
            
    # 2. Apply Albumentations (works best on uint8)
    augmented = train_transform(image=image)["image"]
    
    # 3. Return as float32 in range [0, 1] for the Model
    return augmented.astype(np.float32) / 255.0

def tf_augment(image, label):
    """Wrap augment_image for use inside tf.data.Dataset.map()."""
    aug_image = tf.numpy_function(
        func=augment_image,
        inp=[image],
        Tout=tf.float32,
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