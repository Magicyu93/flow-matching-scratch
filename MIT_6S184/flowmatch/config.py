"""Shared constants and default configuration."""
import os
from typing import Optional

# Label index reserved for the unconditional ("null") class used by classifier-free guidance.
# ImageNet labels are 0..999, so the null token is 1000.
NULL_LABEL = 1000
NUM_LABEL_EMBEDDINGS = NULL_LABEL + 1

IMAGE_SHAPE = (3, 64, 64)

IMAGENET64_HF_NAME = "benjamin-paine/imagenet-1k-64x64"

# UNet configuration used for all ImageNet-64 runs (checkpoints depend on this).
MODEL_CONFIG = dict(
    channels=[128, 256, 512, 512],
    num_residual_layers=4,
    t_embed_dim=256,
    y_embed_dim=256,
    in_channels=3,
    attention_levels=[-1, -2, -3],  # attention at the 3 deepest levels
)


def default_cache_dir() -> Optional[str]:
    """HuggingFace cache dir for ImageNet-64. Set IMAGENET64_CACHE_DIR to override; None uses the HF default."""
    return os.environ.get("IMAGENET64_CACHE_DIR")
