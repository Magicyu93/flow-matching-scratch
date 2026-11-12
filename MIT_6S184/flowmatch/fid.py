"""FID utilities built on pytorch-fid's InceptionV3."""
import os
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def get_inception(device: torch.device) -> torch.nn.Module:
    from pytorch_fid.inception import InceptionV3

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3([block_idx]).to(device).eval()


@torch.no_grad()
def get_activations(images: torch.Tensor, inception: torch.nn.Module) -> np.ndarray:
    """images: (N, 3, H, W) in [-1, 1] -> (N, 2048) pool features"""
    imgs = (images.clamp(-1, 1) + 1) / 2  # -> [0, 1]
    imgs = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
    acts = inception(imgs)[0].squeeze(-1).squeeze(-1)
    return acts.cpu().numpy()


def compute_stats(acts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(acts, axis=0), np.cov(acts, rowvar=False)


def compute_fid(mu1, sigma1, mu2, sigma2) -> float:
    from pytorch_fid.fid_score import calculate_frechet_distance

    return float(calculate_frechet_distance(mu1, sigma1, mu2, sigma2))


def stats_from_generator(generate: Callable[[int], torch.Tensor], n_samples: int, inception: torch.nn.Module,
                         batch_size: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """Inception stats over n_samples images produced by generate(batch_size) -> (bs, 3, H, W) in [-1, 1]."""
    acts = []
    remaining = n_samples
    while remaining > 0:
        bs = min(batch_size, remaining)
        acts.append(get_activations(generate(bs), inception))
        remaining -= bs
    return compute_stats(np.concatenate(acts, axis=0))


def load_or_compute_real_stats(path: str, sample_real: Callable[[int], torch.Tensor], n_samples: int,
                               inception: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """Load cached real-image stats from path, or compute them from sample_real and cache them."""
    if os.path.exists(path):
        print(f"Loading cached real stats from {path}")
        data = np.load(path)
        return data['mu'], data['sigma']

    print(f"Computing real image stats from {n_samples} samples (one-time)...")
    real = sample_real(n_samples)
    mu, sigma = compute_stats(np.concatenate([get_activations(b, inception) for b in real.split(50)], axis=0))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, mu=mu, sigma=sigma)
    print(f"Real stats saved to {path}")
    return mu, sigma
