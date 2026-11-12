"""Plotting helpers that write figures to disk (non-interactive backend)."""
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def _ensure_parent(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def save_guidance_grid(samples_by_w: Dict[float, torch.Tensor], nrow: int, path: str,
                       title: Optional[str] = None, dpi: int = 100):
    """
    One panel per guidance scale; each panel is a grid of images in [-1, 1].
    With labels arranged via repeat_interleave(nrow), each row of a panel is one class.
    """
    n = len(samples_by_w)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 10), squeeze=False)
    for ax, (w, x) in zip(axes[0], samples_by_w.items()):
        grid = make_grid(x, nrow=nrow, normalize=True, value_range=(-1, 1))
        ax.imshow(grid.permute(1, 2, 0).cpu())
        ax.axis("off")
        ax.set_title(f"$w={w:.1f}$", fontsize=25)
    if title:
        fig.suptitle(title, fontsize=30, y=1.01)
    plt.tight_layout()
    _ensure_parent(path)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_loss_curve(losses: List[float], path: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    plt.tight_layout()
    _ensure_parent(path)
    plt.savefig(path, dpi=150)
    plt.close(fig)
