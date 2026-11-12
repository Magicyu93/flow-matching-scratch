"""
Generate a class-conditional sample grid from a trained checkpoint.

Example:
    python scripts/sample.py --ckpt checkpoints/checkpoint_300000.pt --guidance 1 3 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from flowmatch.config import IMAGE_SHAPE
from flowmatch.model import load_unet
from flowmatch.simulate import sample_cfg
from flowmatch.viz import save_guidance_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="checkpoint.pt", help="final checkpoint.pt or a raw state_dict checkpoint")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--num_classes", type=int, default=10, help="sample labels 0..num_classes-1")
    p.add_argument("--samples_per_class", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=100, help="Euler steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="outputs/samples.png")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model = load_unet(args.ckpt, device)
    print(f"Loaded {args.ckpt}")

    torch.manual_seed(args.seed)
    y = torch.arange(args.num_classes, device=device).repeat_interleave(args.samples_per_class)
    x0 = torch.randn(len(y), *IMAGE_SHAPE, device=device)
    samples = {w: sample_cfg(model, y, w, num_steps=args.num_steps, x0=x0, progress=True) for w in args.guidance}
    save_guidance_grid(samples, nrow=args.samples_per_class, path=args.out, dpi=150)
    print(f"Saved samples to {args.out}")


if __name__ == "__main__":
    main()
