"""
Render sample grids (several guidance scales, fixed noise) for every N-step checkpoint,
to visualize how samples evolve during training. Existing images are skipped.

Example:
    python scripts/generate_progression.py --every 10000 --guidance 1 3 5
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from flowmatch.config import IMAGE_SHAPE, MODEL_CONFIG
from flowmatch.model import UNet
from flowmatch.simulate import sample_cfg
from flowmatch.viz import save_guidance_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs/checkpoints_viz")
    p.add_argument("--every", type=int, default=10000, help="only checkpoints whose step is a multiple of this")
    p.add_argument("--guidance", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--samples_per_class", type=int, default=4)
    p.add_argument("--num_steps", type=int, default=200, help="Euler steps")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    def step_of(p):
        return int(re.search(r"checkpoint_(\d+)", p).group(1))

    ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint_*.pt")), key=step_of)
    ckpts = [p for p in ckpts if step_of(p) % args.every == 0]
    print(f"Found {len(ckpts)} checkpoints to process")

    # Fixed noise for consistency across checkpoints
    torch.manual_seed(42)
    y = torch.arange(args.num_classes, device=device).repeat_interleave(args.samples_per_class)
    x0 = torch.randn(len(y), *IMAGE_SHAPE, device=device)

    model = UNet(**MODEL_CONFIG).to(device)
    for ckpt in ckpts:
        step = step_of(ckpt)
        out = os.path.join(args.out_dir, f"epoch_{step:06d}.png")
        if os.path.exists(out):
            print(f"[step {step:6d}] already exists, skipping")
            continue

        print(f"[step {step:6d}] loading {ckpt} ...", flush=True)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        samples = {w: sample_cfg(model, y, w, num_steps=args.num_steps, x0=x0.clone()) for w in args.guidance}
        save_guidance_grid(samples, nrow=args.samples_per_class, path=out, title=f"Step {step:,}")
        print(f"[step {step:6d}] saved to {out}", flush=True)

    print("\nAll done!")


if __name__ == "__main__":
    main()
