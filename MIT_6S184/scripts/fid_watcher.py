"""
Watch a checkpoint directory; for each new checkpoint, save a sample grid and compute FID.
Results are appended to a CSV, and checkpoints already in the CSV are skipped, so it is safe to restart.

Example (alongside training on another GPU):
    python scripts/fid_watcher.py --device cuda:0
"""
import argparse
import csv
import glob
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from flowmatch.config import MODEL_CONFIG
from flowmatch.fid import compute_fid, get_inception, load_or_compute_real_stats, stats_from_generator
from flowmatch.model import UNet
from flowmatch.simulate import sample_cfg
from flowmatch.viz import save_guidance_grid


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--out_dir", default="outputs/fid_viz", help="sample grids per checkpoint")
    p.add_argument("--log", default="outputs/fid_log.csv")
    p.add_argument("--real_stats", default="outputs/real_stats_10class.npz",
                   help="cached Inception stats of real images; computed if missing")
    p.add_argument("--cache_dir", default=None, help="HF dataset cache dir (default: $IMAGENET64_CACHE_DIR)")
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--guidance", type=float, default=1.5)
    p.add_argument("--samples_per_class", type=int, default=4, help="for the visual grid")
    p.add_argument("--num_steps", type=int, default=100, help="Euler steps")
    p.add_argument("--n_samples", type=int, default=500, help="samples used for FID")
    p.add_argument("--poll", type=int, default=30, help="seconds between directory scans")
    p.add_argument("--once", action="store_true", help="process existing checkpoints and exit")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def step_of(path: str) -> int:
    return int(re.search(r"checkpoint_(\d+)", path).group(1))


def process_checkpoint(args, ckpt_path, model, inception, real_mu, real_sigma, step, device):
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # Visual grid with fixed noise so grids are comparable across checkpoints
    torch.manual_seed(42)
    y_viz = torch.arange(args.num_classes, device=device).repeat_interleave(args.samples_per_class)
    x1_viz = sample_cfg(model, y_viz, args.guidance, num_steps=args.num_steps)
    out_img = os.path.join(args.out_dir, f"epoch_{step:06d}.png")
    save_guidance_grid({args.guidance: x1_viz}, nrow=args.samples_per_class, path=out_img,
                       title=f"Step {step:,}")

    # FID over n_samples images with random labels
    torch.manual_seed(123)

    def generate(bs):
        y = torch.randint(0, args.num_classes, (bs,), device=device)
        return sample_cfg(model, y, args.guidance, num_steps=args.num_steps)

    gen_mu, gen_sigma = stats_from_generator(generate, args.n_samples, inception)
    return compute_fid(real_mu, real_sigma, gen_mu, gen_sigma), out_img


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Initializing Inception model...")
    inception = get_inception(device)

    def sample_real(n):
        from flowmatch.data import ImageNetSampler
        return ImageNetSampler(num_classes=args.num_classes, cache_dir=args.cache_dir).sample(n)[0].to(device)

    real_mu, real_sigma = load_or_compute_real_stats(args.real_stats, sample_real, args.n_samples, inception)

    model = UNet(**MODEL_CONFIG).to(device)

    processed = set()
    if os.path.exists(args.log):
        with open(args.log) as f:
            processed = {int(row["epoch"]) for row in csv.DictReader(f)}
    else:
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "fid", "checkpoint", "output_image"])

    print(f"Already processed {len(processed)} checkpoints")
    print(f"Watching {args.ckpt_dir}/ every {args.poll}s...\n")

    while True:
        for ckpt in sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint_*.pt")), key=step_of):
            step = step_of(ckpt)
            if step in processed:
                continue
            print(f"[step {step:6d}] Processing {ckpt} ...", flush=True)
            try:
                fid, out_img = process_checkpoint(args, ckpt, model, inception, real_mu, real_sigma, step, device)
            except Exception as e:
                print(f"[step {step:6d}] ERROR: {e}", flush=True)
                continue
            with open(args.log, "a", newline="") as f:
                csv.writer(f).writerow([step, f"{fid:.2f}", ckpt, out_img])
            processed.add(step)
            print(f"[step {step:6d}] FID={fid:.2f}  ->  {out_img}", flush=True)

        if args.once:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
