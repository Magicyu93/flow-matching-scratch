"""
Train a class-conditional flow-matching UNet with classifier-free guidance on ImageNet 64x64.

Example:
    python scripts/train.py --device cuda:1 --steps 300000 --num_classes 10
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch

from flowmatch.config import IMAGE_SHAPE, MODEL_CONFIG
from flowmatch.data import ImageNetSampler
from flowmatch.model import UNet
from flowmatch.paths import GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta
from flowmatch.simulate import sample_cfg
from flowmatch.trainer import CFGTrainer
from flowmatch.viz import save_guidance_grid, save_loss_curve


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=300000, help="number of optimizer steps")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--eta", type=float, default=0.1, help="probability of dropping the label (CFG)")
    p.add_argument("--use_muon", action="store_true", help="use Muon for weight matrices (Adam otherwise)")
    p.add_argument("--num_classes", type=int, default=10, help="train on ImageNet labels 0..num_classes-1")
    p.add_argument("--cache_dir", default=None, help="HF dataset cache dir (default: $IMAGENET64_CACHE_DIR)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt_dir", default="checkpoints", help="where periodic checkpoints are written")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--resume", default=None, help="state_dict to initialize from, e.g. checkpoints/latest.pt")
    p.add_argument("--final_ckpt", default="checkpoint.pt", help="final checkpoint (weights + config + losses)")
    p.add_argument("--out_dir", default="outputs", help="where the loss curve and sample grid are written")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    path = GaussianConditionalProbabilityPath(
        p_data=ImageNetSampler(num_classes=args.num_classes, cache_dir=args.cache_dir),
        p_simple_shape=list(IMAGE_SHAPE),
        alpha=LinearAlpha(),
        beta=LinearBeta(),
    ).to(device)
    model = UNet(**MODEL_CONFIG)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    trainer = CFGTrainer(path=path, model=model, eta=args.eta)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed weights from {args.resume} (optimizer and LR schedule start fresh)")

    losses = trainer.train(
        num_steps=args.steps,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        use_muon=args.use_muon,
        ckpt_dir=args.ckpt_dir,
        save_every=args.save_every,
    )

    torch.save({"model_state_dict": model.state_dict(), "model_config": MODEL_CONFIG, "losses": losses},
               args.final_ckpt)
    print(f"Checkpoint saved to {args.final_ckpt}")

    save_loss_curve(losses, os.path.join(args.out_dir, "loss_curve.png"))

    # Sample grid: one row per class, one panel per guidance scale, shared initial noise
    samples_per_class = 4
    y = torch.arange(args.num_classes, device=device).repeat_interleave(samples_per_class)
    x0 = torch.randn(len(y), *IMAGE_SHAPE, device=device)
    samples = {w: sample_cfg(model, y, w, num_steps=100, x0=x0) for w in [1.0, 3.0, 5.0, 7.0]}
    out = os.path.join(args.out_dir, "guidance.png")
    save_guidance_grid(samples, nrow=samples_per_class, path=out, dpi=150)
    print(f"Saved guidance comparison to {out}")


if __name__ == "__main__":
    main()
