# Flow Matching from Scratch (MIT 6.S184)

A from-scratch implementation of **conditional flow matching with classifier-free guidance (CFG)**,
following MIT 6.S184 (*Generative AI with Stochastic Differential Equations*), trained on
**ImageNet 64×64** (first 10 classes by default).

- **Probability path:** Gaussian path `x_t = α_t z + β_t ε` with linear schedule `α_t = t`, `β_t = 1 − t`
- **Model:** UNet (~87M params) with Fourier time embedding, label embedding (1000 classes + null token),
  residual blocks, and self-attention at the deepest levels
- **Training:** regress onto the conditional vector field `u_t(x|z)`; labels dropped to null with prob. `η = 0.1`
- **Sampling:** Euler integration of the guided field `(1 − w) u(x|∅) + w u(x|y)` from noise (t=0) to data (t=1)

## Layout

```
flowmatch/              library code
  config.py             model config, null label, image shape, dataset name
  data.py               Sampleable interface, Gaussian noise, MNIST / ImageNet-64 samplers
  paths.py              conditional probability paths and α/β schedules
  model.py              UNet vector field + load_unet() checkpoint loader
  simulate.py           ODE/SDE, Euler / Euler–Maruyama, CFG ODE, sample_cfg()
  trainer.py            Trainer loop, CFGTrainer, Muon optimizer
  fid.py                Inception features + FID
  viz.py                sample-grid and loss-curve plots
scripts/
  train.py              train a model
  sample.py             sample a grid from a checkpoint
  fid_watcher.py        compute FID for each new checkpoint during training
  generate_progression.py   sample grids across training checkpoints
notebooks/              exploratory notebooks (may reference older module names)
```

## Setup

```bash
pip install -r requirements.txt
# Optional: where the HuggingFace ImageNet-64 dataset (benjamin-paine/imagenet-1k-64x64) is cached
export IMAGENET64_CACHE_DIR=/path/to/ImageNet64_HF
```

## Usage

Run from this directory (scripts work from anywhere, but default paths are relative).

```bash
# Train (writes checkpoints/checkpoint_<step>.pt every 500 steps, then checkpoint.pt + outputs/)
python scripts/train.py --device cuda:1 --steps 300000 --batch_size 256 --lr 2e-4

# Resume weights from a previous run (optimizer and LR schedule restart)
python scripts/train.py --resume checkpoints/latest.pt

# Track FID while training (on another GPU); --once processes existing checkpoints and exits
python scripts/fid_watcher.py --device cuda:0

# Sample from a trained model
python scripts/sample.py --ckpt checkpoints/checkpoint_300000.pt --guidance 1 3 5 --num_steps 100

# Grids over training
python scripts/generate_progression.py --every 10000
```

## Results

10-class ImageNet-64, 300k steps, batch 256, Adam lr 2e-4 with cosine decay:

| Step | FID |
|-----:|----:|
| 500 | 280.4 |
| 50k | 122.9 |
| 277.5k (best) | 78.1 |
| 300k | 79.2 |

FID plateaus around 78–80 after ~240k steps. These numbers use only **500 generated vs. 500 real
samples** (w = 1.5, 100 Euler steps), so they are noisy and biased upward — useful for comparing
checkpoints, not for comparison with published results.

## Possible next steps

- EMA of weights for sampling
- FID on 10k+ samples
- Higher-order ODE solvers (Heun / midpoint) and fewer steps
- Save optimizer/scheduler state for exact resume
