import os
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import NULL_LABEL
from .model import ConditionalVectorField
from .paths import GaussianConditionalProbabilityPath


MiB = 1024 ** 2


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-Schulz.
    Applies Nesterov momentum then orthogonalizes the update for 2D+ params
    (weight matrices). 1D params (biases, norms) are left to Adam.
    Reference: https://github.com/KellerJordan/Muon
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(G: torch.Tensor, steps: int) -> torch.Tensor:
        """Newton-Schulz iteration to compute the orthogonal factor of G."""
        a, b, c = 3.4445, -4.7750, 2.0315
        X = G / (G.norm() + 1e-7)
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        for _ in range(steps):
            A = X @ X.T
            X = a * X + (b * A + c * (A @ A)) @ X
        if transposed:
            X = X.T
        return X

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf.clone()

                # Orthogonalize weight matrices; leave 1D params as-is
                if g.ndim >= 2:
                    g = self._orthogonalize(g.view(g.shape[0], -1), ns_steps).view(g.shape)

                p.add_(g, alpha=-lr)


def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @abstractmethod
    def get_train_loss(self, batch_size: int = 64, **kwargs) -> torch.Tensor:
        pass

    def get_optimizers(self, lr: float, use_muon: bool = False) -> List[torch.optim.Optimizer]:
        if not use_muon:
            return [torch.optim.Adam(self.model.parameters(), lr=lr)]

        # Muon for 2D+ weight matrices; Adam for biases, norms, embeddings
        muon_params = [p for p in self.model.parameters() if p.ndim >= 2]
        adam_params = [p for p in self.model.parameters() if p.ndim < 2]
        return [
            Muon(muon_params, lr=lr),
            torch.optim.Adam(adam_params, lr=lr),
        ]

    def train(self, num_steps: int, device: torch.device, lr: float = 1e-3, use_muon: bool = False,
              batch_size: int = 64, ckpt_dir: str = "checkpoints", save_every: int = 500,
              **kwargs) -> List[float]:
        """
        Runs num_steps optimizer steps (one freshly sampled batch per step) with a cosine LR schedule.
        Every save_every steps, writes the model state_dict to
        {ckpt_dir}/checkpoint_<step>.pt and {ckpt_dir}/latest.pt.
        Returns the per-step losses.
        """
        print(f'Training model with size: {model_size_b(self.model) / MiB:.3f} MiB')

        self.model.to(device)
        optimizers = self.get_optimizers(lr, use_muon)
        schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_steps, eta_min=lr * 0.01)
                      for opt in optimizers]
        self.model.train()
        os.makedirs(ckpt_dir, exist_ok=True)

        losses = []
        pbar = tqdm(range(num_steps))
        for step in pbar:
            for opt in optimizers:
                opt.zero_grad()
            loss = self.get_train_loss(batch_size=batch_size, **kwargs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            for opt in optimizers:
                opt.step()
            for sched in schedulers:
                sched.step()
            losses.append(loss.item())
            pbar.set_description(f'Step {step}, loss: {loss.item():.3f}')

            if save_every and (step + 1) % save_every == 0:
                state = self.model.state_dict()
                torch.save(state, os.path.join(ckpt_dir, f'checkpoint_{step + 1}.pt'))
                torch.save(state, os.path.join(ckpt_dir, 'latest.pt'))

        self.model.eval()
        return losses

    def load_checkpoint(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))


class CFGTrainer(Trainer):
    """
    Conditional flow matching with classifier-free guidance: labels are replaced by the
    null label with probability eta so a single network learns both u(x|y) and u(x|null).
    """

    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField, eta: float):
        assert 0 < eta < 1
        super().__init__(model)
        self.eta = eta
        self.path = path

    def get_train_loss(self, batch_size: int = 64, **kwargs) -> torch.Tensor:
        # Step 1: Sample z, y from p_data
        z, y = self.path.sample_conditioning_variable(batch_size)

        # Step 2: Set each label to the null label with probability eta
        xi = torch.rand(y.shape[0], device=y.device)
        y[xi < self.eta] = NULL_LABEL

        # Step 3: Sample t and x.
        # Clamp away from t=1 to avoid the 1/(1-t) divergence in the conditional vector field.
        t = torch.rand(y.shape[0], 1, 1, 1, device=y.device) * (1 - 1e-4)
        x = self.path.sample_conditional_path(z, t)

        # Step 4: Regress the model onto the conditional vector field
        u_model = self.model(x, t, y)
        u_target = self.path.conditional_vector_field(x, z, t)
        return torch.nn.functional.mse_loss(u_model, u_target)
