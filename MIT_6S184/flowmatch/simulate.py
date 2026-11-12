"""ODE/SDE abstractions, numerical simulators, and classifier-free-guided sampling."""
from abc import ABC, abstractmethod
from typing import Optional, Sequence

import torch
from tqdm import tqdm

from .config import IMAGE_SHAPE, NULL_LABEL
from .model import ConditionalVectorField


class ODE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the ODE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass


class SDE(ABC):
    @abstractmethod
    def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the drift coefficient of the SDE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - drift_coefficient: shape (bs, c, h, w)
        """
        pass

    @abstractmethod
    def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Returns the diffusion coefficient of the SDE.
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
        Returns:
            - diffusion_coefficient: shape (bs, c, h, w)
        """
        pass


class Simulator(ABC):
    @abstractmethod
    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        """
        Takes one simulation step
        Args:
            - xt: state at time t, shape (bs, c, h, w)
            - t: time, shape (bs, 1, 1, 1)
            - h: step size, shape (bs, 1, 1, 1)
        Returns:
            - nxt: state at time t + h (bs, c, h, w)
        """
        pass

    @torch.no_grad()
    def simulate(self, x: torch.Tensor, ts: torch.Tensor, progress: bool = True, **kwargs):
        """
        Simulates using the discretization given by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - x_final: final state at time ts[-1], shape (bs, c, h, w)
        """
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1), disable=not progress, leave=False):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
        return x

    @torch.no_grad()
    def simulate_with_trajectory(self, x: torch.Tensor, ts: torch.Tensor, progress: bool = True, **kwargs):
        """
        Simulates using the discretization given by ts
        Args:
            - x: initial state, shape (bs, c, h, w)
            - ts: timesteps, shape (bs, nts, 1, 1, 1)
        Returns:
            - xs: trajectory of xts over ts, shape (batch_size, nts, c, h, w)
        """
        xs = [x.clone()]
        nts = ts.shape[1]
        for t_idx in tqdm(range(nts - 1), disable=not progress, leave=False):
            t = ts[:, t_idx]
            h = ts[:, t_idx + 1] - ts[:, t_idx]
            x = self.step(x, t, h, **kwargs)
            xs.append(x.clone())
        return torch.stack(xs, dim=1)


class EulerSimulator(Simulator):
    def __init__(self, ode: ODE):
        self.ode = ode

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        return xt + self.ode.drift_coefficient(xt, t, **kwargs) * h


class EulerMaruyamaSimulator(Simulator):
    def __init__(self, sde: SDE):
        self.sde = sde

    def step(self, xt: torch.Tensor, t: torch.Tensor, h: torch.Tensor, **kwargs):
        drift = self.sde.drift_coefficient(xt, t, **kwargs)
        diffusion = self.sde.diffusion_coefficient(xt, t, **kwargs)
        return xt + drift * h + diffusion * torch.sqrt(h) * torch.randn_like(xt)


def record_every(num_timesteps: int, record_every: int) -> torch.Tensor:
    """
    Compute the indices to record in the trajectory given a record_every parameter
    """
    if record_every == 1:
        return torch.arange(num_timesteps)
    return torch.cat(
        [
            torch.arange(0, num_timesteps - 1, record_every),
            torch.tensor([num_timesteps - 1]),
        ]
    )


class CFGVectorFieldODE(ODE):
    """ODE whose drift is the classifier-free-guided vector field (1 - w) u(x|null) + w u(x|y)."""

    def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0):
        self.net = net
        self.guidance_scale = guidance_scale

    def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        """
        guided_vector_field = self.net(x, t, y)
        unguided_y = torch.full_like(y, NULL_LABEL)  # must match the null label used in training
        unguided_vector_field = self.net(x, t, unguided_y)
        return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field


@torch.no_grad()
def sample_cfg(model: ConditionalVectorField, y: torch.Tensor, guidance_scale: float, num_steps: int = 100,
               x0: Optional[torch.Tensor] = None, shape: Sequence[int] = IMAGE_SHAPE,
               progress: bool = False) -> torch.Tensor:
    """
    Generate samples for labels y by integrating the CFG ODE from t=0 (noise) to t=1 with Euler steps.
    Args:
        - model: trained vector field
        - y: labels, (bs,)
        - guidance_scale: CFG weight w
        - num_steps: number of time grid points in [0, 1]
        - x0: optional initial noise (bs, *shape); sampled from N(0, I) if None
    Returns:
        - x1: samples, (bs, *shape), roughly in [-1, 1]
    """
    device = y.device
    bs = y.shape[0]
    if x0 is None:
        x0 = torch.randn(bs, *shape, device=device)
    ts = torch.linspace(0, 1, num_steps, device=device).view(1, -1, 1, 1, 1).expand(bs, -1, 1, 1, 1)
    simulator = EulerSimulator(CFGVectorFieldODE(model, guidance_scale=guidance_scale))
    return simulator.simulate(x0, ts, progress=progress, y=y)
