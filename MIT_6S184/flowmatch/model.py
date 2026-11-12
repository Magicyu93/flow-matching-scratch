from abc import ABC, abstractmethod
from typing import List, Sequence, Union
import math

import torch
import torch.nn as nn

from .config import MODEL_CONFIG, NUM_LABEL_EMBEDDINGS


class ConditionalVectorField(nn.Module, ABC):
    """
    Parameterization of the learned conditional vector field u_t^theta(x|y)
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        pass


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bs, c, h, w)
        bs, c, h, w = x.shape
        res = x
        x = self.norm(x)
        x = x.view(bs, c, h * w).transpose(1, 2)  # (bs, h*w, c)
        x, _ = self.attn(x, x, x)
        x = x.transpose(1, 2).view(bs, c, h, w)  # (bs, c, h, w)
        return x + res


class ResidualLayer(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int, y_embed_dim: int):
        super().__init__()
        num_groups = min(8, channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        # Converts (bs, time_embed_dim) -> (bs, channels)
        self.time_adapter = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        # Converts (bs, y_embed_dim) -> (bs, channels)
        self.y_adapter = nn.Sequential(
            nn.Linear(y_embed_dim, y_embed_dim),
            nn.SiLU(),
            nn.Linear(y_embed_dim, channels)
        )

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        res = x.clone()  # (bs, c, h, w)

        # Initial conv block
        x = self.block1(x)  # (bs, c, h, w)

        # Add time embedding
        t_embed = self.time_adapter(t_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + t_embed

        # Add y embedding (conditional embedding)
        y_embed = self.y_adapter(y_embed).unsqueeze(-1).unsqueeze(-1)  # (bs, c, 1, 1)
        x = x + y_embed

        # Second conv block
        x = self.block2(x)  # (bs, c, h, w)

        # Add back residual
        x = x + res  # (bs, c, h, w)

        return x


class Encoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 y_embed_dim: int, use_attention: bool = False):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.attn = SelfAttentionLayer(channels_in) if use_attention else None
        self.downsample = nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c_in, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c_in, h, w) -> (bs, c_in, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Optional self-attention after residual blocks, before downsample
        if self.attn is not None:
            x = self.attn(x)

        # Downsample: (bs, c_in, h, w) -> (bs, c_out, h // 2, w // 2)
        x = self.downsample(x)

        return x


class Midcoder(nn.Module):
    def __init__(self, channels: int, num_residual_layers: int, t_embed_dim: int, y_embed_dim: int,
                 use_attention: bool = True):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])
        self.attn = SelfAttentionLayer(channels) if use_attention else None

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Pass through residual blocks: (bs, c, h, w) -> (bs, c, h, w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        # Optional self-attention after all residual blocks
        if self.attn is not None:
            x = self.attn(x)

        return x


class Decoder(nn.Module):
    def __init__(self, channels_in: int, channels_out: int, num_residual_layers: int, t_embed_dim: int,
                 y_embed_dim: int):
        super().__init__()
        self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                                      nn.Conv2d(channels_in, channels_out, kernel_size=3, padding=1))
        self.res_blocks = nn.ModuleList([
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim) for _ in range(num_residual_layers)
        ])

    def forward(self, x: torch.Tensor, t_embed: torch.Tensor, y_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: (bs, c, h, w)
        - t_embed: (bs, t_embed_dim)
        - y_embed: (bs, y_embed_dim)
        """
        # Upsample: (bs, c_in, h, w) -> (bs, c_out, 2 * h, 2 * w)
        x = self.upsample(x)

        # Pass through residual blocks: (bs, c_out, h, w) -> (bs, c_out, 2 * h, 2 * w)
        for block in self.res_blocks:
            x = block(x, t_embed, y_embed)

        return x


class UNet(ConditionalVectorField):
    def __init__(self, channels: List[int], num_residual_layers: int, t_embed_dim: int, y_embed_dim: int,
                 in_channels: int = 3, attention_levels: Sequence[int] = (-1,)):
        super().__init__()
        self.in_channels = in_channels

        # Initial convolution: (bs, in_channels, h, w) -> (bs, c_0, h, w)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(min(8, channels[0]), channels[0]),
            nn.SiLU()
        )

        # Initialize time embedder
        self.time_embedder = FourierEncoder(t_embed_dim)

        # Initialize y embedder
        self.y_embedder = nn.Embedding(num_embeddings=NUM_LABEL_EMBEDDINGS, embedding_dim=y_embed_dim)  # 1000 classes + null

        # Resolve attention_levels to encoder indices and midcoder flag.
        # Levels are indexed as [enc_0, enc_1, ..., enc_{n-1}, midcoder] with n = len(channels)-1.
        # -1 -> midcoder, -2 -> last encoder, etc.
        num_encoders = len(channels) - 1
        num_levels = num_encoders + 1  # encoders + midcoder
        attn_set = set()
        for lvl in attention_levels:
            idx = lvl if lvl >= 0 else num_levels + lvl
            attn_set.add(idx)
        mid_use_attn = num_encoders in attn_set

        # Encoders, Midcoders, and Decoders
        encoders = []
        decoders = []
        for i, (curr_c, next_c) in enumerate(zip(channels[:-1], channels[1:])):
            enc_use_attn = i in attn_set
            encoders.append(Encoder(curr_c, next_c, num_residual_layers, t_embed_dim, y_embed_dim,
                                    use_attention=enc_use_attn))
            decoders.append(Decoder(next_c, curr_c, num_residual_layers, t_embed_dim, y_embed_dim))
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(reversed(decoders))

        self.midcoder = Midcoder(channels[-1], num_residual_layers, t_embed_dim, y_embed_dim,
                                 use_attention=mid_use_attn)

        # Final convolution
        self.final_conv = nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Args:
        - x: (bs, c, h, w)
        - t: (bs, 1, 1, 1)
        - y: (bs,)
        Returns:
        - u_t^theta(x|y): (bs, c, h, w)
        """
        # Embed t and y
        t_embed = self.time_embedder(t)  # (bs, time_embed_dim)
        y_embed = self.y_embedder(y)  # (bs, y_embed_dim)

        # Initial convolution
        x = self.init_conv(x)  # (bs, c_0, h, w)

        residuals = []

        # Encoders — save features BEFORE downsampling for proper skip connections
        for encoder in self.encoders:
            residuals.append(x.clone())  # save pre-downsample features
            x = encoder(x, t_embed, y_embed)  # (bs, c_i, h, w) -> (bs, c_{i+1}, h // 2, w //2)

        # Midcoder
        x = self.midcoder(x, t_embed, y_embed)

        # Decoders — add skip connection AFTER upsampling so resolutions match
        for decoder in self.decoders:
            res = residuals.pop()  # (bs, c_i, h, w) — pre-downsample features
            x = decoder(x, t_embed, y_embed)  # upsample: (bs, c_{i+1}, h, w) -> (bs, c_i, 2h, 2w)
            x = x + res  # shapes now match

        # Final convolution
        x = self.final_conv(x)  # (bs, c, h, w)

        return x


def load_unet(path: str, device: Union[str, torch.device] = "cpu") -> UNet:
    """
    Load a UNet from either checkpoint format produced by this repo:
    - a raw state_dict (checkpoints/checkpoint_<step>.pt, latest.pt), built with MODEL_CONFIG
    - a dict with 'model_state_dict' and 'model_config' (final checkpoint.pt from scripts/train.py)
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model = UNet(**ckpt.get("model_config", MODEL_CONFIG))
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model = UNet(**MODEL_CONFIG)
        model.load_state_dict(ckpt)
    return model.to(device).eval()
