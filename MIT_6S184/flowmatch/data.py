from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from .config import IMAGENET64_HF_NAME, default_cache_dir


class Sampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @abstractmethod
    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
            - labels: shape (batch_size,), or None for unlabeled distributions
        """
        pass


class IsotropicGaussian(nn.Module, Sampleable):
    """
    Sampleable wrapper around torch.randn
    """

    def __init__(self, shape: List[int], std: float = 1.0):
        """
        shape: shape of sampled data
        """
        super().__init__()
        self.shape = shape
        self.std = std
        self.register_buffer("dummy", torch.zeros(1))  # tracks the device when self.to(...) is called

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, None]:
        return self.std * torch.randn(num_samples, *self.shape, device=self.dummy.device), None


class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset, resized to 32x32 and normalized to [-1, 1]
    """

    def __init__(self, root: str = "./data"):
        super().__init__()
        self.dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        )
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy.device)
        labels = torch.tensor(labels, dtype=torch.int64, device=self.dummy.device)
        return samples, labels


class ImageNetSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for ImageNet 64x64 (HuggingFace), restricted to labels < num_classes
    and normalized to [-1, 1]
    """

    def __init__(self, num_classes: int = 10, cache_dir: Optional[str] = None, split: str = "train"):
        super().__init__()
        from datasets import load_dataset  # imported lazily: only needed for ImageNet

        full_dataset = load_dataset(IMAGENET64_HF_NAME, cache_dir=cache_dir or default_cache_dir(), split=split)
        self.dataset = full_dataset.filter(lambda x: x["label"] < num_classes)
        print(f"ImageNetSampler: using {len(self.dataset)} samples from labels 0-{num_classes - 1}")
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.register_buffer("dummy", torch.zeros(1))

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples].tolist()
        batch = self.dataset[indices]  # single batched fetch

        samples = torch.stack([self.transform(img.convert("RGB")) for img in batch["image"]])
        labels = torch.tensor(batch["label"], dtype=torch.int64)
        return samples.to(self.dummy.device), labels.to(self.dummy.device)
