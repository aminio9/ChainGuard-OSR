"""Abstract base class for network models."""

from typing import Any
import abc
import torch
import torch.nn as nn


class BaseNetwork(nn.Module, abc.ABC):
    """Abstract interface for all model components in the research framework."""

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        pass
