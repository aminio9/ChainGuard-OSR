"""Base dataset classes for network traffic flows."""

from __future__ import annotations

from typing import Any, Optional, Tuple
import torch
from torch.utils.data import Dataset


class FlowDataset(Dataset):
    """PyTorch Dataset container for network flow feature vectors and classification labels."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        raw_df: Optional[Any] = None,
        recipe_name: Optional[str] = None,
    ):
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Features count ({features.shape[0]}) does not match labels count ({labels.shape[0]})."
            )
        self.features = features.float()
        self.labels = labels.long()
        self.recipe_name = recipe_name
        self._raw_df = None
        if isinstance(raw_df, dict):
            self.raw_df_dict = raw_df
        elif raw_df is not None and hasattr(raw_df, "columns"):
            self.raw_df_dict = {str(c): raw_df[c].to_numpy() for c in raw_df.columns}
        else:
            self.raw_df_dict = None

    @property
    def raw_df(self) -> Optional[Any]:
        if self._raw_df is None and self.raw_df_dict is not None:
            import pandas as pd
            self._raw_df = pd.DataFrame(self.raw_df_dict)
        return self._raw_df

    @raw_df.setter
    def raw_df(self, val: Optional[Any]) -> None:
        self._raw_df = val
        if val is not None and hasattr(val, "columns"):
            self.raw_df_dict = {str(c): val[c].to_numpy() for c in val.columns}
        elif isinstance(val, dict):
            self.raw_df_dict = val
        else:
            self.raw_df_dict = None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def unique_classes(self) -> torch.Tensor:
        return torch.unique(self.labels)


class FastBatchSampler:
    """High-throughput In-Device mini-batch generator.
    
    Eliminates PyTorch DataLoader CPU iteration overhead and per-batch PCIe bus transfers
    by maintaining contiguous feature/label buffers directly on target hardware (GPU VRAM or CPU).
    """

    def __init__(self, dataset: FlowDataset, device: torch.device):
        self.device = device
        # Cache device tensors directly on the dataset to avoid per-round re-allocation
        dev_cache = getattr(dataset, "_device_cache", None)
        if dev_cache is None or dev_cache[0].device != device:
            dev_feats = dataset.features.to(device, non_blocking=True)
            dev_labs = dataset.labels.to(device, non_blocking=True)
            dataset._device_cache = (dev_feats, dev_labs)

        self.features, self.labels = dataset._device_cache
        self.num_samples = self.features.size(0)

    def iter_batches(self, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        if self.num_samples == 0:
            return

        drop = drop_last and (self.num_samples > batch_size)
        end_limit = self.num_samples - (batch_size if drop else 0) + 1

        if shuffle:
            perm = torch.randperm(self.num_samples, device=self.device)
            feats = self.features[perm]
            labs = self.labels[perm]
        else:
            feats = self.features
            labs = self.labels

        for i in range(0, end_limit, batch_size):
            bx = feats[i : i + batch_size]
            by = labs[i : i + batch_size]
            if bx.size(0) <= 1:
                continue
            yield bx, by
