from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def collect_pairs(
    dirty_root: str | Path,
    clean_root: str | Path,
    pattern: str = "*.bag.bmp.jpg",
) -> List[Tuple[Path, Path]]:
    dirty_root = Path(dirty_root).resolve()
    clean_root = Path(clean_root).resolve()
    pairs: List[Tuple[Path, Path]] = []
    missing: List[Path] = []

    for dirty_path in sorted(dirty_root.rglob(pattern)):
        rel = dirty_path.relative_to(dirty_root)
        clean_path = clean_root / rel
        if clean_path.is_file():
            pairs.append((dirty_path, clean_path))
        else:
            missing.append(rel)

    if missing:
        n = len(missing)
        sample = missing[:5]
        msg = f"{n} dirty files have no matching clean file under clean_root."
        if sample:
            msg += f" Examples (relative): {[str(p) for p in sample]}"
        raise FileNotFoundError(msg)

    return pairs


def train_val_split(
    pairs: Sequence[Tuple[Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)
    if val_ratio <= 0 or len(items) <= 1:
        return items, []
    n_val = max(1, int(len(items) * val_ratio))
    n_val = min(n_val, len(items) - 1)
    val = items[:n_val]
    train = items[n_val:]
    return train, val


class PairedImageFolder(Dataset):
    """Returns (dirty_tensor, clean_tensor) in [0, 1], shape [3, H, W]."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        patch_size: int,
        augment: bool,
        transform_dirty: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
        | None = None,
    ) -> None:
        self.pairs = list(pairs)
        self.patch_size = patch_size
        self.augment = augment
        self.transform_dirty = transform_dirty

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_pair(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        dp, cp = self.pairs[idx]
        dirty = Image.open(dp).convert("RGB")
        clean = Image.open(cp).convert("RGB")
        d = torch.from_numpy(np.array(dirty, dtype=np.float32) / 255.0).permute(2, 0, 1)
        c = torch.from_numpy(np.array(clean, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return d, c

    def _random_crop_sync(
        self, d: torch.Tensor, c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, h, w = d.shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            # pad bottom/right
            d = torch.nn.functional.pad(d, (0, pad_w, 0, pad_h), mode="reflect")
            c = torch.nn.functional.pad(c, (0, pad_w, 0, pad_h), mode="reflect")
            _, h, w = d.shape
        i = random.randint(0, h - ps)
        j = random.randint(0, w - ps)
        d = d[:, i : i + ps, j : j + ps]
        c = c[:, i : i + ps, j : j + ps]
        return d, c

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        d, c = self._load_pair(idx)
        d, c = self._random_crop_sync(d, c)
        if self.augment and random.random() < 0.5:
            d = torch.flip(d, dims=(2,))
            c = torch.flip(c, dims=(2,))
        if self.transform_dirty is not None:
            d, c = self.transform_dirty(d, c)
        return d, c
