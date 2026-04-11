from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from denoise.model import UNetDenoise


def load_image_tensor(path: str | Path, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return t.unsqueeze(0).to(device)


def save_tensor_image(t: torch.Tensor, path: str | Path) -> None:
    x = t.squeeze(0).detach().cpu().clamp(0, 1).numpy()
    x = (x * 255.0).round().astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))
    Image.fromarray(x).save(path)


def _tiles(
    h: int, w: int, tile: int, overlap: int
) -> Iterator[Tuple[int, int, int, int]]:
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, h - tile + 1), step))
    xs = list(range(0, max(1, w - tile + 1), step))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    for y in ys:
        for x in xs:
            yield y, x, min(tile, h - y), min(tile, w - x)


@torch.inference_mode()
def predict_tiled(
    model: UNetDenoise,
    dirty: torch.Tensor,
    *,
    tile_size: int = 512,
    overlap: int = 64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    dirty: [1, 3, H, W] in [0,1]
    Returns same shape on device.
    """
    if device is None:
        device = next(model.parameters()).device
    b, _, h, w = dirty.shape
    assert b == 1
    out = torch.zeros_like(dirty)
    weight = torch.zeros((1, 1, h, w), device=device, dtype=dirty.dtype)

    dirty = dirty.to(device)
    model.eval()
    for y, x, th, tw in _tiles(h, w, tile_size, overlap):
        patch = dirty[:, :, y : y + th, x : x + tw]
        if th < tile_size or tw < tile_size:
            pad_h = tile_size - th
            pad_w = tile_size - tw
            patch_in = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            pad_h = pad_w = 0
            patch_in = patch
        pred = model(patch_in)
        if pad_h or pad_w:
            pred = pred[:, :, :th, :tw]
        out[:, :, y : y + th, x : x + tw] += pred
        weight[:, :, y : y + th, x : x + tw] += 1.0

    out = out / weight.clamp(min=1e-6)
    return out
