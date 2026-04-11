#!/usr/bin/env python3
"""
Apply a trained checkpoint to one image or a full folder tree.

Example:
  python predict_denoise.py --weights checkpoints/best.pt --input dirty_folder --output out_folder
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from denoise.inference import load_image_tensor, predict_tiled, save_tensor_image
from denoise.model import UNetDenoise


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run denoising on images")
    p.add_argument("--weights", type=str, required=True, help="best.pt or last.pt from training")
    p.add_argument("--input", type=str, required=True, help="Image file or folder")
    p.add_argument("--output", type=str, required=True, help="Output file or folder")
    p.add_argument("--pattern", type=str, default="*.bag.bmp.jpg")
    p.add_argument("--tile", type=int, default=512, help="Tile size (multiple of 16)")
    p.add_argument("--overlap", type=int, default=64)
    return p.parse_args()


def load_model(weights_path: Path, device: torch.device) -> UNetDenoise:
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt.get("config") or {}
    base = int(cfg.get("base_channels", 48))
    model = UNetDenoise(base=base).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.tile % 16 != 0:
        raise SystemExit("--tile must be divisible by 16")

    device = pick_device()
    weights = Path(args.weights)
    inp = Path(args.input)
    out = Path(args.output)

    model = load_model(weights, device)

    if inp.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        t = load_image_tensor(inp, device)
        pred = predict_tiled(model, t, tile_size=args.tile, overlap=args.overlap, device=device)
        save_tensor_image(pred, out)
        print(f"Wrote {out}")
        return

    if not inp.is_dir():
        raise SystemExit(f"Not a file or directory: {inp}")

    # Mirror tree: for each matching file under inp, write under out with same relative path
    out.mkdir(parents=True, exist_ok=True)
    meta = {"weights": str(weights.resolve()), "config": None}
    cfg_path = weights.parent / "config.json"
    if cfg_path.is_file():
        meta["config"] = json.loads(cfg_path.read_text(encoding="utf-8"))
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    files = sorted(inp.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No files matching {args.pattern!r} under {inp}")

    for f in files:
        rel = f.relative_to(inp)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        t = load_image_tensor(f, device)
        pred = predict_tiled(model, t, tile_size=args.tile, overlap=args.overlap, device=device)
        save_tensor_image(pred, dest)
        print(dest)

    print(f"Done: {len(files)} images -> {out.resolve()}")


if __name__ == "__main__":
    main()


