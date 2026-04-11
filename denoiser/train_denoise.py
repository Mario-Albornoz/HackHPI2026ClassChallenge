from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from denoise.dataset import PairedImageFolder, collect_pairs, train_val_split
from denoise.model import UNetDenoise


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train paired denoising U-Net")
    p.add_argument("--dirty", type=str, required=True, help="Root folder with dirty images")
    p.add_argument("--clean", type=str, required=True, help="Root folder with clean images")
    p.add_argument("--pattern", type=str, default="*.bag.bmp.jpg", help="Glob under each root")
    p.add_argument("--out", type=str, default="checkpoints", help="Directory for weights + config")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch-size", type=int, default=256, help="Must be divisible by 16")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-channels", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--resume", type=str, default="", help="Path to .pth to resume")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.patch_size % 16 != 0:
        raise SystemExit("--patch-size must be divisible by 16 (U-Net pooling depth).")

    device = pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = collect_pairs(args.dirty, args.clean, pattern=args.pattern)
    train_pairs, val_pairs = train_val_split(pairs, args.val_ratio, args.seed)

    train_ds = PairedImageFolder(train_pairs, args.patch_size, augment=True)
    val_ds = PairedImageFolder(val_pairs, args.patch_size, augment=False) if val_pairs else None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        if val_ds is not None
        else None
    )

    model = UNetDenoise(base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    criterion = nn.L1Loss()

    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", best_val)

    config = {
        "dirty_root": str(Path(args.dirty).resolve()),
        "clean_root": str(Path(args.clean).resolve()),
        "pattern": args.pattern,
        "patch_size": args.patch_size,
        "base_channels": args.base_channels,
        "pairs_total": len(pairs),
        "pairs_train": len(train_pairs),
        "pairs_val": len(val_pairs),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        for dirty, clean in pbar:
            dirty = dirty.to(device)
            clean = clean.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(dirty)
            loss = criterion(pred, clean)
            loss.backward()
            opt.step()
            running += loss.item() * dirty.size(0)
            n += dirty.size(0)
            pbar.set_postfix(loss=f"{running/n:.5f}")

        sched.step()
        train_loss = running / max(1, n)

        val_loss = None
        if val_loader is not None:
            model.eval()
            vsum, vn = 0.0, 0
            with torch.no_grad():
                for dirty, clean in val_loader:
                    dirty = dirty.to(device)
                    clean = clean.to(device)
                    pred = model(dirty)
                    vsum += criterion(pred, clean).item() * dirty.size(0)
                    vn += dirty.size(0)
            val_loss = vsum / max(1, vn)
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "best_val": best_val,
                        "config": config,
                    },
                    out_dir / "best.pt",
                )

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val": best_val,
                "config": config,
            },
            out_dir / "last.pt",
        )

        msg = f"epoch {epoch+1} train L1={train_loss:.5f}"
        if val_loss is not None:
            msg += f" val L1={val_loss:.5f}"
        print(msg)

    print(f"Done. Checkpoints in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
