from __future__ import annotations
import argparse
import os
import random
import shutil
import sys
from pathlib import Path
import cv2
import numpy as np

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create degraded copies of an image dataset.")
    p.add_argument("--src", type=Path, required=True, help="Root folder of clean images (date subfolders, etc.)")
    p.add_argument("--dst", type=Path, required=True, help="Output root; structure mirrors --src")
    p.add_argument(
        "--pattern",
        default="*bag.bmp.jpg",
        help="Glob relative to each folder (default: *bag.bmp.jpg)",
    )
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument(
        "--intensity",
        type=float,
        default=0.65,
        help="Overall degradation strength in [0, 1] (default: 0.65)",
    )
    p.add_argument(
        "--copy-nonimages",
        action="store_true",
        help="Copy files that do not match --pattern into the same relative paths",
    )
    p.add_argument("--dry-run", action="store_true", help="Print actions only, do not write files")
    return p.parse_args()


def clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _rand(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def add_gaussian_noise_bgr(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = rng.normal(0.0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_salt_pepper(img: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    if amount <= 0:
        return img
    out = img.copy()
    n = int(amount * img.size / 3)
    h, w = img.shape[:2]
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    salt = rng.random(n) > 0.5
    out[ys[salt], xs[salt]] = 255
    out[ys[~salt], xs[~salt]] = 0
    return out


def add_motion_blur_bgr(img: np.ndarray, size: int, angle_deg: float) -> np.ndarray:
    if size < 3:
        return img
    size = size | 1
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    kernel /= kernel.sum()
    center = (size / 2 - 0.5, size / 2 - 0.5)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    k = cv2.warpAffine(kernel, M, (size, size))
    k /= k.sum() + 1e-8
    return cv2.filter2D(img, -1, k)


def add_gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize < 3:
        return img
    ksize = ksize | 1
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_fog_haze(img: np.ndarray, strength: float, rng: random.Random) -> np.ndarray:
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    tint = np.array(
        [
            _rand(rng, 200, 255),
            _rand(rng, 200, 255),
            _rand(rng, 200, 255),
        ],
        dtype=np.float32,
    )
    fog = np.full((h, w, 3), tint, dtype=np.float32)
    alpha = strength
    out = img.astype(np.float32) * (1.0 - alpha) + fog * alpha
    gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray3 = np.stack([gray, gray, gray], axis=-1)
    mix = 0.15 * strength
    out = out * (1.0 - mix) + gray3 * mix
    return np.clip(out, 0, 255).astype(np.uint8)


def add_lens_droplets(
    img: np.ndarray,
    count: int,
    rng: random.Random,
) -> np.ndarray:
    if count <= 0:
        return img
    h, w = img.shape[:2]
    overlay = img.astype(np.float32)
    for _ in range(count):
        cx = int(rng.random() * w)
        cy = int(rng.random() * h)
        ax = int(_rand(rng, w * 0.02, w * 0.08))
        ay = int(_rand(rng, h * 0.02, h * 0.08))
        angle = _rand(rng, 0, 180)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(
            mask,
            (cx, cy),
            (max(ax, 2), max(ay, 2)),
            angle,
            0,
            360,
            1.0,
            -1,
        )
        k = int(_rand(rng, 15, 45)) | 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        darken = _rand(rng, 0.75, 0.92)
        brighten = _rand(rng, 1.05, 1.18)
        m = mask[..., None]
        droplet = overlay * (1.0 - m * (1.0 - darken))
        sx = int(cx + _rand(rng, -ax * 0.3, ax * 0.3))
        sy = int(cy + _rand(rng, -ay * 0.3, ay * 0.3))
        spot = np.zeros((h, w), dtype=np.float32)
        r = max(2, int(min(ax, ay) * _rand(rng, 0.15, 0.35)))
        cv2.circle(spot, (sx, sy), r, 1.0, -1)
        spot = cv2.GaussianBlur(spot, (0, 0), sigmaX=r * 0.4)
        spot3 = spot[..., None]
        droplet = droplet * (1.0 - spot3) + droplet * spot3 * brighten
        overlay = droplet

    if rng.random() < 0.4:
        b, g, rch = cv2.split(overlay.astype(np.uint8))
        dx = int(_rand(rng, -2, 2))
        dy = int(_rand(rng, -2, 2))
        b = np.roll(b, dx, axis=1)
        rch = np.roll(rch, -dx, axis=1)
        overlay = cv2.merge([b, g, rch]).astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def add_dust_particles(img: np.ndarray, num: int, rng: random.Random) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(num):
        x = int(rng.random() * w)
        y = int(rng.random() * h)
        length = int(_rand(rng, 3, max(4, w // 80)))
        angle = _rand(rng, 0, 360)
        rad = np.deg2rad(angle)
        x2 = int(x + length * np.cos(rad))
        y2 = int(y + length * np.sin(rad))
        shade = int(_rand(rng, 180, 255))
        thickness = int(_rand(rng, 1, 2))
        cv2.line(out, (x, y), (x2, y2), (shade, shade, shade), thickness, cv2.LINE_AA)
        if rng.random() < 0.3:
            cv2.circle(out, (x, y), int(_rand(rng, 1, 3)), (shade, shade, shade), -1, cv2.LINE_AA)
    return out


def jpeg_degrade(img: np.ndarray, quality: int) -> np.ndarray:
    q = int(np.clip(quality, 10, 95))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def random_exposure(img: np.ndarray, gamma: float, brightness: float) -> np.ndarray:
    x = img.astype(np.float32) / 255.0
    x = np.clip(x * brightness, 0, 1)
    x = np.power(x, gamma)
    return (np.clip(x, 0, 1) * 255.0).astype(np.uint8)


def degrade_image(img: np.ndarray, intensity: float, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    t = clamp01(intensity)
    out = img

    if rng.random() < 0.55 + 0.25 * t:
        out = add_fog_haze(out, strength=_rand(rng, 0.05, 0.22) * (0.5 + t), rng=rng)

    if rng.random() < 0.45 + 0.2 * t:
        if rng.random() < 0.55:
            k = int(_rand(rng, 3, 7 + int(5 * t))) | 1
            out = add_gaussian_blur(out, k)
        else:
            msize = int(_rand(rng, 5, 9 + int(10 * t))) | 1
            out = add_motion_blur_bgr(out, msize, angle_deg=_rand(rng, 0, 180))

    if rng.random() < 0.5 + 0.25 * t:
        n = int(_rand(rng, 2, 8 + int(12 * t)))
        out = add_lens_droplets(out, count=n, rng=rng)

    if rng.random() < 0.55 + 0.2 * t:
        particles = int(_rand(rng, 20, 120 + int(200 * t)))
        out = add_dust_particles(out, num=particles, rng=rng)

    if rng.random() < 0.65:
        sigma = _rand(rng, 2, 12 + 18 * t)
        out = add_gaussian_noise_bgr(out, sigma=sigma, rng=np_rng)
    if rng.random() < 0.35 + 0.25 * t:
        out = add_salt_pepper(out, amount=_rand(rng, 0.001, 0.012) * (0.5 + t), rng=np_rng)

    if rng.random() < 0.4:
        g = _rand(rng, 0.75, 1.35)
        b = _rand(rng, 0.85, 1.15)
        out = random_exposure(out, gamma=g, brightness=b)

    if rng.random() < 0.55 + 0.25 * t:
        q = int(_rand(rng, 35, 92 - int(35 * t)))
        out = jpeg_degrade(out, quality=max(15, q))

    if rng.random() < 0.25 * t:
        out = add_gaussian_blur(out, 3)

    return out


def ensure_dir(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if not args.src.is_dir():
        print(f"Source is not a directory: {args.src}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    np_seed = args.seed if args.seed is not None else None
    np_rng = np.random.default_rng(np_seed)

    t = clamp01(args.intensity)
    n_files = 0
    n_copied = 0

    for root, _dirnames, filenames in os.walk(args.src, topdown=True):
        dirpath = Path(root)
        rel = dirpath.relative_to(args.src)
        out_dir = args.dst / rel
        ensure_dir(out_dir, args.dry_run)

        matched = list(dirpath.glob(args.pattern))
        matched_set = {p.name for p in matched}

        for path in matched:
            n_files += 1
            dst_path = out_dir / path.name
            if args.dry_run:
                print(f"would degrade: {path} -> {dst_path}")
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"skip (unreadable): {path}", file=sys.stderr)
                continue
            dirty = degrade_image(img, intensity=t, rng=rng, np_rng=np_rng)
            cv2.imwrite(str(dst_path), dirty)
            n_copied += 1

        if args.copy_nonimages:
            for name in filenames:
                if name in matched_set:
                    continue
                src_file = dirpath / name
                if not src_file.is_file():
                    continue
                dst_file = out_dir / name
                if args.dry_run:
                    print(f"would copy: {src_file} -> {dst_file}")
                    continue
                shutil.copy2(src_file, dst_file)
                n_copied += 1

    if args.dry_run:
        print(f"Dry run: {n_files} image(s) matched '{args.pattern}'.")
    else:
        print(f"Done. Wrote {n_copied} file(s) under {args.dst} ({n_files} images matched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

