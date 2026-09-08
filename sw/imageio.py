# ABOUTME: Image loading, DA-V2 preprocessing, and calibration-set construction.
# ABOUTME: Also renders depth maps to PNG so emulator output can be eyeballed, not just scored.

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def load_image(path: Path | str, size: int) -> np.ndarray:
    """Load an image as a preprocessed [1,3,size,size] tensor.

    The reference implementation preserves aspect ratio and rounds each side to
    a multiple of 14. This graph is compiled for one fixed square resolution, so
    we resize to a square instead; that changes framing, not correctness.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def find_images(directory: Path | str) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory} is not a directory")
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def synthetic_image(size: int, rng: np.random.Generator) -> np.ndarray:
    """Band-limited 1/f noise, as a stand-in when no real images are available.

    Natural images have roughly 1/f spatial spectra; white noise does not, and
    calibrating on white noise produces activation ranges that no real photo
    would ever reach. This is still a poor substitute for real data and the
    callers say so out loud.
    """
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs, indexing="ij")
    radial = np.sqrt(fx**2 + fy**2)
    radial[0, 0] = 1.0 / size
    envelope = 1.0 / radial
    planes = []
    for _ in range(3):
        spectrum = np.fft.fft2(rng.standard_normal((size, size))) * envelope
        plane = np.real(np.fft.ifft2(spectrum))
        plane = (plane - plane.min()) / max(float(np.ptp(plane)), 1e-6)
        planes.append(plane)
    arr = np.stack(planes, axis=-1).astype(np.float32)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)[None].astype(np.float32)


def calibration_batches(size: int, *, images: Sequence[Path] | None = None,
                        count: int = 8, seed: int = 0) -> Iterator[dict[str, np.ndarray]]:
    """Yield ``{"image": tensor}`` batches for the calibration pass."""
    if images:
        for path in list(images)[:count]:
            yield {"image": load_image(path, size)}
        return
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield {"image": synthetic_image(size, rng)}


def save_depth_png(depth: np.ndarray, path: Path | str) -> None:
    """Write a min-max normalized inverse-depth visualization."""
    from PIL import Image

    d = np.asarray(depth, dtype=np.float32).squeeze()
    if d.ndim != 2:
        raise ValueError(f"expected a 2-D depth map, got shape {depth.shape}")
    lo, hi = float(d.min()), float(d.max())
    norm = np.zeros_like(d) if hi <= lo else (d - lo) / (hi - lo)
    Image.fromarray((norm * 255).astype(np.uint8)).save(path)
