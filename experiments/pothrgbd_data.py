"""
pothrgbd_data.py — Shared data utilities for the PothRGBD depth-distillation experiments.

This module is imported by BOTH notebooks (probe + distill) so the loader,
pairing logic and polygon->mask rasterization live in exactly one place and
are unit-smoke-tested once.

Design notes (verified against the actual dataset, 2026-06):
  - images/  : 1000 .jpg, RGB, 640x480 (W x H). Roboflow basename, e.g.
               20250227_135438_color_png.rf.<hash>.jpg
  - depths/  : 1000 .npy, uint16, shape (480, 640) = (H, W), millimetres,
               raw RealSense D415 metric depth, range ~0..1470 mm.
               ~6% of pixels are 0 == invalid/hole (no return). Handle those.
  - labels/  : 1000 .txt, YOLO polygon-segmentation format. One line per
               instance: "class xn1 yn1 xn2 yn2 ...", coords normalised to
               [0,1] w.r.t. (W, H). Only class 0 (pothole) exists.
               4 label files are empty == valid background/negative images.

  PAIRING:
    - image <-> label : SAME basename (image.jpg  -> image.txt).
    - image <-> depth : by the leading timestamp prefix YYYYMMDD_HHMMSS,
                        because depth files are named <prefix>_depth.npy while
                        images carry the Roboflow "_color_png.rf.<hash>" suffix.
    - 998 unique timestamp prefixes (2 prefixes appear twice as distinct
      Roboflow augmentations sharing one depth map). We key the index by the
      FULL image basename so every image/label pair is kept; both members of a
      duplicate-prefix pair legitimately point at the same depth map.

This file has NO hard dependency on torch: the pure loading + rasterization
path uses only numpy + PIL so it can be smoke-tested on a CPU-only box.
The torch `Dataset` wrapper is defined lazily and only used in Colab.
"""

from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

# Image geometry is fixed for this dataset.
IMG_W, IMG_H = 640, 480

_TS_RE = re.compile(r"(\d{8}_\d{6})")


def timestamp_prefix(filename: str) -> Optional[str]:
    """Return the leading YYYYMMDD_HHMMSS timestamp of a filename, or None."""
    m = _TS_RE.match(os.path.basename(filename))
    return m.group(1) if m else None


@dataclass(frozen=True)
class Sample:
    """One fully-paired example."""
    key: str          # unique key == image basename (no extension)
    image_path: str
    depth_path: str
    label_path: str


def build_index(root: str) -> list[Sample]:
    """
    Scan <root>/{images,depths,labels} and return the list of fully-paired
    Samples. Skips any image that lacks a matching label or depth map.

    Parameters
    ----------
    root : str
        Dataset root containing the images/ depths/ labels/ subfolders.
    """
    img_dir = os.path.join(root, "images")
    dep_dir = os.path.join(root, "depths")
    lab_dir = os.path.join(root, "labels")
    for d in (img_dir, dep_dir, lab_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Expected dataset subfolder missing: {d}")

    # depth lookup: timestamp prefix -> depth file
    depth_by_ts: dict[str, str] = {}
    for f in os.listdir(dep_dir):
        if not f.endswith(".npy"):
            continue
        ts = timestamp_prefix(f)
        if ts is not None:
            depth_by_ts[ts] = os.path.join(dep_dir, f)

    label_names = set(os.listdir(lab_dir))

    samples: list[Sample] = []
    n_no_depth = n_no_label = 0
    for f in sorted(os.listdir(img_dir)):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        base = os.path.splitext(f)[0]
        lab_name = base + ".txt"
        if lab_name not in label_names:
            n_no_label += 1
            continue
        ts = timestamp_prefix(f)
        dep_path = depth_by_ts.get(ts) if ts else None
        if dep_path is None:
            n_no_depth += 1
            continue
        samples.append(
            Sample(
                key=base,
                image_path=os.path.join(img_dir, f),
                depth_path=dep_path,
                label_path=os.path.join(lab_dir, lab_name),
            )
        )
    if n_no_depth or n_no_label:
        print(f"[build_index] skipped {n_no_label} without label, "
              f"{n_no_depth} without depth.")
    return samples


def split_index(
    samples: list[Sample],
    ratios=(0.70, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[Sample]]:
    """
    Deterministic train/val/test split. Seed is FIXED so every run and both
    notebooks see the same partition (required for fair baseline-vs-distill
    comparison and for the >=3-run protocol in blueprint section 6).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    items = sorted(samples, key=lambda s: s.key)  # stable order before shuffle
    rng = random.Random(seed)
    rng.shuffle(items)
    n = len(items)
    n_tr = int(round(ratios[0] * n))
    n_va = int(round(ratios[1] * n))
    return {
        "train": items[:n_tr],
        "val": items[n_tr:n_tr + n_va],
        "test": items[n_tr + n_va:],
    }


def polygons_from_label(label_path: str) -> list[list[tuple[float, float]]]:
    """
    Parse a YOLO polygon-seg .txt into a list of polygons in PIXEL coords.
    Each polygon is a list of (x, y) tuples. class id is ignored (single class).
    Returns [] for an empty (background) file.
    """
    polys: list[list[tuple[float, float]]] = []
    with open(label_path, "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 7:   # class + >=3 (x,y) pairs minimum
                continue
            coords = list(map(float, parts[1:]))
            pts = [
                (coords[i] * IMG_W, coords[i + 1] * IMG_H)
                for i in range(0, len(coords) - 1, 2)
            ]
            if len(pts) >= 3:
                polys.append(pts)
    return polys


def rasterize_mask(label_path: str, out_size=(IMG_W, IMG_H)) -> np.ndarray:
    """
    Rasterize all polygons of a label file into a binary {0,1} uint8 mask.

    out_size : (W, H). The mask is drawn at native resolution then resized
    with NEAREST so it stays binary.
    """
    mask = Image.new("L", (IMG_W, IMG_H), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons_from_label(label_path):
        draw.polygon(poly, fill=1)
    if out_size != (IMG_W, IMG_H):
        mask = mask.resize(out_size, Image.NEAREST)
    return np.asarray(mask, dtype=np.uint8)


def split_from_json(samples, path):
    """Load an external {train,val,test: [keys]} split (e.g. the scene-disjoint
    split from run_t3_build_split.py) and map keys back to Sample objects, mirroring
    the dict returned by split_index()."""
    import json as _json
    by_key = {s.key: s for s in samples}
    spec = _json.load(open(path))
    return {name: [by_key[k] for k in spec[name] if k in by_key]
            for name in ("train", "val", "test")}


def load_depth(
    depth_path: str,
    out_size=(IMG_W, IMG_H),
    max_mm: float = 4000.0,
    fill_invalid: bool = True,
) -> np.ndarray:
    """
    Load a RealSense depth .npy (uint16 mm) and return a float32 array in [0,1].

    Steps:
      1. read raw mm (H, W).
      2. optionally inpaint zero/invalid pixels with the per-image median of
         valid depth (cheap, avoids feeding hard 0-cliffs that confuse a CNN).
      3. normalise by `max_mm` and clip to [0,1] (metric, dataset-agnostic
         scaling — NOT min/max per image, so the depth channel keeps an
         absolute metric meaning across samples).
      4. resize (BILINEAR) to out_size = (W, H).

    Returns (H_out, W_out) float32 in [0,1].
    """
    depth = np.load(depth_path).astype(np.float32)  # (H, W) mm
    if fill_invalid:
        valid = depth > 0
        if valid.any():
            depth = np.where(valid, depth, float(np.median(depth[valid])))
    depth = np.clip(depth / max_mm, 0.0, 1.0)
    if out_size != (depth.shape[1], depth.shape[0]):
        depth = np.asarray(
            Image.fromarray(depth).resize(out_size, Image.BILINEAR),
            dtype=np.float32,
        )
    return depth


def load_rgb(image_path: str, out_size=(IMG_W, IMG_H)) -> np.ndarray:
    """Load an RGB image as float32 HxWx3 in [0,1] at out_size=(W,H)."""
    im = Image.open(image_path).convert("RGB")
    if im.size != out_size:
        im = im.resize(out_size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def load_rgbd(image_path: str, depth_path: str, out_size=(IMG_W, IMG_H),
              **depth_kw) -> np.ndarray:
    """
    Return an early-fusion RGB-D tensor HxWx4 (float32, [0,1]):
    channels 0..2 = RGB, channel 3 = normalised depth.
    This is the teacher / RGB-D probe input.
    """
    rgb = load_rgb(image_path, out_size)
    depth = load_depth(depth_path, out_size, **depth_kw)[..., None]
    return np.concatenate([rgb, depth], axis=-1)


# --------------------------------------------------------------------------- #
# Lazy torch Dataset (only imported/used in Colab where torch is installed).   #
# --------------------------------------------------------------------------- #
def make_torch_dataset():
    """
    Factory that returns a `PothRGBDDataset` class bound to torch.
    Kept lazy so this module imports cleanly on a torch-less CPU box for the
    loader smoke test. Call once inside Colab:

        PothRGBDDataset = make_torch_dataset()
        ds = PothRGBDDataset(samples, mode="rgbd")
    """
    import torch
    from torch.utils.data import Dataset

    class PothRGBDDataset(Dataset):
        """
        mode = "rgb"  -> x: (3,H,W),  y: (1,H,W) mask   (student / baseline)
        mode = "rgbd" -> x: (4,H,W),  y: (1,H,W) mask   (teacher / RGB-D probe)
        """

        def __init__(self, samples: list[Sample], mode: str = "rgb",
                     out_size=(IMG_W, IMG_H), augment: bool = False):
            assert mode in ("rgb", "rgbd")
            self.samples = samples
            self.mode = mode
            self.out_size = out_size
            self.augment = augment

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            if self.mode == "rgbd":
                x = load_rgbd(s.image_path, s.depth_path, self.out_size)
            else:
                x = load_rgb(s.image_path, self.out_size)
            y = rasterize_mask(s.label_path, self.out_size)[..., None]

            # simple, mask-consistent horizontal flip augmentation
            if self.augment and random.random() < 0.5:
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1, :].copy()

            x = torch.from_numpy(x).permute(2, 0, 1).float()      # C,H,W
            y = torch.from_numpy(y).permute(2, 0, 1).float()      # 1,H,W
            return x, y

    return PothRGBDDataset


if __name__ == "__main__":
    # Allow `python pothrgbd_data.py <root>` as a quick CLI smoke test.
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    idx = build_index(root)
    print(f"Indexed {len(idx)} paired samples.")
    sp = split_index(idx)
    print({k: len(v) for k, v in sp.items()})
