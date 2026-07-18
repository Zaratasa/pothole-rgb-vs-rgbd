"""
fig_qualitative.py — qualitative RGB vs RGB-D pothole-segmentation examples.

Loads the two probe checkpoints (ckpt_rgb.pt, ckpt_rgbd.pt), runs them on a few
held-out PothRGBD test images, and saves a grid figure:
columns = [RGB input, ground truth, RGB prediction, RGB-D prediction],
one row per example. Illustrates that the two modalities produce visually
near-identical masks (they trade precision for recall, not overlap).
"""
import os, sys
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
IMG = (256, 256)
OUT = os.path.join(HERE, "..", "figures", "qualitative.pdf")


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=ck["in_ch"], classes=1)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


def predict(model, x):
    xt = torch.from_numpy(x).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        p = torch.sigmoid(model(xt))[0, 0].numpy()
    return (p >= 0.5).astype(np.float32)


def main():
    m_rgb = load_model(os.path.join(HERE, "ckpt_rgb.pt"))
    m_rgbd = load_model(os.path.join(HERE, "ckpt_rgbd.pt"))
    samples = D.build_index(DATA_ROOT)
    split = D.split_index(samples, seed=42)
    test = split["test"]

    # rank test images by GT pothole area; pick a spread (small, medium, large)
    areas = []
    for s in test:
        y = D.rasterize_mask(s.label_path, IMG)
        areas.append((float(y.mean()), s))
    areas.sort(key=lambda t: t[0])
    n = len(areas)
    picks = [areas[int(n * f)][1] for f in (0.35, 0.6, 0.8, 0.93)]  # avoid empties/extremes

    rows = len(picks)
    fig, ax = plt.subplots(rows, 4, figsize=(8, 2.1 * rows))
    col_titles = ["RGB input", "Ground truth", "RGB prediction", "RGB-D prediction"]
    for j, t in enumerate(col_titles):
        ax[0, j].set_title(t, fontsize=10)
    for i, s in enumerate(picks):
        rgb = D.load_rgb(s.image_path, IMG)
        rgbd = D.load_rgbd(s.image_path, s.depth_path, IMG)
        gt = D.rasterize_mask(s.label_path, IMG)
        pr = predict(m_rgb, rgb)
        pd = predict(m_rgbd, rgbd)
        for j, im in enumerate([rgb, gt, pr, pd]):
            if j == 0:
                ax[i, j].imshow(np.clip(im, 0, 1))
            else:
                ax[i, j].imshow(im, cmap="gray", vmin=0, vmax=1)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=150)
    print(f"saved -> {OUT}  ({rows} examples)")


if __name__ == "__main__":
    main()
