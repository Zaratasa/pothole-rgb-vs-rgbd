"""
run_t2_normals.py — T2: does a STRONGER depth encoding (surface normals) change the answer?

Reviewer concern: early fusion of RAW metric depth is the weakest way to use depth;
normal-encoded depth is reported to help much more (cited work ~+8 IoU). So we give
depth its best shot: a 6-channel RGB+N input (3 colour + 3 surface-normal channels).

Surface normals are approximated from the metric-depth gradients (Sobel/np.gradient) —
the D415 intrinsics are not shipped with PothRGBD, so we use the documented gradient
fallback: n ~ normalize(-dz/dx, -dz/dy, s), with s=NORMAL_SCALE mm/px controlling the
z balance. Holes are median-filled BEFORE normals (load_depth already does this), so
the comparison stays controlled. Normals are mapped [-1,1]->[0,1] for input.

Protocol is otherwise IDENTICAL to T1 (pretrained encoder, conv1 inflated — RGB filters
copied, extra channels = mean of the 3 RGB filters; ImageNet-normalised RGB channels;
Dice+BCE, AdamW, 18 ep, best-val-IoU). Runs {rgb, rgbn} x seeds so the pairing is
self-contained. DO NOT pre-commit to the conclusion.
"""
import os, sys, time, json, subprocess, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "18"))
IMG = (int(os.environ.get("IMW", "256")), int(os.environ.get("IMH", "256")))
BS = int(os.environ.get("BS", "8"))
ENCODER = os.environ.get("ENCODER", "resnet18")
ARCHS = [a.strip().lower() for a in os.environ.get("ARCHS", "unet").split(",")]
MODES = [m.strip().lower() for m in os.environ.get("MODES", "rgb,rgbn").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "41,42,43").split(",")]
NORMAL_SCALE = float(os.environ.get("NORMAL_SCALE", "10.0"))   # mm/px z-balance (validated non-degenerate; smaller=richer)
OUT = os.path.join(HERE, "..", "results", os.environ.get("OUT", "t2_normals.json"))
IN_CH = {"rgb": 3, "rgbn": 6}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return "unknown"


def compute_normals(depth_path, out_size, scale=NORMAL_SCALE):
    """Surface normals HxWx3 in [0,1] from metric-depth gradients (hole-filled)."""
    d_mm = D.load_depth(depth_path, out_size).astype(np.float32) * 4000.0   # back to mm
    gy, gx = np.gradient(d_mm)
    nz = np.full_like(d_mm, scale)
    n = np.stack([-gx, -gy, nz], axis=-1)
    n = n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-7)
    return ((n + 1.0) * 0.5).astype(np.float32)


def load_input(mode, s, out_size):
    rgb = D.load_rgb(s.image_path, out_size)
    if mode == "rgb":
        return rgb
    return np.concatenate([rgb, compute_normals(s.depth_path, out_size)], axis=-1)


class DS(Dataset):
    def __init__(self, samples, mode, out_size, augment):
        self.samples, self.mode, self.out_size, self.augment = samples, mode, out_size, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = load_input(self.mode, s, self.out_size)
        y = D.rasterize_mask(s.label_path, self.out_size)[..., None].astype(np.float32)
        if self.augment and random.random() < 0.5:
            x = x[:, ::-1, :].copy(); y = y[:, ::-1, :].copy()
        return (torch.from_numpy(x).permute(2, 0, 1).float(),
                torch.from_numpy(y).permute(2, 0, 1).float())


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def norm_in(x):
    x = x.clone()
    x[:, :3] = (x[:, :3] - IMAGENET_MEAN) / IMAGENET_STD
    return x


def build_model(arch, in_ch):
    arch_cls = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[arch]
    model = arch_cls(encoder_name=ENCODER, encoder_weights="imagenet", in_channels=3, classes=1)
    if in_ch != 3:
        old = model.encoder.conv1
        new = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=(old.bias is not None))
        with torch.no_grad():
            w = old.weight
            new.weight[:, :3] = w
            new.weight[:, 3:] = w.mean(dim=1, keepdim=True)   # extra channels = mean of RGB
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.encoder.conv1 = new
    return model.to(DEVICE)


def per_image_metrics(model, split, mode):
    rows = []; model.eval()
    with torch.no_grad():
        for s in split:
            x = load_input(mode, s, IMG)
            y = D.rasterize_mask(s.label_path, IMG).astype(np.float32)
            xt = norm_in(torch.from_numpy(x).permute(2, 0, 1)[None].float().to(DEVICE))
            pred = (torch.sigmoid(model(xt))[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
            tp = float((pred * y).sum()); fp = float((pred * (1 - y)).sum())
            fn = float(((1 - pred) * y).sum()); eps = 1e-7
            depth = D.load_depth(s.depth_path, IMG); in_mask = depth[y > 0]
            rows.append({"key": s.key, "IoU": tp / (tp + fp + fn + eps),
                         "F1": 2 * tp / (2 * tp + fp + fn + eps),
                         "Precision": tp / (tp + fp + eps), "Recall": tp / (tp + fn + eps),
                         "gt_area_frac": float(y.mean()),
                         "mean_depth": float(in_mask.mean()) if in_mask.size else float(depth.mean())})
    return rows


def train_one(arch, mode, seed):
    set_seed(seed)
    samples = D.build_index(DATA_ROOT)
    split = (D.split_from_json(samples, os.environ["POTHRGBD_SPLIT"])
             if os.environ.get("POTHRGBD_SPLIT") else D.split_index(samples, seed=42))
    tr = DataLoader(DS(split["train"], mode, IMG, True), batch_size=BS, shuffle=True, drop_last=True)
    va = DataLoader(DS(split["val"], mode, IMG, False), batch_size=BS, shuffle=False)
    model = build_model(arch, IN_CH[mode])
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True); bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_iou, best_state, curve = -1.0, None, []
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr:
            x, y = norm_in(x.to(DEVICE)), y.to(DEVICE)
            logits = model(x); loss = dice(logits, y) + bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); iou = 0.0; nb = 0
        with torch.no_grad():
            for x, y in va:
                x, y = norm_in(x.to(DEVICE)), y.to(DEVICE)
                p = (torch.sigmoid(model(x)) >= 0.5).float()
                tp = (p*y).sum().item(); fp = (p*(1-y)).sum().item(); fn = ((1-p)*y).sum().item()
                iou += tp/(tp+fp+fn+1e-7); nb += 1
        iou /= max(nb, 1); curve.append(iou)
        if iou > best_iou:
            best_iou = iou; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{arch} {mode} seed{seed}] ep{ep+1:02d}/{EPOCHS} valIoU={iou:.3f}", flush=True)
    model.load_state_dict(best_state)
    rows = per_image_metrics(model, split["test"], mode)
    tm = {k: float(np.mean([r[k] for r in rows])) for k in ["IoU", "F1", "Precision", "Recall"]}
    print(f"  [{arch} {mode} seed{seed}] TEST mean {tm}", flush=True)
    return tm, rows, curve


def main():
    print(f"device={DEVICE} archs={ARCHS} modes={MODES} normal_scale={NORMAL_SCALE} "
          f"epochs={EPOCHS} seeds={SEEDS} split={'disjoint' if os.environ.get('POTHRGBD_SPLIT') else 'random42'}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = {"config": {"encoder": ENCODER, "pretrained": "imagenet", "normal_scale": NORMAL_SCALE,
                          "epochs": EPOCHS, "img": IMG, "seeds": SEEDS, "modes": MODES,
                          "git_hash": git_hash()}, "runs": [], "per_image": {}, "curves": {}}
    for arch in ARCHS:
        for seed in SEEDS:
            for mode in MODES:
                print(f"\n=== {arch} | seed {seed} | {mode} ===", flush=True)
                tm, rows, curve = train_one(arch, mode, seed)
                results["runs"].append({"arch": arch, "seed": seed, "mode": mode, "test_mean": tm})
                results["per_image"][f"{arch}_{seed}_{mode}"] = rows
                results["curves"][f"{arch}_{seed}_{mode}"] = curve
                json.dump(results, open(OUT, "w"), indent=1)
                print(f"  saved -> {OUT}", flush=True)
    print("\n(analyse with: python stats/analyze.py results/t2_normals.json ; rgbn vs rgb)", flush=True)
    print("Total done.", flush=True)


if __name__ == "__main__":
    main()
