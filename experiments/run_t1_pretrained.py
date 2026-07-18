"""
run_t1_pretrained.py — T1: does the RGB~=RGB-D null hold with ImageNet-pretrained encoders?

Identical controlled protocol to run_study_local.py (same FIXED split seed=42,
loss Dice+BCE, AdamW lr1e-3/wd1e-4, 18 epochs, best-val-IoU model selection,
per-image test metrics + stratifiers). The ONLY differences vs the from-scratch
study are the reviewer-requested changes:
  * encoder_weights="imagenet" -- both arms use the SAME pretrained RGB weights.
  * 4-channel RGB-D: conv1 is inflated -- the three RGB filters are copied from
    the pretrained conv1, and the depth-channel filter is initialised as the
    MEAN of the three RGB filters (primary; DEPTH_INIT=zero also available).
    By default we do NOT rescale, so at init the RGB pathway is IDENTICAL to the
    pretrained RGB model (documented, keeps the comparison fair). RESCALE=1 scales
    conv1 by 3/4 to preserve activation magnitude.
  * ImageNet input normalisation is applied to the RGB channels (pretrained
    encoders expect it); the metric-depth 4th channel stays in [0,1].

Runs BOTH architectures (U-Net, DeepLabV3+) x {rgb, rgbd} x seeds, checkpointing
to results/ after every (arch, seed, mode) so a crash never loses work. Saves
per-image rows (for later paired t-test / TOST / stratification) and per-epoch
val-IoU curves (for T4 convergence).

*** DO NOT pre-commit to the conclusion. If depth helps more once the encoder is
pretrained, that overturns the paper's framing and must be reported honestly. ***
"""
import os, sys, time, json, subprocess
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "18"))
IMG = (int(os.environ.get("IMW", "256")), int(os.environ.get("IMH", "256")))
BS = int(os.environ.get("BS", "8"))
ENCODER = os.environ.get("ENCODER", "resnet18")
ARCHS = [a.strip().lower() for a in os.environ.get("ARCHS", "unet,deeplabv3plus").split(",")]
SEEDS = [int(s) for s in os.environ.get("SEEDS", "41,42,43").split(",")]
DEPTH_INIT = os.environ.get("DEPTH_INIT", "mean")   # mean | zero
RESCALE = os.environ.get("RESCALE", "0") == "1"     # scale conv1 by 3/4 if set
OUT = os.path.join(HERE, "..", "results", os.environ.get("OUT", "t1_pretrained.json"))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEVICE)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEVICE)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=HERE).decode().strip()
    except Exception:
        return "unknown"


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    import random; random.seed(s)


def norm_in(x):
    """ImageNet-normalise the RGB channels (0:3); leave the depth channel (3) as-is."""
    x = x.clone()
    x[:, :3] = (x[:, :3] - IMAGENET_MEAN) / IMAGENET_STD
    return x


def build_model(arch, in_ch):
    """Pretrained encoder; inflate conv1 to 4ch for RGB-D (mean-of-RGB depth init)."""
    arch_cls = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[arch]
    model = arch_cls(encoder_name=ENCODER, encoder_weights="imagenet",
                     in_channels=3, classes=1)
    if in_ch == 4:
        old = model.encoder.conv1          # resnet: Conv2d(3,64,7,2,3,bias=False)
        new = nn.Conv2d(4, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding,
                        bias=(old.bias is not None))
        with torch.no_grad():
            w = old.weight                 # [out,3,kh,kw]
            new.weight[:, :3] = w
            if DEPTH_INIT == "mean":
                new.weight[:, 3:4] = w.mean(dim=1, keepdim=True)
            else:
                new.weight[:, 3:4].zero_()
            if RESCALE:
                new.weight.mul_(3.0 / 4.0)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.encoder.conv1 = new
    return model.to(DEVICE)


def per_image_metrics(model, samples_split, mode):
    rows = []
    model.eval()
    with torch.no_grad():
        for s in samples_split:
            x = (D.load_rgbd(s.image_path, s.depth_path, IMG) if mode == "rgbd"
                 else D.load_rgb(s.image_path, IMG))
            y = D.rasterize_mask(s.label_path, IMG).astype(np.float32)
            xt = norm_in(torch.from_numpy(x).permute(2, 0, 1)[None].float().to(DEVICE))
            prob = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
            pred = (prob >= 0.5).astype(np.float32)
            tp = float((pred * y).sum()); fp = float((pred * (1 - y)).sum())
            fn = float(((1 - pred) * y).sum()); eps = 1e-7
            depth = D.load_depth(s.depth_path, IMG)
            in_mask = depth[y > 0]
            rows.append({
                "key": s.key,
                "IoU": tp / (tp + fp + fn + eps),
                "F1": 2 * tp / (2 * tp + fp + fn + eps),
                "Precision": tp / (tp + fp + eps),
                "Recall": tp / (tp + fn + eps),
                "gt_area_frac": float(y.mean()),
                "mean_depth": float(in_mask.mean()) if in_mask.size else float(depth.mean()),
            })
    return rows


def train_one(arch, mode, seed):
    set_seed(seed)
    samples = D.build_index(DATA_ROOT)
    split = (D.split_from_json(samples, os.environ["POTHRGBD_SPLIT"])
             if os.environ.get("POTHRGBD_SPLIT")            # scene-disjoint rerun (T3)
             else D.split_index(samples, seed=42))          # FIXED random split across seeds
    in_ch = 4 if mode == "rgbd" else 3
    DS = D.make_torch_dataset()
    tr = DataLoader(DS(split["train"], mode=mode, out_size=IMG, augment=True),
                    batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(DS(split["val"], mode=mode, out_size=IMG, augment=False),
                    batch_size=BS, shuffle=False, num_workers=0)
    model = build_model(arch, in_ch)
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
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
            best_iou = iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{arch} {mode} seed{seed}] ep{ep+1:02d}/{EPOCHS} valIoU={iou:.3f}", flush=True)
    model.load_state_dict(best_state)
    rows = per_image_metrics(model, split["test"], mode)
    test_mean = {k: float(np.mean([r[k] for r in rows]))
                 for k in ["IoU", "F1", "Precision", "Recall"]}
    print(f"  [{arch} {mode} seed{seed}] TEST mean {test_mean}", flush=True)
    return test_mean, rows, curve


def paired_t(deltas):
    d = np.asarray(deltas, dtype=np.float64); n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float(d.mean()), float("nan")
    return float(d.mean()), float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))


def main():
    print(f"device={DEVICE} archs={ARCHS} encoder={ENCODER} pretrained=imagenet "
          f"depth_init={DEPTH_INIT} rescale={RESCALE} epochs={EPOCHS} seeds={SEEDS}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = {"config": {"encoder": ENCODER, "pretrained": "imagenet",
                          "depth_init": DEPTH_INIT, "rescale": RESCALE,
                          "epochs": EPOCHS, "img": IMG, "seeds": SEEDS,
                          "loss": "dice+bce", "opt": "adamw lr1e-3 wd1e-4",
                          "git_hash": git_hash()},
               "runs": [], "per_image": {}, "curves": {}}
    t0 = time.time()
    for arch in ARCHS:
        for seed in SEEDS:
            for mode in ["rgb", "rgbd"]:
                print(f"\n=== {arch} | seed {seed} | {mode} ===", flush=True)
                mean, rows, curve = train_one(arch, mode, seed)
                results["runs"].append({"arch": arch, "seed": seed, "mode": mode, "test_mean": mean})
                results["per_image"][f"{arch}_{seed}_{mode}"] = rows
                results["curves"][f"{arch}_{seed}_{mode}"] = curve
                json.dump(results, open(OUT, "w"), indent=1)
                print(f"  saved -> {OUT}", flush=True)

    print("\n================  T1 SUMMARY (pretrained, mean+/-std over seeds, TEST)  ================", flush=True)
    for arch in ARCHS:
        def agg(mode, metric):
            vals = [r["test_mean"][metric] for r in results["runs"]
                    if r["mode"] == mode and r["arch"] == arch]
            return (np.mean(vals), np.std(vals)) if vals else (float("nan"), float("nan"))
        print(f"\n--- {arch} ---", flush=True)
        print(f"{'metric':11s}{'RGB':>18s}{'RGB-D':>18s}", flush=True)
        for m in ["IoU", "F1", "Precision", "Recall"]:
            rm, rs = agg("rgb", m); dm, ds = agg("rgbd", m)
            print(f"{m:11s}{rm:10.4f}+/-{rs:5.4f}{dm:10.4f}+/-{ds:5.4f}", flush=True)
        deltas = []
        for seed in SEEDS:
            rgb = {r["key"]: r["IoU"] for r in results["per_image"].get(f"{arch}_{seed}_rgb", [])}
            rgbd = {r["key"]: r["IoU"] for r in results["per_image"].get(f"{arch}_{seed}_rgbd", [])}
            for k in rgb.keys() & rgbd.keys():
                deltas.append(rgbd[k] - rgb[k])
        md, t = paired_t(deltas)
        print(f"Paired IoU delta (RGB-D - RGB): mean={md:+.4f}  t={t:.2f}  n={len(deltas)}", flush=True)
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
