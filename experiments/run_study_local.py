"""
run_study_local.py — Core study: "Is depth worth it for pothole segmentation?"

Controlled RGB vs RGB-D comparison on PothRGBD (the only dataset here with
genuine *metric* depth), run LOCALLY on Apple Silicon (MPS).

This script produces the credibility backbone of the paper:
  * MULTI-SEED robustness (default seeds 41,42,43) -> mean +/- std,
  * PER-IMAGE test metrics saved (IoU/F1/P/R) together with per-image
    pothole size (GT area fraction) and distance proxy (mean in-mask depth),
    so the stratified "when does depth help?" analysis needs no retraining,
  * a PAIRED comparison (same image, RGB vs RGB-D) per seed.

Architecture/loss/split are IDENTICAL across RGB and RGB-D; the ONLY difference
is the input channel count (3 vs 4). Results are appended to a JSON after every
(seed, mode) so a crash never loses completed work.
"""
import os, sys, time, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
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
ARCH = os.environ.get("ARCH", "unet").lower()
SEEDS = [int(s) for s in os.environ.get("SEEDS", "41,42,43").split(",")]
OUT = os.path.join(HERE, os.environ.get("OUT", "study_results.json"))


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)
    import random; random.seed(s)


def per_image_metrics(model, samples_split, mode):
    """Return a list of dicts, one per test image, with metrics + stratifiers."""
    DS = D.make_torch_dataset()
    rows = []
    model.eval()
    with torch.no_grad():
        for s in samples_split:
            x = (D.load_rgbd(s.image_path, s.depth_path, IMG) if mode == "rgbd"
                 else D.load_rgb(s.image_path, IMG))
            y = D.rasterize_mask(s.label_path, IMG).astype(np.float32)
            xt = torch.from_numpy(x).permute(2, 0, 1)[None].float().to(DEVICE)
            prob = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
            pred = (prob >= 0.5).astype(np.float32)
            tp = float((pred * y).sum()); fp = float((pred * (1 - y)).sum())
            fn = float(((1 - pred) * y).sum()); eps = 1e-7
            # distance proxy = mean metric depth (normalised) inside the GT mask
            depth = D.load_depth(s.depth_path, IMG)
            in_mask = depth[y > 0]
            rows.append({
                "key": s.key,
                "IoU": tp / (tp + fp + fn + eps),
                "F1": 2 * tp / (2 * tp + fp + fn + eps),
                "Precision": tp / (tp + fp + eps),
                "Recall": tp / (tp + fn + eps),
                "gt_area_frac": float(y.mean()),                       # size proxy
                "mean_depth": float(in_mask.mean()) if in_mask.size else float(depth.mean()),
            })
    return rows


def train_one(mode, seed):
    set_seed(seed)
    samples = D.build_index(DATA_ROOT)
    split = D.split_index(samples, seed=42)  # FIXED split across seeds (only init/shuffle vary)
    in_ch = 4 if mode == "rgbd" else 3
    DS = D.make_torch_dataset()
    tr = DataLoader(DS(split["train"], mode=mode, out_size=IMG, augment=True),
                    batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(DS(split["val"], mode=mode, out_size=IMG, augment=False),
                    batch_size=BS, shuffle=False, num_workers=0)
    arch_cls = {"unet": smp.Unet, "deeplabv3plus": smp.DeepLabV3Plus}[ARCH]
    model = arch_cls(encoder_name=ENCODER, encoder_weights=None, in_channels=in_ch, classes=1).to(DEVICE)
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_iou, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x); loss = dice(logits, y) + bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        # val IoU for model selection
        model.eval(); iou = 0.0; nb = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = (torch.sigmoid(model(x)) >= 0.5).float()
                tp = (p*y).sum().item(); fp = (p*(1-y)).sum().item(); fn = ((1-p)*y).sum().item()
                iou += tp/(tp+fp+fn+1e-7); nb += 1
        iou /= max(nb, 1)
        if iou > best_iou:
            best_iou = iou
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{mode} seed{seed}] ep{ep+1:02d}/{EPOCHS} valIoU={iou:.3f}", flush=True)
    model.load_state_dict(best_state)
    rows = per_image_metrics(model, split["test"], mode)
    test_mean = {k: float(np.mean([r[k] for r in rows]))
                 for k in ["IoU", "F1", "Precision", "Recall"]}
    print(f"  [{mode} seed{seed}] TEST mean {test_mean}", flush=True)
    return test_mean, rows


def paired_t(deltas):
    d = np.asarray(deltas, dtype=np.float64); n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return float(d.mean()), float("nan")
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    return float(d.mean()), float(t)


def main():
    print(f"device={DEVICE} arch={ARCH} encoder={ENCODER} epochs={EPOCHS} seeds={SEEDS}", flush=True)
    results = {"config": {"encoder": ENCODER, "epochs": EPOCHS, "img": IMG, "seeds": SEEDS},
               "runs": [], "per_image": {}}
    t0 = time.time()
    for seed in SEEDS:
        for mode in ["rgb", "rgbd"]:
            print(f"\n=== seed {seed} | {mode} ===", flush=True)
            mean, rows = train_one(mode, seed)
            results["runs"].append({"seed": seed, "mode": mode, "test_mean": mean})
            results["per_image"][f"{seed}_{mode}"] = rows
            json.dump(results, open(OUT, "w"), indent=1)  # checkpoint after each model
            print(f"  saved -> {OUT}", flush=True)

    # ---- summary: mean +/- std over seeds, and paired per-image deltas ----
    def agg(mode, metric):
        vals = [r["test_mean"][metric] for r in results["runs"] if r["mode"] == mode]
        return np.mean(vals), np.std(vals)
    print("\n================  SUMMARY (mean +/- std over seeds, TEST)  ================", flush=True)
    print(f"{'metric':11s}{'RGB':>16s}{'RGB-D':>16s}", flush=True)
    for m in ["IoU", "F1", "Precision", "Recall"]:
        rm, rs = agg("rgb", m); dm, ds = agg("rgbd", m)
        print(f"{m:11s}{rm:8.3f}+/-{rs:4.3f}{dm:9.3f}+/-{ds:4.3f}", flush=True)

    # paired per-image IoU deltas (RGB-D minus RGB), pooled across seeds by key
    deltas = []
    for seed in SEEDS:
        rgb = {r["key"]: r["IoU"] for r in results["per_image"].get(f"{seed}_rgb", [])}
        rgbd = {r["key"]: r["IoU"] for r in results["per_image"].get(f"{seed}_rgbd", [])}
        for k in rgb.keys() & rgbd.keys():
            deltas.append(rgbd[k] - rgb[k])
    md, t = paired_t(deltas)
    print("-" * 60, flush=True)
    print(f"Per-image paired IoU delta (RGB-D - RGB): mean={md:+.4f}  t={t:.2f}  n={len(deltas)}", flush=True)
    print("Interpretation: |t| small / mean ~0  => depth gives no significant gain.", flush=True)
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
