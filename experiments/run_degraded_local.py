"""
run_degraded_local.py — "Is depth MORE useful when RGB degrades?"

Reviewer-objection experiment (generalisation): train a 3-ch RGB U-Net and a
4-ch RGB-D U-Net with the IDENTICAL recipe of run_study_local.py
(U-Net resnet18, Dice+BCE, AdamW 1e-3, img 256, 18 epoch, split seed=42).
Then evaluate BOTH on the TEST split under four conditions:

    clean  : original RGB
    dark   : RGB * 0.4 (under-exposure)
    blur   : moderate Gaussian blur on RGB
    noise  : additive Gaussian noise sigma=0.1 on RGB in [0,1]

CRITICAL: degradation is applied to the RGB CHANNELS ONLY. The depth channel
of the RGB-D input stays CLEAN. This tests whether a clean depth modality
"rescues" the model when the photometric (RGB) modality is corrupted.

Deliverables:
  - experiments/ckpt_rgb.pt , experiments/ckpt_rgbd.pt   (best val-IoU weights)
  - experiments/degraded_results.json
  - figures/degraded.pdf  (Delta IoU vs degradation condition, vector PDF)

Hypothesis: Delta IoU = IoU(RGB-D) - IoU(RGB) GROWS as RGB degrades more.
"""
import os, sys, time, json, argparse, random

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from PIL import Image, ImageFilter
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_FIG = os.path.abspath(os.path.join(HERE, "..", "..", "paper", "figures"))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
IMG = (256, 256)
BS = 8
ENCODER = "resnet18"
ARCH = "unet"

CONDITIONS = ["clean", "dark", "blur", "noise"]
DARK_FACTOR = 0.4
BLUR_SIGMA = 2.0
NOISE_SIGMA = 0.1
CKPT = {"rgb": os.path.join(HERE, "ckpt_rgb.pt"),
        "rgbd": os.path.join(HERE, "ckpt_rgbd.pt")}
OUT = os.path.join(HERE, "degraded_results.json")


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


# --------------------------------------------------------------------------- #
# RGB degradations. Input rgb: HxWx3 float32 in [0,1]. Returns HxWx3 [0,1].    #
# A per-image deterministic rng (keyed on `seed`) keeps eval reproducible.     #
# --------------------------------------------------------------------------- #
def _gauss_blur_rgb(rgb, radius):
    # PIL GaussianBlur on the 8-bit RGB image, returned back in [0,1].
    im = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), mode="RGB")
    im = im.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(im, dtype=np.float32) / 255.0


def degrade_rgb(rgb, cond, rng):
    if cond == "clean":
        return rgb
    if cond == "dark":
        return np.clip(rgb * DARK_FACTOR, 0.0, 1.0)
    if cond == "blur":
        return np.clip(_gauss_blur_rgb(rgb, BLUR_SIGMA), 0.0, 1.0)
    if cond == "noise":
        noise = rng.normal(0.0, NOISE_SIGMA, size=rgb.shape).astype(np.float32)
        return np.clip(rgb + noise, 0.0, 1.0)
    raise ValueError(cond)


# --------------------------------------------------------------------------- #
# Training (mirrors run_study_local.train_one, but saves best ckpt to disk)    #
# --------------------------------------------------------------------------- #
def train_one(mode, seed, epochs):
    set_seed(seed)
    samples = D.build_index(DATA_ROOT)
    split = D.split_index(samples, seed=42)  # FIXED split
    in_ch = 4 if mode == "rgbd" else 3
    DS = D.make_torch_dataset()
    tr = DataLoader(DS(split["train"], mode=mode, out_size=IMG, augment=True),
                    batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(DS(split["val"], mode=mode, out_size=IMG, augment=False),
                    batch_size=BS, shuffle=False, num_workers=0)
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None,
                     in_channels=in_ch, classes=1).to(DEVICE)
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_iou, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x); loss = dice(logits, y) + bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
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
        print(f"  [{mode} seed{seed}] ep{ep+1:02d}/{epochs} valIoU={iou:.3f}", flush=True)
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "in_ch": in_ch, "best_val_iou": best_iou,
                "mode": mode, "seed": seed}, CKPT[mode])
    print(f"  saved best ({mode}, valIoU={best_iou:.3f}) -> {CKPT[mode]}", flush=True)
    return model, split


def load_model(mode):
    ck = torch.load(CKPT[mode], map_location=DEVICE)
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=None,
                     in_channels=ck["in_ch"], classes=1).to(DEVICE)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model


# --------------------------------------------------------------------------- #
# Evaluation: mean test IoU per (model, condition). Depth kept CLEAN for RGB-D #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_condition(model, mode, test_samples, cond, eval_seed=1234):
    ious = []
    for i, s in enumerate(test_samples):
        rgb = D.load_rgb(s.image_path, IMG)                  # HxWx3 [0,1]
        rng = np.random.default_rng(eval_seed + i)           # per-image, reproducible
        rgb_d = degrade_rgb(rgb, cond, rng)
        if mode == "rgbd":
            depth = D.load_depth(s.depth_path, IMG)[..., None]   # CLEAN depth
            x = np.concatenate([rgb_d, depth], axis=-1)
        else:
            x = rgb_d
        y = D.rasterize_mask(s.label_path, IMG).astype(np.float32)
        xt = torch.from_numpy(x).permute(2, 0, 1)[None].float().to(DEVICE)
        prob = torch.sigmoid(model(xt))[0, 0].cpu().numpy()
        pred = (prob >= 0.5).astype(np.float32)
        tp = float((pred*y).sum()); fp = float((pred*(1-y)).sum()); fn = float(((1-pred)*y).sum())
        ious.append(tp/(tp+fp+fn+1e-7))
    return float(np.mean(ious)), ious


# --------------------------------------------------------------------------- #
def make_figure(per_cond_delta, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    xs = list(range(len(CONDITIONS)))
    ys = [per_cond_delta[c] for c in CONDITIONS]
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
    ax.plot(xs, ys, marker="o", color="#1f6feb", lw=2.0, ms=7,
            label=r"$\Delta$IoU = IoU$_{\mathrm{RGB\text{-}D}}-$IoU$_{\mathrm{RGB}}$")
    for x, yv in zip(xs, ys):
        ax.annotate(f"{yv:+.3f}", (x, yv), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xticks(xs); ax.set_xticklabels([c.capitalize() for c in CONDITIONS])
    ax.set_xlabel("RGB degradation condition")
    ax.set_ylabel(r"$\Delta$IoU (RGB-D $-$ RGB)")
    ax.set_title("Does depth help more as RGB degrades?")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figure -> {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--smoke", action="store_true", help="1-epoch pipeline check")
    ap.add_argument("--reuse-ckpt", action="store_true",
                    help="skip training, evaluate existing checkpoints")
    args = ap.parse_args()
    epochs = 1 if args.smoke else args.epochs
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"device={DEVICE} arch={ARCH} encoder={ENCODER} epochs={epochs} "
          f"seeds={seeds} smoke={args.smoke}", flush=True)
    print(f"conditions={CONDITIONS} dark={DARK_FACTOR} blur_sigma={BLUR_SIGMA} "
          f"noise_sigma={NOISE_SIGMA}", flush=True)
    t0 = time.time()

    # Accumulate per-condition IoU across seeds for each mode.
    acc = {mode: {c: [] for c in CONDITIONS} for mode in ("rgb", "rgbd")}
    per_seed = []

    for seed in seeds:
        models = {}
        for mode in ("rgb", "rgbd"):
            print(f"\n=== seed {seed} | train {mode} ===", flush=True)
            if args.reuse_ckpt and os.path.exists(CKPT[mode]):
                samples = D.build_index(DATA_ROOT)
                split = D.split_index(samples, seed=42)
                models[mode] = load_model(mode)
                print(f"  reused checkpoint {CKPT[mode]}", flush=True)
            else:
                models[mode], split = train_one(mode, seed, epochs)

        # evaluate both models on every condition (same test split)
        seed_rec = {"seed": seed, "rgb": {}, "rgbd": {}, "delta": {}}
        for cond in CONDITIONS:
            for mode in ("rgb", "rgbd"):
                miou, _ = eval_condition(models[mode], mode, split["test"], cond)
                acc[mode][cond].append(miou)
                seed_rec[mode][cond] = miou
            seed_rec["delta"][cond] = seed_rec["rgbd"][cond] - seed_rec["rgb"][cond]
            print(f"  [seed {seed}] {cond:6s} RGB={seed_rec['rgb'][cond]:.4f} "
                  f"RGB-D={seed_rec['rgbd'][cond]:.4f} "
                  f"dIoU={seed_rec['delta'][cond]:+.4f}", flush=True)
        per_seed.append(seed_rec)

    # ---- aggregate (mean over seeds) ----
    rgb_mean = {c: float(np.mean(acc["rgb"][c])) for c in CONDITIONS}
    rgbd_mean = {c: float(np.mean(acc["rgbd"][c])) for c in CONDITIONS}
    delta_mean = {c: rgbd_mean[c] - rgb_mean[c] for c in CONDITIONS}
    rgb_std = {c: float(np.std(acc["rgb"][c])) for c in CONDITIONS}
    rgbd_std = {c: float(np.std(acc["rgbd"][c])) for c in CONDITIONS}

    # hypothesis: delta grows as RGB worsens. Test vs clean baseline.
    base = delta_mean["clean"]
    grows = {c: (delta_mean[c] - base) for c in CONDITIONS if c != "clean"}
    supported = all(v > 0 for v in grows.values())

    print("\n" + "=" * 64, flush=True)
    print(f"{'condition':10s}{'IoU RGB':>12s}{'IoU RGB-D':>12s}{'dIoU':>12s}", flush=True)
    print("-" * 64, flush=True)
    for c in CONDITIONS:
        print(f"{c:10s}{rgb_mean[c]:12.4f}{rgbd_mean[c]:12.4f}{delta_mean[c]:+12.4f}", flush=True)
    print("-" * 64, flush=True)
    print("dIoU increase vs clean baseline:", flush=True)
    for c, v in grows.items():
        print(f"  {c:6s}: {v:+.4f}  ({'depth helps MORE' if v > 0 else 'no extra help'})", flush=True)
    print(f"\nHYPOTHESIS (depth helps more as RGB degrades): "
          f"{'SUPPORTED' if supported else 'NOT fully supported'}", flush=True)

    results = {
        "config": {"encoder": ENCODER, "arch": ARCH, "epochs": epochs,
                   "img": IMG, "seeds": seeds, "smoke": args.smoke,
                   "conditions": CONDITIONS, "dark_factor": DARK_FACTOR,
                   "blur_sigma": BLUR_SIGMA, "noise_sigma": NOISE_SIGMA,
                   "depth_kept_clean": True},
        "iou_rgb_mean": rgb_mean, "iou_rgb_std": rgb_std,
        "iou_rgbd_mean": rgbd_mean, "iou_rgbd_std": rgbd_std,
        "delta_iou_mean": delta_mean,
        "delta_increase_vs_clean": grows,
        "hypothesis_supported": bool(supported),
        "per_seed": per_seed,
        "wall_time_min": (time.time() - t0) / 60.0,
    }
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"\nsaved results -> {OUT}", flush=True)

    if not args.smoke:
        make_figure(delta_mean, os.path.join(PAPER_FIG, "degraded.pdf"))
    else:
        print("  (smoke run: skipping figure)", flush=True)
    print(f"Total wall time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
