"""
run_t7_pseudodepth.py — does ESTIMATED (monocular) depth help on PothRGBD?

A within-PothRGBD companion to the metric-depth (T1) and surface-normal (T2) tests:
instead of the sensor's metric depth, we feed a MONOCULAR pseudo-depth channel
(MiDaS) estimated from the same RGB image. This asks whether *any* depth signal ---
even a free, estimated one --- adds pothole-segmentation accuracy.

Same controlled protocol as T1/T2: ImageNet-pretrained U-Net, conv1 inflated
(RGB filters copied, pseudo-depth channel = mean of the three RGB filters),
ImageNet-normalized RGB channels, Dice+BCE, AdamW lr1e-3/wd1e-4, 18 ep,
best-val-IoU, seeds {41,42,43}. Uses the scene-disjoint split when POTHRGBD_SPLIT
is set (primary), else the fixed random split. Pseudo-depth is relative inverse
depth, per-image min-max normalized to [0,1]; it is cached once.
"""
import os, sys, time, json, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "18"))
IMG = (256, 256)
BS = int(os.environ.get("BS", "8"))
ENCODER = "resnet18"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "41,42,43").split(",")]
MODES = ["rgb", "rgbpd"]
CACHE = os.path.join(HERE, "..", "results", "pseudodepth_pothrgbd")
OUT = os.path.join(HERE, "..", "results", os.environ.get("OUT", "t7_pseudodepth.json"))
IN_CH = {"rgb": 3, "rgbpd": 4}

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMEAN = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
ISTD = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)


def midas_model():
    import builtins; builtins.input = lambda *a, **k: "y"   # auto-trust standard MiDaS hub repos
    import warnings; warnings.filterwarnings("ignore")
    return torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to("cpu").eval()


def midas_depth(model, img_path):
    im = Image.open(img_path).convert("RGB").resize(IMG, Image.BICUBIC)
    a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(a).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        pred = model(t)
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=IMG,
                                               mode="bicubic", align_corners=False).squeeze(1)[0]
    d = pred.numpy()
    return ((d - d.min()) / (d.max() - d.min() + 1e-7)).astype(np.float32)


def precompute(samples):
    os.makedirs(CACHE, exist_ok=True)
    todo = [s for s in samples if not os.path.exists(os.path.join(CACHE, s.key + ".npy"))]
    if not todo:
        print("pseudo-depth cache complete.", flush=True); return
    print(f"precomputing MiDaS pseudo-depth for {len(todo)} PothRGBD images ...", flush=True)
    m = midas_model()
    for i, s in enumerate(todo):
        np.save(os.path.join(CACHE, s.key + ".npy"), midas_depth(m, s.image_path))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    print("pseudo-depth cache done.", flush=True)


def load_input(mode, s):
    rgb = D.load_rgb(s.image_path, IMG)
    if mode == "rgb":
        return rgb
    pd = np.load(os.path.join(CACHE, s.key + ".npy"))[..., None]
    return np.concatenate([rgb, pd], axis=-1)


class DS(Dataset):
    def __init__(self, samples, mode, augment):
        self.samples, self.mode, self.augment = samples, mode, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = load_input(self.mode, s)
        y = D.rasterize_mask(s.label_path, IMG)[..., None].astype(np.float32)
        if self.augment and random.random() < 0.5:
            x = x[:, ::-1, :].copy(); y = y[:, ::-1, :].copy()
        return torch.from_numpy(x).permute(2, 0, 1).float(), torch.from_numpy(y).permute(2, 0, 1).float()


def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)


def norm_in(x):
    x = x.clone(); x[:, :3] = (x[:, :3] - IMEAN) / ISTD; return x


def build_model(in_ch):
    model = smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet", in_channels=3, classes=1)
    if in_ch != 3:
        old = model.encoder.conv1
        new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=(old.bias is not None))
        with torch.no_grad():
            new.weight[:, :3] = old.weight
            new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.encoder.conv1 = new
    return model.to(DEVICE)


def per_image_metrics(model, split, mode):
    rows = []; model.eval()
    with torch.no_grad():
        for s in split:
            x = load_input(mode, s)
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


def train_one(mode, seed, split):
    set_seed(seed)
    tr = DataLoader(DS(split["train"], mode, True), batch_size=BS, shuffle=True, drop_last=True)
    va = DataLoader(DS(split["val"], mode, False), batch_size=BS, shuffle=False)
    model = build_model(IN_CH[mode])
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True); bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state = -1.0, None
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
        iou /= max(nb, 1)
        if iou > best:
            best = iou; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  [{mode} seed{seed}] ep{ep+1:02d}/{EPOCHS} valIoU={iou:.3f}", flush=True)
    model.load_state_dict(best_state)
    rows = per_image_metrics(model, split["test"], mode)
    tm = {k: float(np.mean([r[k] for r in rows])) for k in ["IoU", "F1", "Precision", "Recall"]}
    print(f"  [{mode} seed{seed}] TEST mean {tm}", flush=True)
    return tm, rows


def main():
    split_env = os.environ.get("POTHRGBD_SPLIT")
    print(f"device={DEVICE} epochs={EPOCHS} seeds={SEEDS} split={'disjoint' if split_env else 'random42'}", flush=True)
    samples = D.build_index(DATA_ROOT)
    precompute(samples)
    split = D.split_from_json(samples, split_env) if split_env else D.split_index(samples, seed=42)
    print(f"split sizes: train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = {"config": {"encoder": ENCODER, "pretrained": "imagenet", "pseudo_depth": "MiDaS_small",
                          "epochs": EPOCHS, "img": IMG, "seeds": SEEDS,
                          "split": "disjoint" if split_env else "random42"},
               "runs": [], "per_image": {}}
    t0 = time.time()
    for seed in SEEDS:
        for mode in MODES:
            print(f"\n=== seed {seed} | {mode} ===", flush=True)
            tm, rows = train_one(mode, seed, split)
            results["runs"].append({"arch": "unet", "seed": seed, "mode": mode, "test_mean": tm})
            results["per_image"][f"unet_{seed}_{mode}"] = rows
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"  saved -> {OUT}", flush=True)
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
