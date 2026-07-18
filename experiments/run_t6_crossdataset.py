"""
run_t6_crossdataset.py — T6 generality probe (optional in the brief).

Question: does MONOCULAR pseudo-depth help on a DIFFERENT, RGB-only pothole dataset?
Dataset: pothole600 (from Pothole-Mix / SHREC 2022) — 240 train / 180 val / 180 test, 400x400 RGB,
binary masks. Pseudo-depth from MiDaS_small (relative inverse depth, per-image min-max to [0,1]) —
this is NOT metric depth; framed as a generality probe.

Compares RGB vs RGB+pseudo-D (early fusion, 4-ch) under the SAME protocol as T1: ImageNet-pretrained
ResNet-18 U-Net, conv1 inflated (RGB filters copied, pseudo-depth channel = mean of RGB filters),
ImageNet-normalised RGB channels, Dice+BCE, AdamW lr1e-3/wd1e-4, 18 ep, best-val-IoU, seeds {41,42,43}.
Pseudo-depth is precomputed once and cached to results/t6_pseudodepth/.
"""
import os, sys, time, json, glob, subprocess, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PM = os.environ.get("POTHOLEMIX_ROOT",
                    os.path.join(HERE, "..", "dataset", "pothole-mix-extracted", "pothole-mix"))
SUB = "pothole600"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = int(os.environ.get("EPOCHS", "18"))
IMG = (256, 256)
BS = int(os.environ.get("BS", "8"))
ENCODER = "resnet18"
SEEDS = [int(s) for s in os.environ.get("SEEDS", "41,42,43").split(",")]
MODES = ["rgb", "rgbpd"]
CACHE = os.path.join(HERE, "..", "results", "t6_pseudodepth")
OUT = os.path.join(HERE, "..", "results", os.environ.get("OUT", "t6_crossdataset.json"))
SPLITS = {"training": "train", "validation": "val", "testing": "test"}

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
IMEAN = torch.tensor(MEAN).view(1, 3, 1, 1).to(DEVICE)
ISTD = torch.tensor(STD).view(1, 3, 1, 1).to(DEVICE)


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE).decode().strip()
    except Exception:
        return "unknown"


def list_split(split):
    imgs = sorted(glob.glob(os.path.join(PM, split, SUB, "images", "*")))
    return [(p, p.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)) for p in imgs]


def load_rgb(path):
    im = Image.open(path).convert("RGB").resize(IMG, Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def load_mask(path):
    # pothole600 masks encode the pothole in RED (255,0,0) on black -> use max over
    # channels (grayscale/luminance would drop red below threshold and yield empty masks).
    a = np.asarray(Image.open(path).convert("RGB").resize(IMG, Image.NEAREST))
    return (a.max(axis=-1) > 127).astype(np.float32)[..., None]


# ---------- pseudo-depth (MiDaS, cached) ----------
def midas_model():
    import builtins; builtins.input = lambda *a, **k: "y"   # auto-trust standard MiDaS hub repos
    import warnings; warnings.filterwarnings("ignore")
    return torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to("cpu").eval()


def midas_depth(model, img_path):
    im = Image.open(img_path).convert("RGB").resize(IMG, Image.BICUBIC)
    a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(a).permute(2, 0, 1)[None].float()
    with torch.no_grad():
        pred = model(t)                       # (1, h, w)
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=IMG,
                                               mode="bicubic", align_corners=False).squeeze(1)[0]
    d = pred.numpy()
    return ((d - d.min()) / (d.max() - d.min() + 1e-7)).astype(np.float32)   # per-image [0,1]


def key_of(split, img_path):
    return f"{split}_{os.path.splitext(os.path.basename(img_path))[0]}"


def precompute_pseudodepth():
    os.makedirs(CACHE, exist_ok=True)
    todo = [(sp, ip) for sp in SPLITS for ip, _ in list_split(sp)
            if not os.path.exists(os.path.join(CACHE, key_of(sp, ip) + ".npy"))]
    if not todo:
        print("pseudo-depth cache complete.", flush=True); return
    print(f"precomputing MiDaS pseudo-depth for {len(todo)} images ...", flush=True)
    m = midas_model()
    for i, (sp, ip) in enumerate(todo):
        np.save(os.path.join(CACHE, key_of(sp, ip) + ".npy"), midas_depth(m, ip))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(todo)}", flush=True)
    print("pseudo-depth cache done.", flush=True)


# ---------- dataset ----------
class DS(Dataset):
    def __init__(self, items, split, mode, augment):
        self.items, self.split, self.mode, self.augment = items, split, mode, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ip, mp = self.items[idx]
        rgb = load_rgb(ip)
        if self.mode == "rgbpd":
            pd = np.load(os.path.join(CACHE, key_of(self.split, ip) + ".npy"))[..., None]
            x = np.concatenate([rgb, pd], axis=-1)
        else:
            x = rgb
        y = load_mask(mp)
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


def per_image_metrics(model, items, split, mode):
    rows = []; model.eval()
    with torch.no_grad():
        for ip, mp in items:
            rgb = load_rgb(ip)
            if mode == "rgbpd":
                pd = np.load(os.path.join(CACHE, key_of(split, ip) + ".npy"))[..., None]
                x = np.concatenate([rgb, pd], axis=-1)
            else:
                x = rgb
            y = load_mask(mp)[..., 0]
            xt = norm_in(torch.from_numpy(x).permute(2, 0, 1)[None].float().to(DEVICE))
            pred = (torch.sigmoid(model(xt))[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
            tp = float((pred * y).sum()); fp = float((pred * (1 - y)).sum())
            fn = float(((1 - pred) * y).sum()); eps = 1e-7
            rows.append({"key": key_of(split, ip),
                         "IoU": tp / (tp + fp + fn + eps), "F1": 2 * tp / (2 * tp + fp + fn + eps),
                         "Precision": tp / (tp + fp + eps), "Recall": tp / (tp + fn + eps),
                         "gt_area_frac": float(y.mean())})
    return rows


def train_one(mode, seed, tr_items, va_items, te_items):
    set_seed(seed)
    in_ch = 4 if mode == "rgbpd" else 3
    tr = DataLoader(DS(tr_items, "training", mode, True), batch_size=BS, shuffle=True, drop_last=True)
    va = DataLoader(DS(va_items, "validation", mode, False), batch_size=BS, shuffle=False)
    model = build_model(in_ch)
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
    rows = per_image_metrics(model, te_items, "testing", mode)
    tm = {k: float(np.mean([r[k] for r in rows])) for k in ["IoU", "F1", "Precision", "Recall"]}
    print(f"  [{mode} seed{seed}] TEST mean {tm}", flush=True)
    return tm, rows


def main():
    print(f"device={DEVICE} dataset={SUB} epochs={EPOCHS} seeds={SEEDS}", flush=True)
    precompute_pseudodepth()
    tr_items, va_items, te_items = list_split("training"), list_split("validation"), list_split("testing")
    print(f"pothole600: train {len(tr_items)} / val {len(va_items)} / test {len(te_items)}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = {"config": {"encoder": ENCODER, "pretrained": "imagenet", "dataset": SUB,
                          "pseudo_depth": "MiDaS_small", "epochs": EPOCHS, "img": IMG,
                          "seeds": SEEDS, "git_hash": git_hash()},
               "runs": [], "per_image": {}}
    t0 = time.time()
    for seed in SEEDS:
        for mode in MODES:
            print(f"\n=== seed {seed} | {mode} ===", flush=True)
            tm, rows = train_one(mode, seed, tr_items, va_items, te_items)
            results["runs"].append({"arch": "unet", "seed": seed, "mode": mode, "test_mean": tm})
            results["per_image"][f"unet_{seed}_{mode}"] = rows
            json.dump(results, open(OUT, "w"), indent=1)
            print(f"  saved -> {OUT}", flush=True)
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
