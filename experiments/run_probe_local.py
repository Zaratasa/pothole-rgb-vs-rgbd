"""
run_probe_local.py — Experiment #0 GO/NO-GO probe, run LOCALLY on Apple Silicon (MPS).

Trains a U-Net on RGB (3ch) vs RGB-D (4ch early fusion) — identical except the
input channel count — and prints whether depth meaningfully helps pothole
segmentation on PothRGBD. Metrics are computed on the held-out TEST split.
Comparison uses only comparable detection metrics (IoU/Dice/Precision/Recall),
never raw loss values across models.
"""
import os, sys, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # CPU fallback for any unsupported MPS op
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = int(os.environ.get("SEED", "42"))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
IMG = (int(os.environ.get("IMW", "256")), int(os.environ.get("IMH", "256")))  # (W, H)
BS = int(os.environ.get("BS", "8"))

torch.manual_seed(SEED); np.random.seed(SEED)

samples = D.build_index(DATA_ROOT)
split = D.split_index(samples, seed=SEED)
print(f"device={DEVICE} | samples={len(samples)} | split={{'train':{len(split['train'])},"
      f"'val':{len(split['val'])},'test':{len(split['test'])}}} | img={IMG} epochs={EPOCHS}",
      flush=True)


def seg_metrics(prob, target, thr=0.5, eps=1e-7):
    pred = (prob >= thr).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    return {"IoU": tp/(tp+fp+fn+eps), "Dice/F1": 2*tp/(2*tp+fp+fn+eps),
            "Precision": tp/(tp+fp+eps), "Recall": tp/(tp+fn+eps)}


def evaluate(model, samples_split, mode):
    DS = D.make_torch_dataset()
    dl = DataLoader(DS(samples_split, mode=mode, out_size=IMG, augment=False),
                    batch_size=BS, shuffle=False, num_workers=0)
    model.eval()
    agg = {"IoU": 0, "Dice/F1": 0, "Precision": 0, "Recall": 0}; nb = 0
    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            m = seg_metrics(torch.sigmoid(model(x)), y)
            for k in agg: agg[k] += m[k]
            nb += 1
    return {k: v / max(nb, 1) for k, v in agg.items()}


def train_unet(mode, epochs=EPOCHS):
    in_ch = 4 if mode == "rgbd" else 3
    DS = D.make_torch_dataset()
    tr = DataLoader(DS(split["train"], mode=mode, out_size=IMG, augment=True),
                    batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=in_ch, classes=1).to(DEVICE)
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_iou, best_state = -1.0, None
    for ep in range(epochs):
        t0 = time.time(); model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = dice(logits, y) + bce(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        val = evaluate(model, split["val"], mode)
        if val["IoU"] > best_iou:
            best_iou = val["IoU"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{mode}] ep{ep+1:02d}/{epochs} {time.time()-t0:5.1f}s "
              f"valIoU={val['IoU']:.3f} F1={val['Dice/F1']:.3f} "
              f"P={val['Precision']:.3f} R={val['Recall']:.3f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, split["test"], mode)  # report on held-out TEST
    print(f"[{mode}] TEST  IoU={test['IoU']:.3f} F1={test['Dice/F1']:.3f} "
          f"P={test['Precision']:.3f} R={test['Recall']:.3f}", flush=True)
    return test


def decision_gate(m_rgb, m_rgbd, d_iou=0.02, d_rec=0.02):
    print("\n========== DECISION GATE: does depth help? (TEST split) ==========", flush=True)
    print(f"{'metric':11s}{'RGB':>9s}{'RGB-D':>9s}{'delta':>9s}", flush=True)
    for k in ["IoU", "Dice/F1", "Precision", "Recall"]:
        a, b = m_rgb[k], m_rgbd[k]
        print(f"{k:11s}{a:9.3f}{b:9.3f}{b-a:+9.3f}", flush=True)
    go = (m_rgbd["IoU"]-m_rgb["IoU"] >= d_iou) or (m_rgbd["Recall"]-m_rgb["Recall"] >= d_rec)
    print("-"*40, flush=True)
    print("VERDICT:", "GO  -> proceed to full distillation" if go
          else "NO-GO -> Plan B (honest depth-ablation study)", flush=True)
    print("="*52, flush=True)
    return go


if __name__ == "__main__":
    t0 = time.time()
    print(">>> Training RGB baseline ...", flush=True)
    m_rgb = train_unet("rgb")
    print("\n>>> Training RGB-D ...", flush=True)
    m_rgbd = train_unet("rgbd")
    decision_gate(m_rgb, m_rgbd)
    print(f"\nTotal wall time: {(time.time()-t0)/60:.1f} min", flush=True)
