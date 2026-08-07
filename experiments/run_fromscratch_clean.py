"""
run_fromscratch_clean.py — From-scratch U-Net on scene-disjoint split (R1 fix).
Identical protocol to run_study_local.py but uses the scene-disjoint split.
"""
import os, sys, json, time, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import pothrgbd_data as D

HERE   = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD"))
SPLIT_JSON = os.path.join(HERE, "..", "results", "split_scene_disjoint.json")
OUT   = os.path.join(HERE, "..", "results", "t_fromscratch_clean.json")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS, IMG, BS = 18, (256,256), 8
SEEDS = [41, 42, 43]

def set_seed(s):
    torch.manual_seed(s); np.random.seed(s); random.seed(s)

def train_eval(mode, seed, samples, split):
    set_seed(seed)
    in_ch = 4 if mode == "rgbd" else 3
    DS = D.make_torch_dataset()
    tr = DataLoader(DS(split["train"], mode=mode, out_size=IMG, augment=True),
                    batch_size=BS, shuffle=True, num_workers=0, drop_last=True)
    va = DataLoader(DS(split["val"],   mode=mode, out_size=IMG, augment=False),
                    batch_size=BS, shuffle=False, num_workers=0)
    te = DataLoader(DS(split["test"],  mode=mode, out_size=IMG, augment=False),
                    batch_size=BS, shuffle=False, num_workers=0)

    model = smp.Unet(encoder_name="resnet18", encoder_weights=None,
                     in_channels=in_ch, classes=1).to(DEVICE)
    dice = smp.losses.DiceLoss(mode="binary", from_logits=True)
    bce  = torch.nn.BCEWithLogitsLoss()
    opt  = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_iou, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = dice(model(x), y) + bce(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); iou_sum = 0.0; nb = 0
        with torch.no_grad():
            for x, y in va:
                x, y = x.to(DEVICE), y.to(DEVICE)
                p = (torch.sigmoid(model(x)) >= 0.5).float()
                tp=(p*y).sum().item(); fp=(p*(1-y)).sum().item(); fn=((1-p)*y).sum().item()
                iou_sum += tp/(tp+fp+fn+1e-7); nb += 1
        val_iou = iou_sum / max(nb,1)
        if val_iou > best_iou:
            best_iou = val_iou
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"  [{mode} s{seed}] ep{ep+1:02d} val={val_iou:.4f}", flush=True)

    model.load_state_dict(best_state); model.eval()
    per_image = []
    with torch.no_grad():
        for x, y in te:
            x, y = x.to(DEVICE), y.to(DEVICE)
            probs = torch.sigmoid(model(x))
            preds = (probs >= 0.5).float()
            for i in range(x.shape[0]):
                p, g = preds[i,0].cpu().numpy(), y[i,0].cpu().numpy()
                tp=float((p*g).sum()); fp=float((p*(1-g)).sum()); fn=float(((1-p)*g).sum())
                iou = tp/(tp+fp+fn+1e-7)
                prec = tp/(tp+fp+1e-7); rec = tp/(tp+fn+1e-7)
                per_image.append({"IoU":iou,"Precision":prec,"Recall":rec})
    mean_iou = float(np.mean([r["IoU"] for r in per_image]))
    print(f"  [{mode} s{seed}] TEST IoU={mean_iou:.4f}", flush=True)
    return per_image, mean_iou

def main():
    t0 = time.time()
    samples = D.build_index(DATA_ROOT)
    split   = D.split_from_json(samples, SPLIT_JSON)
    print(f"Split: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")

    results = {"config": {"encoder":"resnet18","pretrained":False,"split":"scene_disjoint",
                           "seeds":SEEDS,"epochs":EPOCHS,"img":list(IMG)},
               "per_image": {}}

    for seed in SEEDS:
        for mode in ["rgb", "rgbd"]:
            key = f"unet_{seed}_{mode}"
            if key in results["per_image"]:
                print(f"  skip {key} (cached)"); continue
            print(f"\n=== {key} ===", flush=True)
            pi, _ = train_eval(mode, seed, samples, split)
            results["per_image"][key] = pi
            with open(OUT, "w") as f:
                json.dump(results, f)
            print(f"  saved -> {OUT}", flush=True)

    elapsed = (time.time()-t0)/60
    results["wall_time_min"] = elapsed
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone in {elapsed:.1f} min. Output: {OUT}")

if __name__ == "__main__":
    main()
