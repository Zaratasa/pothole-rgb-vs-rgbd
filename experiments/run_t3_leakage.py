"""
run_t3_leakage.py — T3 audit: near-duplicate images ACROSS the train/val/test split?

Reviewer concern: the 700/150/150 split is "fixed" but maybe not scene-disjoint;
consecutive / near-duplicate frames straddling train and test would inflate IoU and
blur the modality difference. This script quantifies that risk.

Method (self-contained, no external hashing dep):
  * 64-bit dHash per RGB image (9x8 grayscale, horizontal gradient sign),
  * the EXACT split used for training (D.split_index(samples, seed=42)),
  * for every CROSS-split image pair, Hamming distance; count pairs at several
    thresholds and list the closest examples.

Hamming <= 8 on a 64-bit dHash ~ visually near-duplicate. Saves results/t3_leakage.json.
"""
import os, sys, json
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
THRESHOLDS = [0, 2, 4, 6, 8]


def dhash(path, hs=8):
    im = Image.open(path).convert("L").resize((hs + 1, hs), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a, b):
    return bin(a ^ b).count("1")


def main():
    samples = D.build_index(DATA_ROOT)
    split = D.split_index(samples, seed=42)
    hashes, member = {}, {}
    for name, subset in split.items():
        for s in subset:
            hashes[s.key] = dhash(s.image_path); member[s.key] = name
    items = list(hashes.items())
    pairs = []
    for i in range(len(items)):
        ki, hi = items[i]
        for j in range(i + 1, len(items)):
            kj, hj = items[j]
            if member[ki] == member[kj]:
                continue
            dd = hamming(hi, hj)
            if dd <= max(THRESHOLDS):
                pairs.append((dd, ki, member[ki], kj, member[kj]))
    pairs.sort()
    counts = {t: sum(1 for p in pairs if p[0] <= t) for t in THRESHOLDS}
    report = {
        "n_images": len(items),
        "split_sizes": {k: len(v) for k, v in split.items()},
        "cross_split_pairs_by_hamming": counts,
        "examples": [{"hamming": p[0], "a": p[1], "a_split": p[2], "b": p[3], "b_split": p[4]}
                     for p in pairs[:25]],
    }
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    out = os.path.join(HERE, "..", "results", "t3_leakage.json")
    json.dump(report, open(out, "w"), indent=1)

    print(f"images={len(items)}  splits={report['split_sizes']}")
    print(f"cross-split near-duplicate pairs by Hamming threshold: {counts}")
    verdict = ("CLEAN — no cross-split near-duplicates (Hamming<=8)"
               if counts[8] == 0 else
               f"REVIEW — {counts[8]} pairs <=8 ({counts[4]} <=4, {counts[0]} exact)")
    print("VERDICT:", verdict)
    for p in pairs[:10]:
        print(f"  d={p[0]:2d}  {p[1]} ({p[2]})  ~  {p[3]} ({p[4]})")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
