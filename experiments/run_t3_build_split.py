"""
run_t3_build_split.py — build a SCENE-DISJOINT train/val/test split (T3 fix).

The random 700/150/150 split has ~11 cross-split near-duplicate pairs (temporally
adjacent frames) — see run_t3_leakage.py. Here we remove that leakage by construction:

  * dHash every image; connect images with Hamming <= CLUSTER_HAMMING (default 10,
    looser than the >=8 audit threshold so the result is provably clean at <=8),
  * union-find -> connected components = "scenes" (bursts of near-duplicate frames),
  * pack whole scenes into train/val/test targeting 700/150/150 (greedy: each scene
    goes to the split with the largest remaining deficit),
  * VERIFY: recompute cross-split near-duplicates -> must be 0 at Hamming<=8.

Saves the versioned split to results/split_scene_disjoint.json (lists of image keys).
"""
import os, sys, json, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict
from PIL import Image
import pothrgbd_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("POTHRGBD_ROOT", os.path.join(HERE, "..", "dataset", "PothRGBD 2"))
CLUSTER_HAMMING = int(os.environ.get("CLUSTER_HAMMING", "10"))
SEED = 42
TARGET = {"train": 700, "val": 150, "test": 150}


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
    keys = [s.key for s in samples]
    ipath = {s.key: s.image_path for s in samples}
    h = {k: dhash(ipath[k]) for k in keys}

    parent = {k: k for k in keys}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if hamming(h[keys[i]], h[keys[j]]) <= CLUSTER_HAMMING:
                union(keys[i], keys[j])
    comp = defaultdict(list)
    for k in keys:
        comp[find(k)].append(k)
    comps = list(comp.values())
    sizes = sorted((len(c) for c in comps), reverse=True)
    print(f"images={len(keys)}  scenes(components)={len(comps)}  "
          f"largest scenes={sizes[:8]}  singletons={sum(1 for c in comps if len(c)==1)}")

    random.Random(SEED).shuffle(comps)
    comps.sort(key=len, reverse=True)          # big scenes first -> better packing
    assign = {"train": [], "val": [], "test": []}
    for c in comps:
        s = max(("test", "val", "train"), key=lambda s: TARGET[s] - len(assign[s]))
        assign[s].extend(c)

    # verify: no cross-split near-duplicates at <=8
    member = {k: s for s, ks in assign.items() for k in ks}
    cross = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if member[keys[i]] != member[keys[j]] and hamming(h[keys[i]], h[keys[j]]) <= 8:
                cross += 1

    out = {"cluster_hamming": CLUSTER_HAMMING, "seed": SEED,
           "sizes": {s: len(ks) for s, ks in assign.items()},
           "cross_split_pairs_leq8_after": cross,
           "train": sorted(assign["train"]), "val": sorted(assign["val"]),
           "test": sorted(assign["test"])}
    os.makedirs(os.path.join(HERE, "..", "results"), exist_ok=True)
    path = os.path.join(HERE, "..", "results", "split_scene_disjoint.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"split sizes: {out['sizes']}")
    print(f"cross-split near-duplicates (Hamming<=8) AFTER disjoint split: {cross}  "
          f"-> {'CLEAN' if cross == 0 else 'STILL LEAKY'}")
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
