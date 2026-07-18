"""
stats/analyze.py — consolidated statistics for the RGB vs RGB-D study (T5).

Consumes a results JSON (produced by run_study_local.py / run_t1_pretrained.py /
run_t2_normals.py: {"runs":[...], "per_image":{"<arch>_<seed>_<mode>":[rows]}})
and emits EVERY number the paper reports, deterministically and with no training:

  * seed-level RGB vs RGB-D means +/- std (IoU/F1/Precision/Recall),
  * paired t-test on the seed-averaged per-image IoU differences (n = #test imgs)
    with 95% CI,
  * Wilcoxon signed-rank on the same differences (robust companion; IoU deltas
    are typically non-normal),
  * TOST equivalence across margins {0.01, 0.015, 0.02, 0.03},
  * stratified paired tests by pothole size and by distance proxy (mean in-mask
    depth), split at the median, with Benjamini-Hochberg correction.

Usage:  python stats/analyze.py <results.json> [arch]
"""
import json, sys
import numpy as np
from collections import defaultdict
from scipy import stats


def seed_avg_deltas(per, arch, seeds, metric="IoU"):
    """Per test image, average (RGB-D - RGB) over seeds -> one delta per image."""
    img = defaultdict(list)
    for s in seeds:
        rg = {z["key"]: z[metric] for z in per.get(f"{arch}_{s}_rgb", [])}
        rd = {z["key"]: z[metric] for z in per.get(f"{arch}_{s}_rgbd", [])}
        for k in rg.keys() & rd.keys():
            img[k].append(rd[k] - rg[k])
    keys = sorted(img)
    return keys, np.array([np.mean(img[k]) for k in keys])


def paired_report(d):
    n = len(d); mean = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    p = 2 * stats.t.sf(abs(t), n - 1) if se > 0 else float("nan")
    tcrit = stats.t.ppf(0.975, n - 1)
    return dict(n=n, mean=float(mean), sd=float(sd), se=float(se), t=float(t),
                p=float(p), ci95=(float(mean - tcrit * se), float(mean + tcrit * se)))


def wilcoxon_report(d):
    try:
        w, p = stats.wilcoxon(d)
        return dict(W=float(w), p=float(p))
    except Exception as e:
        return dict(error=str(e))


def tost(d, margin):
    n = len(d); mean = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    p_low = stats.t.sf((mean + margin) / se, n - 1)   # H0: mean <= -margin
    p_high = stats.t.sf((margin - mean) / se, n - 1)  # H0: mean >= +margin
    p = max(p_low, p_high)
    return dict(margin=margin, p=float(p), equivalent=bool(p < 0.05))


def bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg; returns boolean rejection list in original order."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = np.asarray(pvals)[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    rej = np.zeros(m, bool)
    if below.any():
        kmax = np.max(np.where(below)[0])
        rej[order[:kmax + 1]] = True
    return rej


def stratified(per, arch, seeds, by):
    """Split images at the median of `by` (gt_area_frac or mean_depth); paired
    delta per stratum, BH-corrected across the two strata."""
    keys, d = seed_avg_deltas(per, arch, seeds)
    vals = defaultdict(list)
    for s in seeds:
        for z in per.get(f"{arch}_{s}_rgb", []):
            vals[z["key"]].append(z[by])
    v = np.array([np.mean(vals[k]) for k in keys])
    med = np.median(v)
    groups = {"low": d[v <= med], "high": d[v > med]}
    reps, ps = {}, []
    for g, dd in groups.items():
        reps[g] = paired_report(dd); ps.append(reps[g]["p"])
    for g, rj in zip(groups, bh_reject(ps)):
        reps[g]["bh_reject"] = bool(rj)
    reps["_by"] = by; reps["_median"] = float(med)
    return reps


def analyze(path, arch=None):
    r = json.load(open(path))
    per, runs = r["per_image"], r["runs"]
    archs = [arch] if arch else []
    if not arch:
        for x in runs:
            if x["arch"] not in archs:
                archs.append(x["arch"])
    print(f"# analyze {path}")
    if "config" in r:
        print(f"# config: {r['config']}")
    for a in archs:
        rgb_seeds = {x["seed"] for x in runs if x["arch"] == a and x["mode"] == "rgb"}
        rgbd_seeds = {x["seed"] for x in runs if x["arch"] == a and x["mode"] == "rgbd"}
        seeds = sorted(rgb_seeds & rgbd_seeds)
        if not seeds:
            print(f"\n########## {a}: no complete seed pairs yet ##########")
            continue
        print(f"\n########## {a}  (complete seeds: {seeds}) ##########")
        for metric in ["IoU", "F1", "Precision", "Recall"]:
            def lvl(mode):
                vv = [x["test_mean"][metric] for x in runs
                      if x["arch"] == a and x["mode"] == mode and x["seed"] in seeds]
                return np.mean(vv), np.std(vv)
            rm, rs = lvl("rgb"); dm, ds = lvl("rgbd")
            print(f"  {metric:10s} RGB {rm:.4f}+/-{rs:.4f}   RGB-D {dm:.4f}+/-{ds:.4f}   d {dm-rm:+.4f}")
        keys, d = seed_avg_deltas(per, a, seeds)
        pr = paired_report(d); wr = wilcoxon_report(d)
        print(f"  paired dIoU (n={pr['n']}): mean={pr['mean']:+.4f}  t={pr['t']:.2f}  p={pr['p']:.3f}"
              f"  95%CI=({pr['ci95'][0]:+.4f},{pr['ci95'][1]:+.4f})")
        wp = f"W={wr['W']:.0f} p={wr['p']:.3f}" if "W" in wr else str(wr)
        print(f"  Wilcoxon signed-rank: {wp}")
        print("  TOST equivalence:")
        for m in [0.01, 0.015, 0.02, 0.03]:
            tr = tost(d, m)
            print(f"    +/-{m:<5}: p={tr['p']:.4f}  -> {'EQUIVALENT' if tr['equivalent'] else 'inconclusive'}")
        for by in ["gt_area_frac", "mean_depth"]:
            st = stratified(per, a, seeds, by)
            print(f"  stratified by {by} (median={st['_median']:.4f}, BH-corrected):")
            for g in ["low", "high"]:
                rr = st[g]
                print(f"    {g:4s}(n={rr['n']:3d}): d={rr['mean']:+.4f}  t={rr['t']:.2f}  p={rr['p']:.3f}"
                      f"  BH_reject={rr['bh_reject']}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "depth-distill/results/t1_pretrained.json"
    arch = sys.argv[2] if len(sys.argv) > 2 else None
    analyze(path, arch)
