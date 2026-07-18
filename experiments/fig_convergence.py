"""
fig_convergence.py — T4 convergence figure from the stored per-epoch val-IoU curves.

Shows, per architecture, the mean +/- std (over seeds) validation-IoU trajectory for
RGB and RGB-D. Demonstrates (a) both arms converge well within 18 epochs, and (b) the
RGB and RGB-D curves track each other throughout (no widening gap) — addressing the
under-training / "gap would grow with more training" concern.

Usage: python experiments/fig_convergence.py [results.json] [out.pdf]
"""
import json, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "t1_pretrained.json")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "figures", "convergence.pdf")

r = json.load(open(src)); curves = r["curves"]; seeds = r["config"]["seeds"]
archs = []
for k in curves:
    a = k.rsplit("_", 2)[0]
    if a not in archs:
        archs.append(a)
labels = {"unet": "U-Net", "deeplabv3plus": "DeepLabV3+"}
colors = {"rgb": "#1f77b4", "rgbd": "#d62728"}

fig, axes = plt.subplots(1, len(archs), figsize=(5 * len(archs), 3.6), sharey=True)
if len(archs) == 1:
    axes = [axes]
for ax, arch in zip(axes, archs):
    for mode in ["rgb", "rgbd"]:
        stack = [curves[f"{arch}_{s}_{mode}"] for s in seeds if f"{arch}_{s}_{mode}" in curves]
        if not stack:
            continue
        L = min(len(c) for c in stack)
        arr = np.array([c[:L] for c in stack])
        m, sd = arr.mean(0), arr.std(0)
        ep = np.arange(1, L + 1)
        ax.plot(ep, m, color=colors[mode], label=("RGB" if mode == "rgb" else "RGB-D"), lw=1.8)
        ax.fill_between(ep, m - sd, m + sd, color=colors[mode], alpha=0.18)
    ax.set_title(labels.get(arch, arch)); ax.set_xlabel("epoch")
    ax.grid(alpha=0.3); ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("validation IoU")
fig.tight_layout()
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight")
print(f"saved -> {out}")
# quick numeric check: gap at last epoch vs best, per arch
for arch in archs:
    for mode in ["rgb", "rgbd"]:
        stack = [curves[f"{arch}_{s}_{mode}"] for s in seeds if f"{arch}_{s}_{mode}" in curves]
        arr = np.array([c[:min(len(x) for x in stack)] for c in stack])
        print(f"  {arch:14s} {mode:4s}: final valIoU={arr.mean(0)[-1]:.3f}  peak={arr.mean(0).max():.3f}")
