# Is a depth camera worth it for pothole segmentation? — controlled RGB vs RGB-D study

Reproducibility code for the paper **"Rethinking Depth for Low-Cost Pothole Segmentation:
A Controlled Multi-Regime RGB versus RGB-D Study."**

All experiments compare RGB against RGB-D (or RGB+Normal, or RGB+pseudo-depth) under a strict
single-variable protocol: within every paired run the architecture, loss (Dice+BCE), optimizer
(AdamW, lr 1e-3, wd 1e-4), schedule (18 epochs, best-val-IoU selection), augmentation, data split,
and random seed are held identical — only the input representation differs.

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export POTHRGBD_ROOT="/path/to/PothRGBD"   # folder with images/ depths/ labels/
```

Runs on Apple Silicon (MPS) or CUDA. **Dataset is third-party and not redistributed here:**
- **PothRGBD** — Yurdakul & Taşdemir, IEEE Dataport, DOI `10.21227/z8eq-sf60`.

All experiments (metric depth, surface normals, and monocular pseudo-depth) run on PothRGBD.

## Reproduce the results

Result JSONs are provided in `results/`; regenerate any of them, or analyze the stored ones directly.

```bash
# From-scratch baseline (secondary table)
SEEDS=41,42,43 OUT=study_results.json python experiments/run_study_local.py

# T1 — pretrained encoders (primary result): U-Net + DeepLabV3+, RGB vs RGB-D
ARCHS=unet,deeplabv3plus SEEDS=41,42,43 OUT=t1_pretrained.json python experiments/run_t1_pretrained.py

# T3 — leakage audit + scene-disjoint split, then rerun on the clean split
python experiments/run_t3_leakage.py
python experiments/run_t3_build_split.py
POTHRGBD_SPLIT="$PWD/results/split_scene_disjoint.json" \
  ARCHS=unet,deeplabv3plus SEEDS=41,42,43 OUT=t1_clean_split.json python experiments/run_t1_pretrained.py

# T2 — surface-normal encoding · T5 — RGB-degradation · T7 — monocular pseudo-depth (within PothRGBD)
ARCHS=unet MODES=rgb,rgbn SEEDS=41,42,43 OUT=t2_normals.json python experiments/run_t2_normals.py
python experiments/run_degraded_local.py --seeds 41,42,43
POTHRGBD_SPLIT="$PWD/results/split_scene_disjoint.json" \
  SEEDS=41,42,43 OUT=t7_pseudodepth.json python experiments/run_t7_pseudodepth.py

# Figures
python experiments/fig_convergence.py     # -> figures/convergence.pdf
python experiments/fig_qualitative.py      # -> figures/qualitative.pdf

# Statistics for any result file (paired t + Wilcoxon + TOST + stratified BH)
python stats/analyze.py results/t1_clean_split.json
```

## Layout

```
experiments/   training/audit scripts, pothrgbd_data.py loader, fig_*.py
stats/         analyze.py — consolidated statistics (regenerates every reported number)
results/       per-run metrics (IoU/F1/P/R per image + stratifiers + curves) and the scene-disjoint split
```

## Authors

Heldiansyah, Riswan Yunida, Muchtar Salim, Muhammad Akbar Hariyono, Rizky Amelia, Novi Shintia
— Department of Informatics Engineering, Politeknik Negeri Banjarmasin, Indonesia.

## Citation

If you use this code or the scene-disjoint split, please cite the paper (Indonesian Journal of
Electrical Engineering and Informatics, forthcoming) — Heldiansyah et al., "Rethinking Depth for
Low-Cost Pothole Segmentation: A Controlled Multi-Regime RGB versus RGB-D Study."

## License

MIT — see [LICENSE](LICENSE).
