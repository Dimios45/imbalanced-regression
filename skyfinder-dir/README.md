# SkyFinder-DIR: Temperature Regression from Outdoor Images

Adapts **Deep Imbalanced Regression (DIR)** — specifically Label Distribution Smoothing (LDS)
and Feature Distribution Smoothing (FDS) — to predict outdoor temperature in °C from
SkyFinder webcam images.

> Paper: [Delving into Deep Imbalanced Regression (ICML 2021)](https://arxiv.org/abs/2102.09554)  
> Original code: [YyzHarry/imbalanced-regression](https://github.com/YyzHarry/imbalanced-regression)  
> Dataset: [SkyFinder](https://cs.valdosta.edu/~rpmihail/skyfinder/)

---

## Results

| Experiment | MAE (L1) ↓ | RMSE ↓ | G-Mean ↓ |
|---|---|---|---|
| Baseline | 7.67 °C | 9.59 °C | 5.05 °C |
| + LDS | 7.33 °C | **9.05 °C** | 4.91 °C |
| + LDS + FDS | **6.93 °C** | 9.16 °C | **4.46 °C** |

**Camera-disjoint test split** (18 train / 3 val / 5 test cameras, no overlap).
Full shot-stratified results and discussion: [`report/REPORT.md`](report/REPORT.md).

W&B runs: https://wandb.ai/voidz7447-ksagar-site/skyfinder-dir

---

## Structure

```
skyfinder-dir/
├── data/
│   ├── complete_table_with_mcr.csv   # raw SkyFinder metadata
│   ├── skyfinder.csv                 # processed master table with splits
│   └── images/{cam_id}/...           # downloaded webcam images
├── src/
│   ├── download_skyfinder.sh         # download 32 camera archives (~2.5 GB)
│   ├── prepare_data.py               # build skyfinder.csv, camera-disjoint split
│   ├── eda.py                        # EDA plots → results/
│   ├── datasets.py                   # SkyFinder Dataset class (DIR adaptation)
│   ├── train.py                      # training loop with LDS/FDS/wandb
│   ├── run_experiments.sh            # run all 3 experiments sequentially
│   ├── plot_results.py               # results plots → results/
│   └── config.yaml                   # all hyperparameters
├── results/
│   ├── training_curves.png
│   ├── overall_comparison.png
│   ├── shot_comparison.png
│   ├── pred_vs_true.png
│   ├── temp_distribution.png
│   ├── shot_regions.png
│   ├── per_camera_temp.png
│   ├── monthly_temp.png
│   ├── results_summary.csv
│   └── training.log
├── report/
│   └── REPORT.md                     # full write-up with embedded figures
│   └── NOTES.md                          # DIR codebase study notes (Phase 0)
```

**Original DIR files used unchanged** (imported via `sys.path`):
`imdb-wiki-dir/{fds,resnet,loss,utils}.py`  
One-line AMP dtype fix in `imdb-wiki-dir/resnet.py` (marked `# SKYFINDER:`).

---

## Setup

```bash
conda activate graspmas   # torch 2.5.1+rocm6.2, scipy, pandas, wandb
cd imbalanced-regression/skyfinder-dir
```

## Quickstart

```bash
# 1. Download images
bash src/download_skyfinder.sh data/images/

# 2. Build dataset
python src/prepare_data.py --data_dir data/ --images_dir data/images/

# 3. EDA
python src/eda.py --data_dir data/ --results_dir results/

# 4. Run all 3 experiments (with wandb)
USE_WANDB=1 EPOCHS=30 bash src/run_experiments.sh 2>&1 | tee results/training.log

# 5. Results plots
python src/plot_results.py --results_dir results/ --data_dir data/
```

## Individual experiment commands

```bash
PYTHON=/mnt/data/mritunjoyh/miniforge3/envs/graspmas/bin/python

# Baseline
$PYTHON src/train.py --data_dir data/ --store_root results/checkpoints \
  --epoch 30 --batch_size 256 --workers 8 --store_name "01_baseline" --wandb

# LDS
$PYTHON src/train.py --data_dir data/ --store_root results/checkpoints \
  --epoch 30 --batch_size 256 --workers 8 --store_name "02_lds" \
  --reweight sqrt_inv --lds --wandb

# LDS + FDS
$PYTHON src/train.py --data_dir data/ --store_root results/checkpoints \
  --epoch 30 --batch_size 256 --workers 8 --store_name "03_lds_fds" \
  --reweight sqrt_inv --lds --fds --wandb
```

## Hardware

Tested on AMD Instinct MI300X (192 GB HBM3, ROCm 6.2).  
Total training time: ~1 h 45 min for all 3 experiments (30 epochs, batch 256, bfloat16 AMP).
