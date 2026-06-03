"""
Generate all results plots for the SkyFinder DIR experiments.

Produces in results/:
  training_curves.png     — train L1 and val L1 vs epoch for all 3 experiments
  overall_comparison.png  — MAE / G-Mean bar chart across experiments
  shot_comparison.png     — shot-stratified MAE bar chart
  pred_vs_true.png        — scatter of predictions vs ground truth (best model)
  results_summary.csv     — clean 3-row table (no smoke/timing runs)

Usage:
    python src/plot_results.py --results_dir results/ --data_dir data/
"""

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.spines.top": False,
                     "axes.spines.right": False})

EXPS = {
    "01_baseline": "Baseline",
    "02_lds":      "LDS",
    "03_lds_fds":  "LDS + FDS",
}
COLORS = {"Baseline": "#4C72B0", "LDS": "#DD8452", "LDS + FDS": "#55A868"}

# ── Hard-coded test metrics from training log ─────────────────────────────────
# (extracted from test evaluation at best checkpoint)
TEST_METRICS = {
    "Baseline": {
        "overall": {"l1": 7.668, "mse": 91.957, "gmean": 5.053},
        "many":    {"l1": 7.132, "mse": 78.709, "gmean": 4.734},
        "median":  {"l1": 16.403, "mse": 290.084, "gmean": 15.828},
        "few":     {"l1": 23.531, "mse": 575.127, "gmean": 23.083},
    },
    "LDS": {
        "overall": {"l1": 7.330, "mse": 81.875, "gmean": 4.910},
        "many":    {"l1": 6.746, "mse": 67.218, "gmean": 4.587},
        "median":  {"l1": 16.983, "mse": 307.258, "gmean": 16.188},
        "few":     {"l1": 23.942, "mse": 584.256, "gmean": 23.713},
    },
    "LDS + FDS": {
        "overall": {"l1": 6.930, "mse": 83.838, "gmean": 4.460},
        "many":    {"l1": 6.233, "mse": 64.367, "gmean": 4.130},
        "median":  {"l1": 18.369, "mse": 380.676, "gmean": 17.188},
        "few":     {"l1": 27.035, "mse": 764.530, "gmean": 26.272},
    },
}


def parse_training_log(log_path):
    """Parse training.log → dict of {exp_key: {epoch: {train, val_l1, val_gmean}}}."""
    curves = {}
    current_exp = None

    exp_patterns = {k: re.compile(rf"store_name='({re.escape(k)}_[^']*)'") for k in EXPS}
    epoch_re = re.compile(r"Epoch (\d+): train=([\d.]+) \| val.*L1=([\d.]+) G.*?=([\d.]+)")

    with open(log_path) as f:
        for line in f:
            # Detect which experiment we're in
            for key, pat in exp_patterns.items():
                if pat.search(line):
                    current_exp = key
                    if key not in curves:
                        curves[key] = {}
                    break
            # Parse epoch summary line
            m = epoch_re.search(line)
            if m and current_exp and current_exp in EXPS:
                ep = int(m.group(1))
                curves[current_exp][ep] = {
                    "train": float(m.group(2)),
                    "val_l1": float(m.group(3)),
                    "val_gmean": float(m.group(4)),
                }

    return curves


def plot_training_curves(curves, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for key, label in EXPS.items():
        if key not in curves:
            continue
        data = curves[key]
        epochs = sorted(data.keys())
        train  = [data[e]["train"]  for e in epochs]
        val_l1 = [data[e]["val_l1"] for e in epochs]
        c = COLORS[label]
        axes[0].plot(epochs, train,  color=c, linewidth=1.8, label=label)
        axes[1].plot(epochs, val_l1, color=c, linewidth=1.8, label=label)

    for ax, title, ylabel in [
        (axes[0], "Training Loss (L1)", "L1 loss"),
        (axes[1], "Validation MAE per Epoch", "L1 (°C)"),
    ]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.axvline(20, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.axvline(25, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.text(20.2, ax.get_ylim()[1] * 0.97, "LR×0.1", fontsize=7, color="gray")

    plt.suptitle("Training curves — LR decays at epochs 20 and 25", y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_overall_comparison(out_path):
    labels = list(EXPS.values())
    mae    = [TEST_METRICS[l]["overall"]["l1"]   for l in labels]
    rmse   = [np.sqrt(TEST_METRICS[l]["overall"]["mse"]) for l in labels]
    gmean  = [TEST_METRICS[l]["overall"]["gmean"] for l in labels]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars_mae   = ax.bar(x - w, mae,   w, label="MAE (L1)",  color="#4C72B0", alpha=0.85)
    bars_rmse  = ax.bar(x,     rmse,  w, label="RMSE",      color="#DD8452", alpha=0.85)
    bars_gmean = ax.bar(x + w, gmean, w, label="G-Mean",    color="#55A868", alpha=0.85)

    for bars in [bars_mae, bars_rmse, bars_gmean]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error (°C)")
    ax.set_title("Overall test metrics — all errors in °C, lower is better")
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(rmse) * 1.18)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_shot_comparison(out_path):
    labels  = list(EXPS.values())
    regions = ["many", "median", "few"]
    region_labels = {"many": "Many-shot\n(>100 train)", "median": "Median-shot\n(20–100)",
                     "few": "Few-shot\n(<20 train)"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

    for ax, region in zip(axes, regions):
        mae_vals = [TEST_METRICS[l][region]["l1"] for l in labels]
        colors   = [COLORS[l] for l in labels]
        bars = ax.bar(labels, mae_vals, color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, mae_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_title(region_labels[region])
        ax.set_ylabel("MAE (°C)")
        ax.set_ylim(0, max(mae_vals) * 1.20)
        ax.tick_params(axis="x", labelsize=9)

    # Improvement arrows on many-shot panel
    baseline_many = TEST_METRICS["Baseline"]["many"]["l1"]
    best_many     = TEST_METRICS["LDS + FDS"]["many"]["l1"]
    axes[0].annotate("", xy=(2, best_many + 0.3), xytext=(0, baseline_many + 0.3),
                     arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
    axes[0].text(1, (baseline_many + best_many) / 2 + 1.2,
                 f"−{baseline_many - best_many:.2f}°C", ha="center", fontsize=8, color="red")

    plt.suptitle("Shot-stratified test MAE — few-shot = extreme temperatures (< −8 °C or > 31 °C)",
                 y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pred_vs_true(data_dir, results_dir, out_path):
    """Re-run best model on test set and scatter predictions vs ground truth."""
    # src/ must come before imdb-wiki-dir so our datasets.py takes priority
    _src = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.normpath(os.path.join(_src, "../../imdb-wiki-dir")))
    sys.path.insert(0, _src)

    try:
        import torch
        from resnet import resnet50
        from datasets import SkyFinder, MAX_TARGET, temp_to_bin
        from torch.utils.data import DataLoader
        import pandas as pd
    except ImportError as e:
        print(f"Skipping pred_vs_true (import error: {e})")
        return

    ckpt_dir = os.path.join(results_dir, "checkpoints",
                            "03_lds_fds_lds_gau_5_2_fds_gau_5_2_0_1_0.9_adam_l1_0.001_256")
    ckpt_path = os.path.join(ckpt_dir, "ckpt.best.pth.tar")
    if not os.path.isfile(ckpt_path):
        print(f"Skipping pred_vs_true (checkpoint not found: {ckpt_path})")
        return

    df = pd.read_csv(os.path.join(data_dir, "skyfinder.csv"))
    df_test = df[df["split"] == "test"]

    from PIL import ImageFile; ImageFile.LOAD_TRUNCATED_IMAGES = True
    ds = SkyFinder(df_test, img_size=224, split="val")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = resnet50(fds=True, bucket_num=MAX_TARGET, bucket_start=0,
                     start_update=0, start_smooth=1, kernel="gaussian",
                     ks=5, sigma=2, momentum=0.9)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.DataParallel(model).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for inp, tgt, _ in loader:
            out = model(inp.to(device))
            preds.extend(out.cpu().numpy().flatten())
            labels.extend(tgt.numpy().flatten())

    preds  = np.array(preds)
    labels = np.array(labels)

    # Shot region colouring
    df_train = df[df["split"] == "train"]
    train_counts = df_train["temp_c"].round(0).astype(int).value_counts()

    def shot_color(t):
        b = int(round(t))
        c = train_counts.get(b, 0)
        if c > 100: return "#4C72B0"
        if c >= 20: return "#DD8452"
        return "#55A868"

    colors = [shot_color(t) for t in labels]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(labels, preds, c=colors, alpha=0.35, s=8, linewidths=0)
    lim = (min(labels.min(), preds.min()) - 2, max(labels.max(), preds.max()) + 2)
    ax.plot(lim, lim, "k--", linewidth=1, alpha=0.5, label="Perfect prediction")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True temperature (°C)")
    ax.set_ylabel("Predicted temperature (°C)")
    ax.set_title("LDS+FDS predictions vs ground truth (test set)")

    from matplotlib.patches import Patch
    legend = [Patch(color="#4C72B0", label="Many-shot (>100)"),
              Patch(color="#DD8452", label="Median-shot (20–100)"),
              Patch(color="#55A868", label="Few-shot (<20)")]
    ax.legend(handles=legend + [plt.Line2D([0],[0], ls="--", color="k")],
              labels=[p.get_label() for p in legend] + ["Perfect"],
              fontsize=9, loc="upper left")

    mae = np.mean(np.abs(preds - labels))
    ax.text(0.97, 0.05, f"MAE = {mae:.2f} °C", transform=ax.transAxes,
            ha="right", fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def write_summary_csv(out_path):
    rows = []
    for key, label in EXPS.items():
        m = TEST_METRICS[label]
        rows.append({
            "experiment":     label,
            "reweight":       "none" if label == "Baseline" else "sqrt_inv",
            "lds":            label != "Baseline",
            "fds":            label == "LDS + FDS",
            "test_mae":       round(m["overall"]["l1"],   3),
            "test_rmse":      round(np.sqrt(m["overall"]["mse"]), 3),
            "test_gmean":     round(m["overall"]["gmean"], 3),
            "many_mae":       round(m["many"]["l1"],   3),
            "median_mae":     round(m["median"]["l1"], 3),
            "few_mae":        round(m["few"]["l1"],    3),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/")
    parser.add_argument("--data_dir",    default="data/")
    parser.add_argument("--log",         default="results/training.log")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print("Parsing training log...")
    curves = parse_training_log(args.log)
    print(f"  Found curves for: {list(curves.keys())}")

    plot_training_curves(curves,
        os.path.join(args.results_dir, "training_curves.png"))
    plot_overall_comparison(
        os.path.join(args.results_dir, "overall_comparison.png"))
    plot_shot_comparison(
        os.path.join(args.results_dir, "shot_comparison.png"))
    plot_pred_vs_true(args.data_dir, args.results_dir,
        os.path.join(args.results_dir, "pred_vs_true.png"))
    write_summary_csv(
        os.path.join(args.results_dir, "results_summary.csv"))

    print("\nAll results plots saved.")


if __name__ == "__main__":
    main()
