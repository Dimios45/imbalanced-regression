"""
EDA for SkyFinder temperature regression.
Produces plots in results/ showing the label imbalance.

Usage:
    python src/eda.py --data_dir data/ --results_dir results/
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 11, "axes.titlesize": 12})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--results_dir", default="results/")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.data_dir, "skyfinder.csv"))
    train = df[df["split"] == "train"]

    print(f"Total rows: {len(df)}")
    print(f"Train: {len(train)}, Val: {(df.split=='val').sum()}, Test: {(df.split=='test').sum()}")
    print(f"Temperature range: {df.temp_c.min():.1f} to {df.temp_c.max():.1f} °C")
    print(f"Cameras: {df.camera_id.nunique()} total, "
          f"{train.camera_id.nunique()} train, "
          f"{df[df.split=='test'].camera_id.nunique()} test")

    # ── 1. Overall temperature distribution ─────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    bins = np.arange(df.temp_c.min() - 1, df.temp_c.max() + 2, 1)
    ax.hist(train["temp_c"], bins=bins, color="steelblue", edgecolor="none", alpha=0.8, label="train")
    ax.hist(df[df.split == "test"]["temp_c"], bins=bins, color="coral",
            edgecolor="none", alpha=0.6, label="test")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Image count")
    ax.set_title("Overall temperature distribution")
    ax.legend()

    # Mark many/median/few-shot thresholds from training counts
    train_counts = train["temp_c"].round(0).astype(int).value_counts()
    many_bins = sorted([b for b, c in train_counts.items() if c > 100])
    few_bins = sorted([b for b, c in train_counts.items() if c < 20])
    ax.axhline(100, color="green", linestyle="--", linewidth=0.9, label="many-shot thr (100)")
    ax.axhline(20, color="orange", linestyle="--", linewidth=0.9, label="few-shot thr (20)")
    ax.legend(fontsize=9)

    # ── 2. Log-scale to show tail imbalance ─────────────────────────────────
    ax = axes[1]
    ax.hist(train["temp_c"], bins=bins, color="steelblue", edgecolor="none", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Image count (log scale)")
    ax.set_title("Training label density (log scale) — imbalance visible")

    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "temp_distribution.png"), dpi=150)
    plt.close(fig)
    print("Saved: temp_distribution.png")

    # ── 3. Shot region breakdown ─────────────────────────────────────────────
    train_int = train["temp_c"].round(0).astype(int)
    bin_counts = train_int.value_counts().sort_index()

    many = bin_counts[bin_counts > 100]
    median = bin_counts[(bin_counts >= 20) & (bin_counts <= 100)]
    few = bin_counts[bin_counts < 20]

    print(f"\nShot regions (based on training label counts per integer °C bin):")
    print(f"  Many-shot   (>100 samples/bin): {len(many)} bins, "
          f"temp range [{many.index.min() if len(many) else 'N/A'}, "
          f"{many.index.max() if len(many) else 'N/A'}] °C")
    print(f"  Median-shot (20–100):           {len(median)} bins")
    print(f"  Few-shot    (<20 samples/bin):  {len(few)} bins, "
          f"includes extremes")

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {b: ("green" if c > 100 else "orange" if c >= 20 else "red")
              for b, c in bin_counts.items()}
    bars = ax.bar(bin_counts.index, bin_counts.values,
                  color=[colors[b] for b in bin_counts.index], width=0.9, alpha=0.8)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="green", label=f"Many-shot >100 ({len(many)} bins)"),
        Patch(color="orange", label=f"Median-shot 20–100 ({len(median)} bins)"),
        Patch(color="red", label=f"Few-shot <20 ({len(few)} bins)"),
    ]
    ax.legend(handles=legend_items, fontsize=9)
    ax.set_xlabel("Temperature bin (°C)")
    ax.set_ylabel("Training samples per bin")
    ax.set_title("Training label density by bin — shot region colouring")
    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "shot_regions.png"), dpi=150)
    plt.close(fig)
    print("Saved: shot_regions.png")

    # ── 4. Per-camera temperature distributions ──────────────────────────────
    cams = sorted(df["camera_id"].unique())
    n_cams = len(cams)
    ncols = 6
    nrows = (n_cams + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.2), sharey=False)
    axes_flat = axes.flatten() if n_cams > 1 else [axes]

    for i, cam in enumerate(cams):
        ax = axes_flat[i]
        sub = df[df["camera_id"] == cam]["temp_c"]
        split_tag = df.loc[df["camera_id"] == cam, "split"].iloc[0]
        color = {"train": "steelblue", "val": "purple", "test": "coral"}.get(split_tag, "gray")
        ax.hist(sub, bins=20, color=color, edgecolor="none", alpha=0.8)
        ax.set_title(f"cam {cam}\n({split_tag}, n={len(sub)})", fontsize=8)
        ax.set_xlabel("°C", fontsize=7)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.suptitle("Per-camera temperature distribution (colour = split)", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "per_camera_temp.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved: per_camera_temp.png")

    # ── 5. Monthly seasonality ───────────────────────────────────────────────
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
    fig, ax = plt.subplots(figsize=(9, 4))
    for split, color in [("train", "steelblue"), ("val", "purple"), ("test", "coral")]:
        sub = df[df["split"] == split]
        monthly = sub.groupby("month")["temp_c"].median()
        ax.plot(monthly.index, monthly.values, marker="o", label=split, color=color)
    ax.set_xlabel("Month")
    ax.set_ylabel("Median temperature (°C)")
    ax.set_title("Median temperature by month across splits")
    ax.legend()
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "monthly_temp.png"), dpi=150)
    plt.close(fig)
    print("Saved: monthly_temp.png")

    print("\nAll EDA plots saved to", args.results_dir)


if __name__ == "__main__":
    main()
