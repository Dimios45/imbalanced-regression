"""
Build master metadata CSV for SkyFinder temperature regression.

Reads complete_table_with_mcr.csv, filters invalid rows, and produces
skyfinder.csv with columns: path, camera_id, timestamp, temp_c, split.

Split strategy:
  PRIMARY   — camera-disjoint: train/val/test cameras are disjoint.
  SECONDARY — random 70/10/20 split (ignores camera identity).

Run:
    python src/prepare_data.py --data_dir data/ --images_dir data/images/
"""

import argparse
import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# Cameras included in our download subset (see download_skyfinder.sh)
DOWNLOADED_CAMERAS = {
    3888, 4795, 5021, 1093, 3297, 4801, 19106, 8438, 17218, 9730,
    9112, 19834, 4679, 4181, 6798, 204, 10066, 10870, 7211, 11331,
    8733, 7233, 9291, 861, 8953, 3395, 4584, 5020, 9483, 3837,
    162, 7371,
}

# Night filter: drop images where the 'night' scene-attribute score > threshold
NIGHT_THRESHOLD = 0.7
# Alternatively: require daylight > 0.3 (belt-and-suspenders)
DAYLIGHT_THRESHOLD = 0.2


def build_master(raw_csv: str, images_dir: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)
    n_raw = len(df)
    log.info(f"Raw rows: {n_raw}")

    # --- Drop missing temperature -----------------------------------------
    mask_missing_temp = df["TempM"].isna() | (df["TempM"] <= -99)
    log.info(f"  Drop missing/invalid TempM: {mask_missing_temp.sum()}")
    df = df[~mask_missing_temp].copy()

    # --- Drop cameras not downloaded ----------------------------------------
    mask_no_archive = ~df["CamId"].isin(DOWNLOADED_CAMERAS)
    log.info(f"  Drop cameras without image archive: {mask_no_archive.sum()}")
    df = df[~mask_no_archive].copy()

    # --- Build image path and check file exists -----------------------------
    images_dir = Path(images_dir)

    # Zips extract to a deep internal path mirroring the server's filesystem.
    # Actual structure: {images_dir}/{cam_id}/home/mihail/mypages/rpmihail/skyfinder/images/{cam_id}/{filename}
    _INNER = Path("home/mihail/mypages/rpmihail/skyfinder/images")

    def make_path(row):
        cam = str(int(row["CamId"]))
        return str(images_dir / cam / _INNER / cam / row["Filename"])

    df["path"] = df.apply(make_path, axis=1)

    if images_dir.exists():
        exists = df["path"].apply(os.path.isfile)
        log.info(f"  Drop missing image files: {(~exists).sum()}")
        df = df[exists].copy()
    else:
        log.warning(f"  images_dir {images_dir} not found — skipping file-existence check")

    # --- Drop night / fully-dark images ------------------------------------
    if "night" in df.columns and "daylight" in df.columns:
        mask_night = (df["night"] > NIGHT_THRESHOLD) | (df["daylight"] < DAYLIGHT_THRESHOLD)
        log.info(f"  Drop night/dark images (night>{NIGHT_THRESHOLD} or daylight<{DAYLIGHT_THRESHOLD}): {mask_night.sum()}")
        df = df[~mask_night].copy()
    else:
        log.warning("  'night'/'daylight' columns not found — skipping night filter")

    # --- Build timestamp column --------------------------------------------
    df["timestamp"] = pd.to_datetime(
        df[["Year", "Month", "Day", "Hour", "Min"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day",
                     "Hour": "hour", "Min": "minute"}
        )
    )

    # --- Select and rename final columns -----------------------------------
    out = df[["path", "CamId", "timestamp", "TempM", "Latitude", "Longitude",
              "Hum", "WspdM", "Conds"]].copy()
    out = out.rename(columns={"CamId": "camera_id", "TempM": "temp_c"})

    log.info(f"Final rows after all filters: {len(out)}")
    log.info(f"Temperature range: {out['temp_c'].min():.1f} to {out['temp_c'].max():.1f} °C")
    log.info(f"Cameras retained: {out['camera_id'].nunique()}")

    return out


def camera_disjoint_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    PRIMARY split: cameras are assigned entirely to train, val, or test.
    Ensures zero frame-level leakage from correlated same-camera sequences.
    Approximate target: 70% train, 10% val, 20% test by image count.
    """
    rng = np.random.default_rng(seed)
    cameras = df["camera_id"].unique()
    rng.shuffle(cameras)

    cam_counts = df.groupby("camera_id").size().loc[cameras]
    total = cam_counts.sum()

    train_target = 0.70 * total
    val_target = 0.10 * total

    split_col = pd.Series("test", index=df.index)
    running = 0
    in_train = True
    in_val = False
    train_cams, val_cams, test_cams = [], [], []

    for cam in cameras:
        cnt = cam_counts[cam]
        if in_train and running + cnt <= train_target * 1.05:
            train_cams.append(cam)
            running += cnt
            if running >= train_target:
                in_train = False
                in_val = True
                running = 0
        elif in_val and running + cnt <= val_target * 1.15:
            val_cams.append(cam)
            running += cnt
            if running >= val_target:
                in_val = False
        else:
            test_cams.append(cam)

    df = df.copy()
    df["split"] = "test"
    df.loc[df["camera_id"].isin(train_cams), "split"] = "train"
    df.loc[df["camera_id"].isin(val_cams), "split"] = "val"

    log.info("=== Camera-disjoint split (PRIMARY) ===")
    for s in ["train", "val", "test"]:
        rows = df[df["split"] == s]
        n_cams = rows["camera_id"].nunique()
        log.info(f"  {s}: {len(rows)} images, {n_cams} cameras")

    return df


def random_split(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    SECONDARY split: random 70/10/20 — for ablation comparison only.
    WARNING: same-camera frames appear in both train and test.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    train_end = int(0.70 * n)
    val_end = int(0.80 * n)

    split_col = np.full(n, "test", dtype=object)
    split_col[idx[:train_end]] = "train"
    split_col[idx[train_end:val_end]] = "val"

    df = df.copy()
    df["split_random"] = split_col
    log.info("=== Random split (SECONDARY) ===")
    for s in ["train", "val", "test"]:
        log.info(f"  {s}: {(df['split_random'] == s).sum()} images")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--images_dir", default="data/images/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_csv = os.path.join(args.data_dir, "complete_table_with_mcr.csv")
    df = build_master(raw_csv, args.images_dir)

    df = camera_disjoint_split(df, seed=args.seed)
    df = random_split(df, seed=args.seed)

    out_path = os.path.join(args.data_dir, "skyfinder.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Saved master CSV → {out_path}")

    # Print split summary
    print("\n=== Split summary ===")
    for split_col in ["split", "split_random"]:
        label = "Camera-disjoint (PRIMARY)" if split_col == "split" else "Random (SECONDARY)"
        print(f"\n{label}:")
        for s in ["train", "val", "test"]:
            rows = df[df[split_col] == s]
            print(f"  {s:5s}: {len(rows):6d} images | "
                  f"temp range [{rows['temp_c'].min():.1f}, {rows['temp_c'].max():.1f}] °C | "
                  f"{rows['camera_id'].nunique()} cameras")


if __name__ == "__main__":
    main()
