#!/usr/bin/env bash
# Downloads a representative subset of SkyFinder camera archives.
# Cameras selected to cover the full temperature range (-27 to 50 C)
# with manageable total size (~2.5 GB).
# Usage: bash src/download_skyfinder.sh [DATA_DIR]

set -euo pipefail

DATA_DIR="${1:-$(dirname "$0")/../data/images}"
BASE_URL="https://cs.valdosta.edu/~rpmihail/skyfinder/images"

# All cameras ≥700 images and <200 MB, plus 162 (coldest: -27.2°C) and 7371 (-18°C, 248MB)
CAMERAS=(
  3888 4795 5021 1093 3297 4801 19106 8438 17218 9730
  9112 19834 4679 4181 6798 204 10066 10870 7211 11331
  8733 7233 9291 861 8953 3395 4584 5020 9483 3837
  162 7371
)

mkdir -p "$DATA_DIR"

for cam in "${CAMERAS[@]}"; do
  ZIP="$DATA_DIR/${cam}.zip"
  CAM_DIR="$DATA_DIR/${cam}"
  if [ -d "$CAM_DIR" ]; then
    echo "[SKIP] Camera $cam already extracted"
    continue
  fi
  echo "[DOWNLOAD] Camera $cam ..."
  curl -L --silent --show-error --max-time 300 \
       -o "$ZIP" "${BASE_URL}/${cam}.zip" && \
  echo "[EXTRACT] Camera $cam ..." && \
  unzip -q "$ZIP" -d "$CAM_DIR" && \
  rm "$ZIP" && \
  echo "[DONE] Camera $cam ($(ls "$CAM_DIR" | wc -l | tr -d ' ') files)"
done

echo "All downloads complete. Total images: $(find "$DATA_DIR" -name '*.jpg' | wc -l)"
