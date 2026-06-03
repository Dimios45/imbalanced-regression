"""
SKYFINDER: SkyFinder Dataset class for temperature regression.

Modelled on imdb-wiki-dir/datasets.py. Changes from the original are
marked with # SKYFINDER: comments.

Original: imdb-wiki-dir/datasets.py  (IMDBWIKI class, age 0-120)
Ours    : SkyFinder class, target = temp_c (°C), arbitrary float range.
"""

import sys, os
# SKYFINDER: import original DIR utilities without copying them
_DIR_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../imdb-wiki-dir"))
sys.path.insert(0, _DIR_ROOT)

import logging
import numpy as np
from PIL import Image, ImageFile
from scipy.ndimage import convolve1d

# Allow PIL to load truncated/incomplete JPEG files (common in webcam datasets)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_lds_kernel_window  # from imdb-wiki-dir/utils.py

print = logging.info

# SKYFINDER: Temperature binning constants.
# We shift temp_c by TEMP_OFFSET so all bin indices are non-negative,
# matching DIR's assumption that bins start at 0.
TEMP_MIN = -30   # °C floor (below global dataset minimum of -27.2)
TEMP_MAX = 55    # °C ceiling (above global dataset maximum of 50)
TEMP_OFFSET = abs(TEMP_MIN)                      # 30 → shift so 0-indexed
MAX_TARGET = TEMP_MAX - TEMP_MIN + 1             # number of 1-°C bins = 86


def temp_to_bin(temp_c):
    """Map a temperature in °C to a non-negative integer bin index."""
    return int(np.clip(round(temp_c) + TEMP_OFFSET, 0, MAX_TARGET - 1))


class SkyFinder(data.Dataset):
    """
    SkyFinder outdoor webcam dataset for temperature regression.

    CSV columns used: path, temp_c, split.
    Compatible with DIR's train loop (returns img, label, weight triples).
    """

    def __init__(self, df, img_size, split="train", reweight="none",
                 lds=False, lds_kernel="gaussian", lds_ks=5, lds_sigma=2):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.split = split

        self.weights = self._prepare_weights(
            reweight=reweight, lds=lds,
            lds_kernel=lds_kernel, lds_ks=lds_ks, lds_sigma=lds_sigma,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index % len(self.df)]
        img = Image.open(row["path"]).convert("RGB")
        img = self.get_transform()(img)

        # SKYFINDER: label is temperature in °C, stored as float32 scalar
        label = np.asarray([row["temp_c"]], dtype="float32")
        weight = (
            np.asarray([self.weights[index]], dtype="float32")
            if self.weights is not None
            else np.asarray([np.float32(1.0)])
        )
        return img, label, weight

    def get_transform(self):
        # SKYFINDER: same transforms as imdb-wiki-dir; outdoor images at 224px
        if self.split == "train":
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def _prepare_weights(self, reweight, lds=False,
                         lds_kernel="gaussian", lds_ks=5, lds_sigma=2):
        # SKYFINDER: identical logic to imdb-wiki-dir/datasets.py _prepare_weights,
        # except max_target and the label→bin mapping differ.
        assert reweight in {"none", "inverse", "sqrt_inv"}
        assert reweight != "none" if lds else True, \
            "Set reweight to 'sqrt_inv' or 'inverse' when using LDS"

        # Build empirical label density histogram over integer-degree bins
        value_dict = {x: 0 for x in range(MAX_TARGET)}
        for temp in self.df["temp_c"].values:
            value_dict[temp_to_bin(temp)] += 1

        if reweight == "sqrt_inv":
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == "inverse":
            # clip avoids explosion for near-empty bins
            value_dict = {k: np.clip(v, 5, 1000) for k, v in value_dict.items()}

        num_per_label = [value_dict[temp_to_bin(t)] for t in self.df["temp_c"].values]

        if not len(num_per_label) or reweight == "none":
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f"Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})")
            smoothed_value = convolve1d(
                np.asarray([v for _, v in sorted(value_dict.items())]),
                weights=lds_kernel_window, mode="constant",
            )
            num_per_label = [smoothed_value[temp_to_bin(t)] for t in self.df["temp_c"].values]

        weights = [np.float32(1.0 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        return [scaling * w for w in weights]
