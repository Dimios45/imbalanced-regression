# Phase 0 Notes: DIR Code Study and SkyFinder Dataset Overview

## Repository Layout (imdb-wiki-dir/ is the closest analog)

```
imdb-wiki-dir/
  train.py       entry point: args, data loading, train/val/test loops, shot_metrics()
  datasets.py    IMDBWIKI Dataset class: transforms and LDS weight computation
  loss.py        weighted loss functions (L1, MSE, focal-L1, focal-MSE, Huber)
  fds.py         FDS nn.Module: running stats, smoothing, calibration
  resnet.py      ResNet-50 backbone with FDS hook in forward pass
  utils.py       kernel construction (get_lds_kernel_window), calibrate_mean_var, misc
```

---

## (a) Label Binning and Empirical Label Density

**File:** `datasets.py`, function `_prepare_weights()` (line 55)

Labels are cast to integers (`int(label)`) and binned into 1-unit-wide integer buckets covering
`[0, max_target)`. For IMDB-WIKI, `max_target=121` covers age 0 to 120.

```python
value_dict = {x: 0 for x in range(max_target)}  # one bucket per integer
for label in labels:
    value_dict[min(max_target - 1, int(label))] += 1   # raw count = empirical density
```

The resulting `value_dict` is the empirical label density histogram as raw sample counts per
integer bucket. No normalisation is applied at this stage; the density enters reweighting
and LDS smoothing directly as counts.

**SkyFinder adaptation:** Replace `max_target=121` with the integer temperature range.
We shift labels by 30 to keep all indices non-negative, giving `MAX_TARGET = 86` bins.
Bin width stays at 1 degree C.

---

## (b) Label Distribution Smoothing (LDS)

**Files:** `utils.py:get_lds_kernel_window()`, `datasets.py:_prepare_weights()`

**Concept:** The raw empirical density is noisy for sparse label bins. LDS replaces it with a
1D kernel-smoothed version so that bins near a populated bin also receive informative density
estimates and thus more reasonable loss weights.

### Kernel construction (`utils.py` line 110)

```python
def get_lds_kernel_window(kernel, ks, sigma):
    # gaussian: place a 1 at centre, apply gaussian_filter1d, normalise by max
    if kernel == 'gaussian':
        base_kernel = [0.]*half_ks + [1.] + [0.]*half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(...)
    # triang: triangular window of length ks
    # laplace: exp(-|x|/sigma)/(2*sigma), normalised by max
```

Defaults from `train.py` argparse: `lds_kernel='gaussian'`, `lds_ks=5`, `lds_sigma=1`.

### Smoothing the density histogram (`datasets.py` line 76)

```python
lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
smoothed_value = convolve1d(
    np.asarray([v for _, v in value_dict.items()]),
    weights=lds_kernel_window, mode='constant')
num_per_label = [smoothed_value[int(label)] for label in labels]
```

### Weight computation (lines 80 to 82)

Applied after the sqrt_inv or inverse transform of the smoothed count:

```python
# sqrt_inv: value_dict = {k: sqrt(v) ...}  -> weight proportional to 1/sqrt(smoothed_count)
# inverse:  value_dict = {k: clip(v,5,1000) ...} -> weight proportional to 1/smoothed_count
weights = [1 / x for x in num_per_label]
scaling = len(weights) / sum(weights)   # normalise so mean weight = 1
weights = [scaling * w for w in weights]
```

These per-sample weights are returned from `__getitem__` as the third tensor and multiplied
into the loss element-wise in `loss.py` via `weighted_l1_loss` and related functions.

**Key detail:** `reweight` must be `'sqrt_inv'` or `'inverse'` when `lds=True`. Setting
`reweight='none'` with LDS raises an assertion error (datasets.py line 57).

---

## (c) Feature Distribution Smoothing (FDS)

**Files:** `fds.py` (FDS class), `resnet.py` (integration), `utils.py:calibrate_mean_var()`

**Concept:** Sparse label bins have few training samples, so the model's feature representations
for those bins are noisy. FDS smooths the mean and variance of the 2048-dim feature vectors
across adjacent label bins, then re-calibrates each sample's features to match the smoothed
statistics. This effectively transfers distribution knowledge from well-populated bins to
sparse neighbours.

### FDS module structure (`fds.py`)

Buffers maintained per bin (`bucket_num - bucket_start` bins total):

- `running_mean` and `running_var`: EMA of mean and variance of features in each bin
- `running_mean_last_epoch` and `running_var_last_epoch`: snapshot at end of last epoch
- `smoothed_mean_last_epoch` and `smoothed_var_last_epoch`: kernel-smoothed version of the snapshot

Defaults: `fds_kernel='gaussian'`, `fds_ks=5`, `fds_sigma=1`, `fds_mmt=0.9`,
`bucket_num=100`, `bucket_start=0`, `start_update=0`, `start_smooth=1`.

### Per-epoch stats update (`train.py` lines 269 to 282, `fds.py` lines 78 to 113)

After each training epoch (if `epoch >= start_update`):

1. **Re-encode all training samples** with the current model (no gradient).
2. Call `model.module.FDS.update_last_epoch_stats(epoch)`, which snapshots running stats and
   applies 1D kernel convolution with reflect padding across the bucket axis:
   ```python
   smoothed_mean = F.conv1d(pad(running_mean, reflect), kernel_window)
   smoothed_var  = F.conv1d(pad(running_var,  reflect), kernel_window)
   ```
3. Call `model.module.FDS.update_running_stats(encodings, labels, epoch)` for the EMA update:
   ```python
   running_mean[bin] = (1 - mmt) * curr_mean + mmt * running_mean[bin]
   ```

### Forward pass calibration (`resnet.py` lines 140 to 153, `fds.py:smooth()`)

During training, after global average pooling and before the linear head:

```python
encoding_s = encoding                          # 2048-d feature
if self.training and self.fds and epoch >= self.start_smooth:
    encoding_s = self.FDS.smooth(encoding_s, targets, epoch)
x = self.linear(encoding_s)
if self.training and self.fds:
    return x, encoding                         # return raw encoding for stat updates
```

`FDS.smooth()` applies `calibrate_mean_var()` per bin:

```python
# utils.py line 97
factor = clamp(smoothed_var / running_var, 0.1, 10)
calibrated = (feature - running_mean) * sqrt(factor) + smoothed_mean
```

This shifts the feature distribution to match the kernel-smoothed statistics, acting like a
per-label-bin batch normalisation guided by neighbouring bins.

---

## (d) Many / Median / Few-Shot Evaluation Protocol

**File:** `train.py`, function `shot_metrics()` (line 338)

Thresholds are hardcoded based on training label counts:

- **Many-shot:** `train_count > 100`
- **Median-shot:** `20 <= train_count <= 100`
- **Few-shot:** `train_count < 20`

```python
def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        ...
    for i, count in enumerate(train_class_count):
        if count > many_shot_thr:   # many
        elif count < low_shot_thr:  # few
        else:                       # median
```

Metrics reported per region: MSE, L1 (MAE), and G-Mean (geometric mean of per-sample
absolute errors). Overall G-Mean uses `scipy.stats.gmean` over all absolute errors.

**SkyFinder adaptation:** Same thresholds (100 and 20) applied to integer-degree-binned
training counts. With a temperature range of roughly -30 to 45 C (75 bins), many bins fall
into few or median shot because counts are spread across cameras and seasons.

---

## (e) Loss Reweighting Options

**Files:** `train.py` (the `--reweight` argument), `datasets.py:_prepare_weights()`, `loss.py`

| `--reweight` | Effect on bin count before inversion | Notes |
|---|---|---|
| `none` | No reweighting; all weights equal 1.0 | Baseline |
| `sqrt_inv` | count becomes sqrt(count), then w = 1/sqrt(count) | Softer than inverse |
| `inverse` | count is clipped to [5, 1000], then w = 1/count | Clipped to avoid explosion at near-empty bins |

When `lds=True`, smoothed counts replace raw counts before inversion. All weights are
normalised so their mean equals 1 (`scaling = N / sum(weights)`).

Loss functions in `loss.py` accept an optional `weights` tensor and multiply element-wise:

```python
def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    return torch.mean(loss)
```

FDS is orthogonal to loss reweighting because it operates on features rather than the loss.
LDS and FDS can be combined using the flags `--lds --fds --reweight sqrt_inv`.

---

## SkyFinder Dataset Overview

**Source:** https://cs.valdosta.edu/~rpmihail/skyfinder/

**Metadata file:** `analysis/complete_table_with_mcr.csv`

Key columns from inspecting the first rows:

| Column | Description |
|---|---|
| `Filename` | Image filename (e.g. `20130101_091628.jpg`) |
| `CamId` | Camera identifier (integer, e.g. `10066`) |
| `Year`, `Month`, `Day`, `Hour`, `Min` | Timestamp components |
| `Date` | Floating-point date (Excel serial format) |
| `Latitude`, `Longitude` | Camera location |
| `TempM` | **Temperature in Celsius, our regression target** |
| `TempI` | Temperature in Fahrenheit |
| `DewPtM/I`, `Hum`, `WspdM/I`, etc. | Other weather fields |
| `Conds`, `Icon` | Weather condition strings |
| `dirty`, `daylight`, `night`, `sunny`, ... | Scene attribute scores (0 to 1 floats) |

Missing values are coded as `-9999` (sometimes `-999`).

Images are organised by camera ID. The `Filename` column gives the file basename and the
full path is `images/{CamId}/{Filename}`.

---

## Files to Touch for SkyFinder Adaptation

| Original file | Our file | Changes |
|---|---|---|
| `imdb-wiki-dir/datasets.py` | `skyfinder-dir/src/datasets.py` | New `SkyFinder` class; adjusted `max_target`; `age` column replaced by `temp_c`; path construction from `CamId/Filename` |
| `imdb-wiki-dir/train.py` | `skyfinder-dir/src/train.py` | Dataset name, `bucket_num` for temperature range, shot-metric thresholds, wandb and AMP support |
| `imdb-wiki-dir/fds.py` | imported directly | No changes needed |
| `imdb-wiki-dir/resnet.py` | imported directly | One-line AMP dtype fix only |
| `imdb-wiki-dir/loss.py` | imported directly | No changes needed |
| `imdb-wiki-dir/utils.py` | imported directly | No changes needed |
| (new) | `skyfinder-dir/src/config.yaml` | All paths and hyperparameters |
| (new) | `skyfinder-dir/src/prepare_data.py` | Build master CSV from SkyFinder metadata |
| (new) | `skyfinder-dir/src/eda.py` | EDA plots and temperature distribution |

`fds.py`, `loss.py`, and `utils.py` are fully label-agnostic and require zero changes.
Only `datasets.py` and `train.py` need SkyFinder-specific edits. All changes are marked
with `# SKYFINDER:` comments.
