"""
SKYFINDER: Training script for temperature regression with DIR methods.

Adapted from imdb-wiki-dir/train.py. All changes marked # SKYFINDER:.
Imports fds, resnet, loss, utils directly from imdb-wiki-dir/ to avoid
code duplication — only datasets.py is ours.

Run examples:
  # Baseline
  python src/train.py --data_dir data/ --store_root results/checkpoints

  # + LDS
  python src/train.py --data_dir data/ --reweight sqrt_inv --lds

  # + FDS
  python src/train.py --data_dir data/ --reweight sqrt_inv --lds --fds

  # Full config from yaml
  python src/train.py --config src/config.yaml
"""

import sys, os
_SRC = os.path.dirname(os.path.abspath(__file__))
_DIR_ROOT = os.path.normpath(os.path.join(_SRC, "../../imdb-wiki-dir"))
# SKYFINDER: src/ must come before imdb-wiki-dir/ so our datasets.py shadows the original
sys.path.insert(0, _DIR_ROOT)
sys.path.insert(0, _SRC)

import time
import argparse
import logging
import yaml
from collections import defaultdict
from tqdm import tqdm

# SKYFINDER: wandb logging (gracefully disabled if not installed / not logged in)
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

import numpy as np
import pandas as pd
from scipy.stats import gmean

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from resnet import resnet50        # imdb-wiki-dir/resnet.py (unchanged)
from loss import (                 # imdb-wiki-dir/loss.py (unchanged)
    weighted_l1_loss, weighted_mse_loss,
    weighted_focal_l1_loss, weighted_focal_mse_loss, weighted_huber_loss,
)
from utils import (                # imdb-wiki-dir/utils.py (unchanged)
    AverageMeter, ProgressMeter, prepare_folders,
    adjust_learning_rate, save_checkpoint,
)

# SKYFINDER: our dataset class
from datasets import SkyFinder, MAX_TARGET, TEMP_OFFSET, temp_to_bin

os.environ["KMP_WARNINGS"] = "FALSE"


# ── Argument parsing ─────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--config", type=str, default="", help="optional YAML config file")

    # LDS
    p.add_argument("--lds", action="store_true", default=False)
    p.add_argument("--lds_kernel", type=str, default="gaussian",
                   choices=["gaussian", "triang", "laplace"])
    p.add_argument("--lds_ks", type=int, default=5)
    p.add_argument("--lds_sigma", type=float, default=2)
    # FDS
    p.add_argument("--fds", action="store_true", default=False)
    p.add_argument("--fds_kernel", type=str, default="gaussian",
                   choices=["gaussian", "triang", "laplace"])
    p.add_argument("--fds_ks", type=int, default=5)
    p.add_argument("--fds_sigma", type=float, default=2)
    p.add_argument("--start_update", type=int, default=0)
    p.add_argument("--start_smooth", type=int, default=1)
    # SKYFINDER: bucket range set to temperature bin range
    p.add_argument("--bucket_num", type=int, default=MAX_TARGET,
                   help="number of FDS buckets (= number of 1-°C bins)")
    p.add_argument("--bucket_start", type=int, default=0)
    p.add_argument("--fds_mmt", type=float, default=0.9)

    # Reweighting
    p.add_argument("--reweight", type=str, default="none",
                   choices=["none", "sqrt_inv", "inverse"])

    # Data / training
    p.add_argument("--data_dir", type=str, default="data/")
    p.add_argument("--store_root", type=str, default="results/checkpoints")
    p.add_argument("--store_name", type=str, default="")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"])
    p.add_argument("--loss", type=str, default="l1",
                   choices=["mse", "l1", "focal_l1", "focal_mse", "huber"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epoch", type=int, default=30)  # SKYFINDER: fewer epochs for speed
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--schedule", type=int, nargs="*", default=[20, 25])
    p.add_argument("--batch_size", type=int, default=256)  # SKYFINDER: large batch for MI300X
    p.add_argument("--print_freq", type=int, default=10)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True,
                   help="use automatic mixed precision (bfloat16 on ROCm/CUDA)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--pretrained", type=str, default="")
    p.add_argument("--evaluate", action="store_true")

    # SKYFINDER: wandb
    p.add_argument("--wandb", action="store_true", default=False,
                   help="enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="skyfinder-dir",
                   help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default="voidz7447-ksagar-site",
                   help="W&B entity (user or team)")

    args, _ = p.parse_known_args()

    # Override with YAML config if provided
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)

    return args


args = get_args()

# Set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Build experiment name
def _build_store_name(args):
    name = args.store_name or "skyfinder_resnet50"
    if args.reweight != "none" and not args.lds:
        name += f"_{args.reweight}"
    if args.lds:
        name += f"_lds_{args.lds_kernel[:3]}_{args.lds_ks}_{args.lds_sigma}"
    if args.fds:
        name += f"_fds_{args.fds_kernel[:3]}_{args.fds_ks}_{args.fds_sigma}"
        name += f"_{args.start_update}_{args.start_smooth}_{args.fds_mmt}"
    name += f"_{args.optimizer}_{args.loss}_{args.lr}_{args.batch_size}"
    return name

args.store_name = _build_store_name(args)
args.start_epoch = 0
args.best_loss = 1e5

prepare_folders(args)

# SKYFINDER: initialise wandb run
_wandb_run = None
if args.wandb and _WANDB_AVAILABLE:
    _wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.store_name,
        config=vars(args),
        tags=[
            args.reweight,
            "lds" if args.lds else "no_lds",
            "fds" if args.fds else "no_fds",
        ],
    )
elif args.wandb and not _WANDB_AVAILABLE:
    print("WARNING: --wandb set but wandb not installed; skipping.")

logging.root.handlers = []
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(args.store_root, args.store_name, "training.log")),
        logging.StreamHandler(),
    ],
)
print = logging.info
print(f"Args: {args}")


# ── Loss dispatch ─────────────────────────────────────────────────────────────

LOSS_FNS = {
    "l1": weighted_l1_loss,
    "mse": weighted_mse_loss,
    "focal_l1": weighted_focal_l1_loss,
    "focal_mse": weighted_focal_mse_loss,
    "huber": weighted_huber_loss,
}


# ── Shot-stratified evaluation ────────────────────────────────────────────────

def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    """
    SKYFINDER: same as imdb-wiki-dir/train.py shot_metrics() but labels are
    float °C — we round to integer bins for the training-count lookup.
    """
    train_bins = np.array([temp_to_bin(t) for t in train_labels])
    label_bins = np.array([temp_to_bin(t) for t in labels])

    unique_bins = np.unique(label_bins)
    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []

    for b in unique_bins:
        train_class_count.append((train_bins == b).sum())
        test_class_count.append((label_bins == b).sum())
        mse_per_class.append(np.sum((preds[label_bins == b] - labels[label_bins == b]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[label_bins == b] - labels[label_bins == b])))
        l1_all_per_class.append(np.abs(preds[label_bins == b] - labels[label_bins == b]))

    many_mse, median_mse, low_mse = [], [], []
    many_l1, median_l1, low_l1 = [], [], []
    many_gm, median_gm, low_gm = [], [], []
    many_cnt, median_cnt, low_cnt = [], [], []

    for i, cnt in enumerate(train_class_count):
        if cnt > many_shot_thr:
            many_mse.append(mse_per_class[i]); many_l1.append(l1_per_class[i])
            many_gm += list(l1_all_per_class[i]); many_cnt.append(test_class_count[i])
        elif cnt < low_shot_thr:
            low_mse.append(mse_per_class[i]); low_l1.append(l1_per_class[i])
            low_gm += list(l1_all_per_class[i]); low_cnt.append(test_class_count[i])
        else:
            median_mse.append(mse_per_class[i]); median_l1.append(l1_per_class[i])
            median_gm += list(l1_all_per_class[i]); median_cnt.append(test_class_count[i])

    def _safe(num_list, cnt_list, gm_list, metric):
        if not cnt_list:
            return float("nan")
        if metric == "mse":
            return np.sum(num_list) / np.sum(cnt_list)
        if metric == "l1":
            return np.sum(num_list) / np.sum(cnt_list)
        return gmean(np.hstack(gm_list), axis=None).astype(float)

    d = defaultdict(dict)
    for region, mse_l, l1_l, gm_l, cnt_l in [
        ("many",   many_mse,   many_l1,   many_gm,   many_cnt),
        ("median", median_mse, median_l1, median_gm, median_cnt),
        ("low",    low_mse,    low_l1,    low_gm,    low_cnt),
    ]:
        d[region]["mse"]   = _safe(mse_l, cnt_l, gm_l, "mse")
        d[region]["l1"]    = _safe(l1_l,  cnt_l, gm_l, "l1")
        d[region]["gmean"] = _safe(None,  cnt_l, gm_l, "gmean")
    return d


# ── Train / validate loops ────────────────────────────────────────────────────

def train(loader, model, optimizer, epoch, scaler=None):
    losses = AverageMeter(f"Loss ({args.loss.upper()})", ":.4f")
    progress = ProgressMeter(len(loader), [losses], prefix=f"Epoch [{epoch}]")

    model.train()
    loss_fn = LOSS_FNS[args.loss]
    device = next(model.parameters()).device
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for idx, (inputs, targets, weights) in enumerate(loader):
        inputs  = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        with autocast("cuda", dtype=amp_dtype, enabled=scaler is not None):
            if args.fds:
                outputs, _ = model(inputs, targets, epoch)
            else:
                outputs = model(inputs, targets, epoch)
            loss = loss_fn(outputs, targets, weights)

        assert not (np.isnan(loss.item()) or loss.item() > 1e6), \
            f"Loss explosion: {loss.item()}"

        losses.update(loss.item(), inputs.size(0))
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if idx % args.print_freq == 0:
            progress.display(idx)

    # FDS stat update (once per epoch, on full training set)
    if args.fds and epoch >= args.start_update:
        print(f"Updating FDS stats for epoch [{epoch}]...")
        encodings, labels_list = [], []
        model.train()  # must stay in train mode so forward() returns (out, encoding)
        with torch.no_grad():
            for (inp, tgt, _) in tqdm(loader):
                device = next(model.parameters()).device
                inp = inp.to(device, non_blocking=True)
                _, feat = model(inp, tgt.to(device), epoch)
                encodings.extend(feat.data.squeeze().cpu().numpy())
                labels_list.extend(tgt.data.squeeze().cpu().numpy())
        device = next(model.parameters()).device
        encodings = torch.from_numpy(np.vstack(encodings)).to(device)
        labels_t  = torch.from_numpy(np.hstack(labels_list)).to(device)
        # SKYFINDER: FDS expects integer bucket indices; convert °C → bin
        label_bins = torch.tensor(
            [temp_to_bin(t.item()) for t in labels_t], dtype=torch.float32
        ).to(device)
        model.module.FDS.update_last_epoch_stats(epoch)
        model.module.FDS.update_running_stats(encodings, label_bins, epoch)

    return losses.avg


def validate(loader, model, train_labels=None, prefix="Val"):
    mse_m = AverageMeter("MSE", ":.4f")
    l1_m  = AverageMeter("L1",  ":.4f")
    progress = ProgressMeter(len(loader), [mse_m, l1_m], prefix=f"{prefix}: ")

    crit_mse   = nn.MSELoss()
    crit_l1    = nn.L1Loss()
    crit_elem  = nn.L1Loss(reduction="none")

    model.eval()
    preds_all, labels_all, errs_all = [], [], []

    with torch.no_grad():
        for idx, (inputs, targets, _) in enumerate(loader):
            device = next(model.parameters()).device
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            preds_all.extend(outputs.cpu().numpy())
            labels_all.extend(targets.cpu().numpy())
            errs_all.extend(crit_elem(outputs, targets).cpu().numpy())

            mse_m.update(crit_mse(outputs, targets).item(), inputs.size(0))
            l1_m.update( crit_l1(outputs,  targets).item(), inputs.size(0))

            if idx % args.print_freq == 0:
                progress.display(idx)

    preds_np  = np.hstack(preds_all)
    labels_np = np.hstack(labels_all)
    errs_np   = np.hstack(errs_all)
    loss_gmean = gmean(errs_np, axis=None).astype(float)

    print(f" * Overall: MSE {mse_m.avg:.3f}  L1 {l1_m.avg:.3f}  G-Mean {loss_gmean:.3f}")

    if train_labels is not None:
        sd = shot_metrics(preds_np, labels_np, train_labels)
        for region in ["many", "median", "low"]:
            print(f" * {region.capitalize():6s}: "
                  f"MSE {sd[region]['mse']:.3f}  "
                  f"L1 {sd[region]['l1']:.3f}  "
                  f"G-Mean {sd[region]['gmean']:.3f}")

    return mse_m.avg, l1_m.avg, loss_gmean


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if args.gpu is not None:
        print(f"Using GPU: {args.gpu}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("=====> Preparing data...")
    df = pd.read_csv(os.path.join(args.data_dir, "skyfinder.csv"))
    df_train = df[df["split"] == "train"]
    df_val   = df[df["split"] == "val"]
    df_test  = df[df["split"] == "test"]
    train_labels = df_train["temp_c"].values

    ds_train = SkyFinder(df_train, img_size=args.img_size, split="train",
                         reweight=args.reweight,
                         lds=args.lds, lds_kernel=args.lds_kernel,
                         lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
    ds_val   = SkyFinder(df_val,   img_size=args.img_size, split="val")
    ds_test  = SkyFinder(df_test,  img_size=args.img_size, split="test")

    loader_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=False)
    loader_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    loader_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    print(f"Train: {len(ds_train)}  Val: {len(ds_val)}  Test: {len(ds_test)}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("=====> Building model...")
    model = resnet50(
        fds=args.fds,
        bucket_num=args.bucket_num,
        bucket_start=args.bucket_start,
        start_update=args.start_update,
        start_smooth=args.start_smooth,
        kernel=args.fds_kernel,
        ks=args.fds_ks,
        sigma=args.fds_sigma,
        momentum=args.fds_mmt,
    )

    # SKYFINDER: use GPU if available, fall back to CPU gracefully
    if torch.cuda.is_available():
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        else:
            model = nn.DataParallel(model).cuda()
    else:
        print("WARNING: no CUDA detected, running on CPU (slow!)")
        model = nn.DataParallel(model)  # DataParallel works on CPU too

    cudnn.benchmark = True

    # SKYFINDER: AMP scaler — enabled by default, skip on CPU
    use_amp = args.amp and torch.cuda.is_available()
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print(f"AMP enabled (dtype={'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'})")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = (
        torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.optimizer == "adam"
        else torch.optim.SGD(model.parameters(), lr=args.lr,
                             momentum=args.momentum, weight_decay=args.weight_decay)
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        args.start_epoch = ckpt["epoch"]
        args.best_loss   = ckpt["best_loss"]
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print(f"Resumed from epoch {args.start_epoch}")

    if args.evaluate:
        assert args.resume, "Provide --resume for evaluation"
        validate(loader_test, model, train_labels=train_labels, prefix="Test")
        return

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(args.start_epoch, args.epoch):
        adjust_learning_rate(optimizer, epoch, args)
        train_loss = train(loader_train, model, optimizer, epoch, scaler=scaler)
        val_mse, val_l1, val_gmean = validate(loader_val, model, train_labels=train_labels)

        metric = val_mse if args.loss == "mse" else val_l1
        is_best = metric < args.best_loss
        args.best_loss = min(metric, args.best_loss)

        save_checkpoint(args, {
            "epoch": epoch + 1,
            "best_loss": args.best_loss,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, is_best)

        print(f"Epoch {epoch}: train={train_loss:.4f} | "
              f"val MSE={val_mse:.4f} L1={val_l1:.4f} GMean={val_gmean:.4f}")

        # SKYFINDER: wandb per-epoch metrics
        if _wandb_run is not None:
            _wandb_run.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/mse": val_mse,
                "val/l1": val_l1,
                "val/gmean": val_gmean,
                "val/best_l1": args.best_loss,
            })

    # ── Test with best checkpoint ──────────────────────────────────────────
    print("=" * 80)
    ckpt_path = f"{args.store_root}/{args.store_name}/ckpt.best.pth.tar"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, best_loss {ckpt['best_loss']:.4f})")
    test_mse, test_l1, test_gmean = validate(loader_test, model,
                                             train_labels=train_labels, prefix="Test")
    print(f"Test: MSE={test_mse:.4f}  L1={test_l1:.4f}  GMean={test_gmean:.4f}")

    # SKYFINDER: save final test numbers to CSV for the results table
    out_row = {
        "experiment": args.store_name,
        "reweight": args.reweight,
        "lds": args.lds,
        "fds": args.fds,
        "test_mse": test_mse,
        "test_l1": test_l1,
        "test_gmean": test_gmean,
    }
    results_csv = os.path.join(args.store_root, "..", "results_table.csv")
    import csv
    write_header = not os.path.isfile(results_csv)
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_row.keys())
        if write_header:
            w.writeheader()
        w.writerow(out_row)
    print(f"Results appended → {results_csv}")

    # SKYFINDER: log final test metrics to wandb and close run
    if _wandb_run is not None:
        _wandb_run.log({
            "test/mse":   test_mse,
            "test/l1":    test_l1,
            "test/gmean": test_gmean,
        })
        _wandb_run.finish()


if __name__ == "__main__":
    main()
