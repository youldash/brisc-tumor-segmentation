"""Train a tumour-segmentation model on BRISC 2025.

Usage (from the project root):
    python -m src.train --config configs/unet_resnet34.yaml
    python -m src.train --config configs/unet_resnet34.yaml --resume      # continue after interruption
    python -m src.train --config configs/unet_scratch.yaml --smoke        # 2-epoch pipeline check

Outputs go to runs/<run_name>/: best.pt, last.pt, history.csv, config.yaml, summary.json
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # must be set before importing torch
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.data_index import load_train_val_frames  # noqa: E402
from src.dataset import BriscSegDataset  # noqa: E402
from src.losses import BCEDiceLoss  # noqa: E402
from src.metrics import confusion_counts, validation_summary  # noqa: E402
from src.models import build_model, count_parameters, get_param_groups  # noqa: E402
from src.transforms import get_eval_transforms, get_train_transforms  # noqa: E402
from src.utils import get_device, load_config, resolve_path, save_json, save_yaml, set_seed  # noqa: E402

HISTORY_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "val_dice", "val_iou",
                  "val_precision", "val_recall", "val_global_dice", "val_dice_tumor",
                  "val_healthy_fp_rate", "epoch_time_s"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--resume", action="store_true", help="resume from runs/<run_name>/last.pt")
    p.add_argument("--smoke", action="store_true", help="tiny 2-epoch run to validate the pipeline")
    p.add_argument("--data-root")
    p.add_argument("--splits-dir")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--run-name")
    return p.parse_args()


def apply_overrides(cfg: dict, args) -> dict:
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.splits_dir:
        cfg["data"]["splits_dir"] = args.splits_dir
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.smoke:
        cfg["train"].update({"epochs": 2, "batch_size": 4, "patience": 5, "warmup_epochs": 0})
        cfg["data"].update({"limit_train": 16, "limit_val": 8, "num_workers": 0})
        cfg["run_name"] = f"{cfg['run_name']}_smoke"
    return cfg


def make_loaders(cfg: dict):
    d, t = cfg["data"], cfg["train"]
    root = resolve_path(d["root"])
    train_df, val_df = load_train_val_frames(resolve_path(d["splits_dir"]), bool(d.get("include_healthy")),
                                             float(d.get("healthy_val_fraction", 0.15)), cfg["seed"])
    if d.get("limit_train"):
        train_df = train_df.sample(n=min(d["limit_train"], len(train_df)), random_state=cfg["seed"])
    if d.get("limit_val"):
        val_df = val_df.sample(n=min(d["limit_val"], len(val_df)), random_state=cfg["seed"])

    size = int(d["img_size"])
    train_ds = BriscSegDataset(train_df, root, get_train_transforms(size))
    val_ds = BriscSegDataset(val_df, root, get_eval_transforms(size))

    nw = int(d["num_workers"])
    common = {"num_workers": nw, "persistent_workers": nw > 0, "pin_memory": False}
    bs = int(t["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=len(train_ds) > bs, **common)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, **common)
    return train_loader, val_loader


def make_scheduler(optimizer, t: dict):
    epochs, warmup = int(t["epochs"]), int(t["warmup_epochs"])
    min_ratio = float(t["min_lr"]) / float(t["lr"])

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip, epoch):
    model.train()
    total, n = 0.0, 0
    for images, masks in tqdm(loader, desc=f"epoch {epoch} train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), masks)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += loss.item() * images.size(0)
        n += images.size(0)
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold: float):
    model.eval()
    total, n, counts = 0.0, 0, []
    for images, masks in tqdm(loader, desc="validate", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        total += criterion(logits, masks).item() * images.size(0)
        n += images.size(0)
        pred = (torch.sigmoid(logits) > threshold).squeeze(1).cpu().numpy()
        target = (masks > 0.5).squeeze(1).cpu().numpy()
        counts.append(confusion_counts(pred, target))
    metrics = validation_summary(np.concatenate(counts))
    metrics["loss"] = total / max(n, 1)
    return metrics


def save_checkpoint(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: an interrupted save never corrupts the previous checkpoint


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    t = cfg["train"]
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    run_dir = resolve_path(cfg["output"]["runs_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, run_dir / "config.yaml")

    train_loader, val_loader = make_loaders(cfg)
    model = build_model(cfg["model"]).to(device)
    n_params = count_parameters(model)
    criterion = BCEDiceLoss(t["bce_weight"], t["dice_weight"])
    optimizer = torch.optim.AdamW(get_param_groups(model, float(t["lr"]), float(t["encoder_lr_mult"])),
                                  weight_decay=float(t["weight_decay"]))
    scheduler = make_scheduler(optimizer, t)

    start_epoch, best_dice, best_epoch, stale = 1, -1.0, 0, 0
    history_path = run_dir / "history.csv"
    if args.resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice, best_epoch, stale = ckpt["best_dice"], ckpt["best_epoch"], ckpt["stale"]
        print(f"Resumed from epoch {ckpt['epoch']} (best val Dice {best_dice:.4f} @ epoch {best_epoch})")
    else:
        with open(history_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=HISTORY_FIELDS).writeheader()
    with open(history_path) as f:  # keep the existing column order when resuming older runs
        history_fields = next(csv.reader(f), HISTORY_FIELDS)

    print(f"Run       : {cfg['run_name']}")
    print(f"Device    : {device}  |  torch {torch.__version__}")
    print(f"Model     : {cfg['model']['name']}"
          + (f" ({cfg['model']['encoder_name']})" if cfg["model"]["name"] != "unet_scratch" else "")
          + f"  |  {n_params / 1e6:.2f} M parameters")
    print(f"Data      : {len(train_loader.dataset)} train / {len(val_loader.dataset)} val images"
          + (" (incl. healthy slices)" if cfg["data"].get("include_healthy") else "") + ", "
          f"{cfg['data']['img_size']}px, batch {t['batch_size']}")

    run_start = time.time()
    epochs = int(t["epochs"])
    epoch = start_epoch - 1
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        lr = optimizer.param_groups[-1]["lr"]
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, t["grad_clip"], epoch)
        val = evaluate(model, val_loader, criterion, device, float(t["threshold"]))
        scheduler.step()
        if device.type == "mps":
            torch.mps.empty_cache()
        epoch_time = time.time() - t0

        improved = val["dice"] > best_dice
        if improved:
            best_dice, best_epoch, stale = val["dice"], epoch, 0
        else:
            stale += 1

        row = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val["loss"],
               "val_dice": val["dice"], "val_iou": val["iou"], "val_precision": val["precision"],
               "val_recall": val["recall"], "val_global_dice": val["global_dice"],
               "val_dice_tumor": val["dice_tumor"], "val_healthy_fp_rate": val["healthy_fp_rate"],
               "epoch_time_s": epoch_time}
        with open(history_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=history_fields, extrasaction="ignore").writerow(row)

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "epoch": epoch, "best_dice": best_dice,
                 "best_epoch": best_epoch, "stale": stale, "config": cfg, "val_metrics": val}
        if improved:
            save_checkpoint(run_dir / "best.pt", state)
        save_checkpoint(run_dir / "last.pt", state)

        print(f"[{epoch:3d}/{epochs}] train_loss {train_loss:.4f} | val_loss {val['loss']:.4f} | "
              f"Dice {val['dice']:.4f} | IoU {val['iou']:.4f} | "
              + (f"tumour Dice {val['dice_tumor']:.4f} | healthy FP {val['healthy_fp_rate']:.1%} | "
                 if val["n_healthy"] else "")
              + f"{epoch_time:.0f}s"
              + ("  * best" if improved else ""))
        if epoch == start_epoch:
            remaining = (epochs - epoch) * epoch_time / 60
            print(f"            ≈ {remaining:.0f} min left for the full schedule (early stopping may cut this)")

        if stale >= int(t["patience"]):
            print(f"Early stopping: no Dice improvement for {stale} epochs.")
            break
        if t.get("cooldown_seconds"):
            time.sleep(float(t["cooldown_seconds"]))

    summary = {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "parameters": n_params,
        "best_epoch": best_epoch,
        "best_val_dice": best_dice,
        "epochs_run_this_session": max(0, epoch - start_epoch + 1),
        "session_time_min": round((time.time() - run_start) / 60, 2),
        "device": str(device),
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    if (run_dir / "best.pt").exists():
        summary["best_val_metrics"] = torch.load(run_dir / "best.pt", map_location="cpu",
                                                 weights_only=False)["val_metrics"]
    save_json(summary, run_dir / "summary.json")
    print(f"\nDone. Best val Dice {best_dice:.4f} at epoch {best_epoch}. Outputs: {run_dir}")


if __name__ == "__main__":
    main()
