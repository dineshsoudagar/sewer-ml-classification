import os
import math
import yaml
import random
import argparse
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.transforms.functional as TF

from dataset import SewerMLDataset
from model import DinoV3MultiLabel
from metrics import search_thresholds, f1_from_thresholds
from dataset_balanced import SewerMLBalancedDataset

import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


class SimpleTransform:
    def __init__(self, img_size: int, train: bool, rotate_degrees=(0, 90, 180, 270)):
        self.img_size = img_size
        self.train = train
        self.rotate_degrees = rotate_degrees

        self.mean = torch.tensor(SEWER_MEAN).view(3, 1, 1)
        self.std = torch.tensor(SEWER_STD).view(3, 1, 1)

    def __call__(self, image):
        # image: HWC uint8 RGB -> CHW float [0,1]
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Always resize first (consistent geometry)
        x = F.resize(
            x,
            [self.img_size, self.img_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        # Train-time: rotate only
        if self.train:
            deg = random.choice(self.rotate_degrees)
            if deg != 0:
                x = F.rotate(x, deg, interpolation=InterpolationMode.BILINEAR, expand=False)

        # Normalize with Sewer-ML stats
        x = (x - self.mean) / self.std
        return {"image": x}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pos_weight(train_csv: str, labels: List[str]) -> torch.Tensor:
    df = pd.read_csv(train_csv)
    y = df[labels].to_numpy(dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pos_weight = (neg + 1.0) / (pos + 1.0)  # avoids div by zero
    return torch.tensor(pos_weight, dtype=torch.float32)


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_targets = []
    for _, x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(y.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def save_checkpoint(
        out_dir: str,
        epoch: int,
        model: nn.Module,
        cfg: dict,
        labels: List[str],
        thresholds: np.ndarray,
        macro_f1: float,
        micro_f1: float,
        best: bool = False,
) -> str:
    fname = f"epoch{epoch:02d}_macroF1_{macro_f1:.5f}_microF1_{micro_f1:.5f}.pt"
    path = os.path.join(out_dir, fname)

    payload = {
        "epoch": epoch,
        "model_name": cfg["model_name"],
        "labels": labels,
        "img_size": cfg["img_size"],
        "state_dict": model.state_dict(),
        "thresholds": thresholds,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "config": cfg,
    }
    torch.save(payload, path)

    if best:
        best_path = os.path.join(out_dir, "best.pt")
        shutil.copyfile(path, best_path)

    return path


def cleanup_checkpoints(out_dir: str, keep: int = 5):
    """Keep only the top-N checkpoints by macroF1 (parsed from filename)."""
    files = [f for f in os.listdir(out_dir) if f.endswith(".pt") and f.startswith("epoch")]
    if len(files) <= keep:
        return

    scored = []
    for f in files:
        # filename pattern: epochXX_macroF1_0.12345_microF1_0.67890.pt
        try:
            parts = f.split("_")
            macro = float(parts[2])  # macroF1 value
            scored.append((macro, f))
        except Exception:
            continue

    scored.sort(reverse=True, key=lambda x: x[0])
    to_keep = set([f for _, f in scored[:keep]])
    for f in files:
        if f not in to_keep:
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config.yaml")
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--val_csv", type=str, required=True)
    ap.add_argument("--train_images", type=str, required=True)
    ap.add_argument("--val_images", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    cfg = yaml.safe_load(open(args.config, "r"))
    labels = cfg["labels"]
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = SimpleTransform(cfg["img_size"], train=True, rotate_degrees=(0, 90, 180))
    val_tf = SimpleTransform(cfg["img_size"], train=False)

    #train_ds = SewerMLDataset(args.train_csv, args.train_images, labels, transform=train_tf)
    #val_ds = SewerMLDataset(args.val_csv, args.val_images, labels, transform=val_tf)
    train_ds = SewerMLBalancedDataset(
        csv_path=args.val_csv,  # same
        images_dir=args.val_images,  # same
        labels=labels,
        transform=train_tf,
        use_balanced_subset=True,
        subset_size=5000,  # recommended
        min_pos_per_class=200,  # good default
        seed=cfg["seed"],
    )

    val_ds = SewerMLBalancedDataset(
        csv_path=args.val_csv,
        images_dir=args.val_images,
        labels=labels,
        transform=val_tf,
        use_balanced_subset=True,
        subset_size=5000,  # keep same subset for evaluation
        min_pos_per_class=200,
        seed=cfg["seed"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train_batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["val_batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    model = DinoV3MultiLabel(cfg["model_name"], num_classes=len(labels), pretrained=True).to(device)

    # Loss
    if cfg.get("use_pos_weight", True):
        pos_weight = compute_pos_weight(args.train_csv, labels).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("use_amp", True))

    total_steps = cfg["epochs"] * len(train_loader)
    warmup_steps = int(cfg["warmup_epochs"] * len(train_loader))
    grad_accum = int(cfg.get("grad_accum_steps", 1))

    eval_every = int(cfg.get("eval_every_epochs", 1))
    monitor = cfg.get("monitor", "macro_f1")  # macro_f1 or micro_f1
    patience = int(cfg.get("early_stopping_patience", 3))
    min_delta = float(cfg.get("min_delta", 0.0))

    best_score = -1.0
    bad_epochs = 0

    global_step = 0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['epochs']}")
        optimizer.zero_grad(set_to_none=True)

        for step, (_, x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = cosine_warmup_lr(global_step, total_steps, cfg["lr"], warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=cfg.get("use_amp", True)):
                logits = model(x)
                loss = criterion(logits, y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if step % grad_accum == 0:
                #if cfg.get("clip_grad_norm", 0.0) and cfg["clip_grad_norm"] > 0:
                #    scaler.unscale_(optimizer)
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item() * grad_accum), lr=lr)

        # --------------------------
        # Evaluate and checkpoint
        # --------------------------
        if epoch % eval_every == 0:
            val_logits, val_targets = run_eval(model, val_loader, device)

            # Tune thresholds ON the validation set you provide (as requested).
            # If you want to avoid using val for threshold tuning, use an internal holdout instead.
            if cfg["threshold_strategy"] == "global":
                thresholds, _best_f1 = search_thresholds(
                    val_logits, val_targets, strategy="global", steps=cfg["threshold_steps"]
                )
                macro_f1, micro_f1 = f1_from_thresholds(val_logits, val_targets, thresholds)
            else:
                thresholds, macro_f1, micro_f1 = search_thresholds(
                    val_logits, val_targets, strategy="per_class", steps=cfg["threshold_steps"]
                )

            print(f"[Epoch {epoch}] val_macro_f1={macro_f1:.5f} val_micro_f1={micro_f1:.5f}")

            current = macro_f1 if monitor == "macro_f1" else micro_f1

            improved = current > (best_score + min_delta)
            if improved:
                best_score = current
                bad_epochs = 0
                ckpt_path = save_checkpoint(
                    args.out_dir, epoch, model, cfg, labels, thresholds, macro_f1, micro_f1, best=True
                )
                print(f"New BEST ({monitor}={best_score:.5f}). Saved: {ckpt_path}")
            else:
                bad_epochs += 1
                ckpt_path = save_checkpoint(
                    args.out_dir, epoch, model, cfg, labels, thresholds, macro_f1, micro_f1, best=False
                )
                print(f"No improvement. Saved: {ckpt_path} (bad_epochs={bad_epochs}/{patience})")

            # Optional checkpoint cleanup
            if not cfg.get("save_all_checkpoints", True):
                cleanup_checkpoints(args.out_dir, keep=int(cfg.get("max_keep", 5)))

            # Early stopping
            if bad_epochs >= patience:
                print(f"Early stopping triggered. Best {monitor}={best_score:.5f}")
                break

    print(f"Training finished. Best {monitor}={best_score:.5f}")
    print(f"Best checkpoint: {os.path.join(args.out_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
