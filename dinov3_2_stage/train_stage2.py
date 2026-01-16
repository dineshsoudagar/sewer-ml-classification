import os
import math
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SewerMLDataset
from model import DinoV3MultiLabel
from metrics import search_thresholds, f1_from_thresholds
from train_utils import SimpleTransform

# Training Config
TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train_stage2_sanity_5k.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train_stage2_sanity_5k.csv"
TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
OUT_DIR = "outputs_stage2_vit_base_tesk_5k"
DEFECT_ONLY = False
# =========================

# =========================
# CONFIG (edit)
# =========================
SEED = 42

LABELS = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA",
          "ND"]
ND_LABEL = "ND"

if DEFECT_ONLY:
    NUM_CLASSES = 1
else:
    NUM_CLASSES = len(LABELS) - 1

FREEZE_BACKBONE = False
IMG_SIZE = 256
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 10
LR = 1.0e-5
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1
USE_AMP = False
GRAD_ACCUM_STEPS = 1
CLIP_GRAD_NORM = 1.0  # optional; keep simple (off by default below)
USE_POS_WEIGHT = True  # stage-2 only
POS_WEIGHT_CLAMP = 50.0  # simple safety clamp
THRESHOLD_STRATEGY = "global"  # "global" or "per_class"
THRESHOLD_STEPS = 200
EVAL_EVERY_EPOCHS = 1
MONITOR = "macro_f1"  # "macro_f1" or "micro_f1"
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0001
SAVE_ALL_CHECKPOINTS = True
MAX_KEEP = 5
# =========================

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


# Utils
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits, all_targets = [], []
    for _, x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(y.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def save_checkpoint_multilabel(out_dir: str, epoch: int, model: nn.Module, labels, thresholds, macro_f1, micro_f1,
                               best: bool):
    fname = f"epoch{epoch:02d}_macroF1_{macro_f1:.5f}_microF1_{micro_f1:.5f}.pt"
    path = os.path.join(out_dir, fname)
    torch.save({
        "epoch": epoch,
        "model_name": MODEL_NAME,
        "labels": labels,
        "img_size": IMG_SIZE,
        "state_dict": model.state_dict(),
        "thresholds": thresholds,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }, path)
    if best:
        shutil.copyfile(path, os.path.join(out_dir, "best.pt"))
    return path


def cleanup_checkpoints(out_dir: str, keep: int):
    files = [f for f in os.listdir(out_dir) if f.endswith(".pt") and f.startswith("epoch")]
    if len(files) <= keep:
        return
    files.sort()
    for f in files[:-keep]:
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass


def compute_pos_weight_defects(train_csv: str, labels_wo_nd, nd_label: str, clamp: float):
    df = pd.read_csv(train_csv)
    df = df[df[nd_label] == 0].reset_index(drop=True)
    y = df[labels_wo_nd].to_numpy(dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = (neg + 1.0) / (pos + 1.0)
    w = np.minimum(w, clamp)
    return torch.tensor(w, dtype=torch.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Stage-2 labels = all except ND
    LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

    train_tf = SimpleTransform(IMG_SIZE, train=True, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    val_tf = SimpleTransform(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)

    # Stage-2: defect_only=False => dataset filters ND==0 rows + removes ND from y (your dataset logic)
    train_ds = SewerMLDataset(TRAIN_CSV, TRAIN_IMAGES, LABELS, transform=train_tf, defect_only=False)
    val_ds = SewerMLDataset(VAL_CSV, VAL_IMAGES, LABELS, transform=val_tf, defect_only=False)

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Model: 18 outputs
    model = DinoV3MultiLabel(MODEL_NAME, num_classes=len(LABELS_WO_ND), pretrained=True).to(device)
    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    if USE_POS_WEIGHT:
        pos_weight = compute_pos_weight_defects(TRAIN_CSV, LABELS_WO_ND, ND_LABEL, POS_WEIGHT_CLAMP).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_EPOCHS * len(train_loader))

    best_score = -1.0
    bad_epochs = 0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage2] Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for step, (_, x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr = cosine_warmup_lr(global_step, total_steps, LR, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(x)
            loss = criterion(logits.float(), y.float()) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0:
                # Optional:
                # if CLIP_GRAD_NORM and CLIP_GRAD_NORM > 0:
                #     scaler.unscale_(optimizer)
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item() * GRAD_ACCUM_STEPS), lr=lr)

        if epoch % EVAL_EVERY_EPOCHS == 0:
            val_logits, val_targets = run_eval(model, val_loader, device)

            if THRESHOLD_STRATEGY == "global":
                thresholds, _ = search_thresholds(val_logits, val_targets, strategy="global", steps=THRESHOLD_STEPS)
                macro_f1, micro_f1 = f1_from_thresholds(val_logits, val_targets, thresholds)
            else:
                thresholds, macro_f1, micro_f1 = search_thresholds(val_logits, val_targets, strategy="per_class",
                                                                   steps=THRESHOLD_STEPS)

            print(f"[Stage2][Epoch {epoch}] macro_f1={macro_f1:.5f} micro_f1={micro_f1:.5f}")

            current = macro_f1 if MONITOR == "macro_f1" else micro_f1
            improved = current > (best_score + MIN_DELTA)

            if improved:
                best_score = current
                bad_epochs = 0
                ckpt = save_checkpoint_multilabel(OUT_DIR, epoch, model, LABELS_WO_ND, thresholds, macro_f1, micro_f1,
                                                  best=True)
                print(f"New BEST ({MONITOR}={best_score:.5f}). Saved: {ckpt}")
                # Save threshold in a simple text file for quick inference usage
                with open(os.path.join(OUT_DIR, "best_threshold.txt"), "w") as f:
                    f.write(f"{model['threshold']}\n")
            else:
                bad_epochs += 1
                ckpt = save_checkpoint_multilabel(OUT_DIR, epoch, model, LABELS_WO_ND, thresholds, macro_f1, micro_f1,
                                                  best=False)
                print(f"No improvement. Saved: {ckpt} (bad_epochs={bad_epochs}/{EARLY_STOPPING_PATIENCE})")

            if not SAVE_ALL_CHECKPOINTS:
                cleanup_checkpoints(OUT_DIR, keep=MAX_KEEP)

            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping. Best {MONITOR}={best_score:.5f}")
                break

    print(f"[Stage2] Finished. Best {MONITOR}={best_score:.5f}")
    print(f"[Stage2] Best checkpoint: {os.path.join(OUT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
