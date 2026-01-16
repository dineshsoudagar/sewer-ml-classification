import os
import math
import random
import shutil
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_utils import SimpleTransform
from dataset import SewerMLDataset
from model import DinoV3MultiLabel
from metrics import binary_search_threshold_for_f1, binary_metrics_from_logits

# Training Config
TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train_gate_sanity_5k.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train_gate_sanity_5k.csv"
TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
RESUME_CKPT = None
MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
OUT_DIR = "outputs_stage1_vit_base_tesk_5k"
DEFECT_ONLY = True
SEED = 42
LABELS = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA",
          "ND"]
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
THRESHOLD_STEPS = 200
EVAL_EVERY_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0001
SAVE_ALL_CHECKPOINTS = True
MAX_KEEP = 5
# =========================
SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


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


def save_checkpoint_binary(
    out_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    threshold: float,
    f1: float,
    acc: float,
    best: bool,
    global_step: int,
    best_f1: float,
    bad_epochs: int,
):
    fname = f"epoch{epoch:02d}_f1_{f1:.5f}_acc_{acc:.5f}.pt"
    path = os.path.join(out_dir, fname)

    payload = {
        "epoch": epoch,
        "global_step": int(global_step),
        "best_f1": float(best_f1),
        "bad_epochs": int(bad_epochs),

        "model_name": MODEL_NAME,
        "labels": ["ND"],
        "img_size": IMG_SIZE,

        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,

        "threshold": float(threshold),
        "f1": float(f1),
        "acc": float(acc),
    }
    torch.save(payload, path)

    if best:
        shutil.copyfile(path, os.path.join(out_dir, "best.pt"))

    return path


def cleanup_checkpoints(out_dir: str, keep: int):
    files = [f for f in os.listdir(out_dir) if f.endswith(".pt") and f.startswith("epoch")]
    if len(files) <= keep:
        return
    files.sort()  # keep latest by filename
    for f in files[:-keep]:
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = SimpleTransform(IMG_SIZE, train=True, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    val_tf = SimpleTransform(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)

    # Stage-1: ND gate (dataset returns ND scalar)
    train_ds = SewerMLDataset(TRAIN_CSV, TRAIN_IMAGES, LABELS, transform=train_tf, defect_only=DEFECT_ONLY)
    val_ds = SewerMLDataset(VAL_CSV, VAL_IMAGES, LABELS, transform=val_tf, defect_only=DEFECT_ONLY)

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Model: 1 output logit
    model = DinoV3MultiLabel(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True).to(device)
    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_EPOCHS * len(train_loader))

    best_f1 = -1.0
    bad_epochs = 0
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage1] Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for step, (_, x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)  # [B,1]

            lr = cosine_warmup_lr(global_step, total_steps, LR, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # AMP forward, FP32 loss (more stable)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(x)
            loss = criterion(logits.float(), y.float()) / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0:
                # Optional (keep off unless you need it):
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

            best_t, _ = binary_search_threshold_for_f1(val_logits, val_targets, steps=THRESHOLD_STEPS)
            m = binary_metrics_from_logits(val_logits, val_targets, threshold=best_t)

            print(
                f"[Stage1][Epoch {epoch}] bce={m['bce']:.5f} acc={m['acc']:.5f} f1={m['f1']:.5f} thr={m['threshold']:.3f}")

            improved = m["f1"] > (best_f1 + MIN_DELTA)
            if improved:
                best_f1 = m["f1"]
                bad_epochs = 0
                ckpt = save_checkpoint_binary(OUT_DIR, epoch, model, m["threshold"], m["f1"], m["acc"], best=True)
                print(f"New BEST f1={best_f1:.5f}. Saved: {ckpt}")
                # Save threshold in a simple text file for quick inference usage
                with open(os.path.join(OUT_DIR, "best_threshold.txt"), "w") as f:
                    f.write(f"{m['threshold']}\n")
            else:
                bad_epochs += 1
                ckpt = save_checkpoint_binary(OUT_DIR, epoch, model, m["threshold"], m["f1"], m["acc"], best=False)
                print(f"No improvement. Saved: {ckpt} (bad_epochs={bad_epochs}/{EARLY_STOPPING_PATIENCE})")

            if not SAVE_ALL_CHECKPOINTS:
                cleanup_checkpoints(OUT_DIR, keep=MAX_KEEP)

            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping. Best f1={best_f1:.5f}")
                break

    print(f"[Stage1] Finished. Best f1={best_f1:.5f}")
    print(f"[Stage1] Best checkpoint: {os.path.join(OUT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
