import os

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SewerMLDataset
from model import DinoV3MultiLabel
from metrics import binary_search_threshold_for_f1, binary_metrics_from_logits
from train_utils import (
    set_seed, cosine_warmup_lr, run_eval,
    SimpleTransform, SimpleTransform_SEWER_BASE,
    maybe_resume, save_checkpoint_binary, cleanup_checkpoints
)

# -------------------------
# Training Config
# -------------------------
TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

OUT_DIR = "outputs_stage1_vit_small_plus_with_ND_1_defect"

MODEL_NAME = "vit_small_plus_patch16_dinov3.lvd1689m"
RESUME_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_small_plus_with_ND_1_defect\best.pt"

DEFECT_ONLY = True  # stage1 must be True
SEED = 42
LABELS = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA",

          "ND"]

NUM_CLASSES = 1  # stage1 output is 1 logit
FREEZE_BACKBONE = False

IMG_SIZE = 256
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 64
NUM_WORKERS = 8

EPOCHS = 10
LR = 1.0e-5
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1

USE_AMP = False
GRAD_ACCUM_STEPS = 1

THRESHOLD_STEPS = 200
EVAL_EVERY_EPOCHS = 1

EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0001

SAVE_ALL_CHECKPOINTS = True
MAX_KEEP = 5

PREDICT_DEFECT_PRESENT = True   # True: y = 1 - ND (defect present is positive)
USE_POS_WEIGHT_GATE = True      # optional but recommended for imbalance

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=True, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    val_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)

    train_ds = SewerMLDataset(TRAIN_CSV, TRAIN_IMAGES, LABELS, transform=train_tf, defect_only=DEFECT_ONLY)
    val_ds = SewerMLDataset(VAL_CSV, VAL_IMAGES, LABELS, transform=val_tf, defect_only=DEFECT_ONLY)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = DinoV3MultiLabel(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True).to(device)
    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # --- Loss (optionally switch target so positive means "defect present") ---
    if PREDICT_DEFECT_PRESENT and USE_POS_WEIGHT_GATE:
        # train_ds.df still contains the full CSV columns including ND
        nd = train_ds.df["ND"].to_numpy(dtype=np.float32)  # 1=no defect, 0=defect
        defect = 1.0 - nd  # 1=defect present, 0=no defect

        pos = float(defect.sum())
        neg = float(len(defect) - pos)

        # pos_weight = neg/pos (standard for BCEWithLogitsLoss)
        eps = 1e-6
        pos_weight = torch.tensor([neg / (pos + eps)], dtype=torch.float32, device=device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"[Stage1] Using DEFECT_PRESENT target. pos={pos:.0f} neg={neg:.0f} pos_weight={pos_weight.item():.3f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        if PREDICT_DEFECT_PRESENT:
            print("[Stage1] Using DEFECT_PRESENT target (y = 1 - ND), but WITHOUT pos_weight.")
        else:
            print("[Stage1] Using ND target directly (ND=1 is positive).")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_EPOCHS * len(train_loader))

    start_epoch, global_step, best_f1, bad_epochs = maybe_resume(
        RESUME_CKPT, model, optimizer, scaler if USE_AMP else None, device
    )

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage1] Epoch {epoch}/{EPOCHS}")
        optimizer.zero_grad(set_to_none=True)

        for step, (_, x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1, 1)  # [B,1]
            if PREDICT_DEFECT_PRESENT:
                y = 1.0 - y
            lr = cosine_warmup_lr(global_step, total_steps, LR, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(x)

            loss = criterion(logits.float(), y.float()) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item() * GRAD_ACCUM_STEPS), lr=lr)

        if epoch % EVAL_EVERY_EPOCHS == 0:
            val_logits, val_targets = run_eval(model, val_loader, device)
            if PREDICT_DEFECT_PRESENT:
                val_targets = 1.0 - val_targets
            best_t, _ = binary_search_threshold_for_f1(val_logits, val_targets, steps=THRESHOLD_STEPS)
            m = binary_metrics_from_logits(val_logits, val_targets, threshold=best_t)

            print(
                f"[Stage1][Epoch {epoch}] bce={m['bce']:.5f} acc={m['acc']:.5f} f1={m['f1']:.5f} thr={m['threshold']:.3f}")

            improved = m["f1"] > (best_f1 + MIN_DELTA)
            if improved:
                best_f1 = m["f1"]
                bad_epochs = 0
            else:
                bad_epochs += 1

            ckpt = save_checkpoint_binary(
                out_dir=OUT_DIR,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler if USE_AMP else None,
                model_name=MODEL_NAME,
                img_size=IMG_SIZE,
                labels=["DEFECT_PRESENT"] if PREDICT_DEFECT_PRESENT else ["ND"],
                threshold=m["threshold"],
                f1=m["f1"],
                acc=m["acc"],
                best=improved,
                global_step=global_step,
                best_score=best_f1,
                bad_epochs=bad_epochs,
            )
            print((
                      "New BEST. " if improved else "Saved. ") + f"{ckpt} (bad_epochs={bad_epochs}/{EARLY_STOPPING_PATIENCE})")

            if improved:
                with open(os.path.join(OUT_DIR, "best_threshold.txt"), "w") as f:
                    f.write(f"{m['threshold']}\n")

            if not SAVE_ALL_CHECKPOINTS:
                cleanup_checkpoints(OUT_DIR, keep=MAX_KEEP)

            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping. Best f1={best_f1:.5f}")
                break

    print(f"[Stage1] Finished. Best f1={best_f1:.5f}")
    print(f"[Stage1] Best checkpoint: {os.path.join(OUT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
