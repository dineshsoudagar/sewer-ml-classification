# train_stage3_low_labels.py
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DinoV3MultiLabel
from metrics import search_thresholds, f1_from_thresholds
from train_utils import (
    set_seed, cosine_warmup_lr, run_eval,
    SimpleTransform,
    maybe_resume, save_checkpoint_multilabel, cleanup_checkpoints
)

# -------------------------
# Stage-3 Config
# -------------------------
LOW_F1_LABELS = ["FO", "RB", "IS", "DE", "IN"]
ND_LABEL = "ND"

# Paths (adjust)
TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

OUT_DIR = "outputs_stage3_low_labels_384"

MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"

# Option A: resume stage3 training
RESUME_CKPT = r"outputs_stage3_low_labels_384\best.pt"

# Option B: init backbone from a stage2 checkpoint (recommended)
INIT_BACKBONE_FROM_CKPT = None  # e.g. r"outputs_stage2_vit_base\best.pt"

IMG_SIZE = 384
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
NUM_WORKERS = 8

EPOCHS = 10
LR = 1.0e-5
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 0.1

USE_AMP = False
GRAD_ACCUM_STEPS = 1

FREEZE_BACKBONE = False

# Add ND negatives into Stage-3 train set:
# - "multiplier": how many ND samples to add relative to #positive samples
#   e.g. 1.0 => add ~same number of ND as positive
# - "cap": optional max ND samples
NEG_ND_MULTIPLIER = 1.0
NEG_ND_CAP = None  # e.g. 100_000

USE_POS_WEIGHT = True
POS_WEIGHT_CLAMP = 50.0

THRESHOLD_STRATEGY = "per_class"  # "global" or "per_class"
THRESHOLD_STEPS = 200

EVAL_EVERY_EPOCHS = 1
MONITOR = "macro_f1"  # "macro_f1" or "micro_f1"

EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0001

SAVE_ALL_CHECKPOINTS = True
MAX_KEEP = 5

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


# -------------------------
# Dataset: train on LOW_F1_LABELS only, with optional ND negatives
# -------------------------
class LowLabelStage3Dataset(Dataset):
    """
    - Train split:
        keep rows where any LOW_F1_LABELS == 1 (positives)
        plus sampled ND==1 rows (negatives => all-zero targets)
    - Val split:
        keep ALL rows, targets are LOW_F1_LABELS only
    """

    def __init__(
            self,
            csv_path: str,
            images_dir: str,
            low_labels: list[str],
            transform=None,
            nd_label: str = "ND",
            mode: str = "train",  # "train" or "val"
            neg_nd_multiplier: float = 1.0,
            neg_nd_cap: int | None = None,
            seed: int = 42,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.low_labels = list(low_labels)
        self.transform = transform
        self.nd_label = nd_label
        self.mode = mode
        self.seed = int(seed)

        # First column is image name
        self.image_col = self.df.columns[0]

        # Sanity checks
        for lab in self.low_labels + [self.nd_label]:
            if lab not in self.df.columns:
                raise ValueError(f"Missing column '{lab}' in {csv_path}")

        if self.mode == "train":
            # positives: any low label present
            pos_mask = (self.df[self.low_labels].sum(axis=1) > 0)
            pos_df = self.df[pos_mask].copy()

            # ND negatives (all-zero targets for low labels)
            nd_df = self.df[self.df[self.nd_label] == 1].copy()

            # sample ND rows
            n_pos = len(pos_df)
            target_n_nd = int(round(neg_nd_multiplier * n_pos))
            if neg_nd_cap is not None:
                target_n_nd = min(target_n_nd, int(neg_nd_cap))
            target_n_nd = min(target_n_nd, len(nd_df))

            if target_n_nd > 0:
                nd_df = nd_df.sample(n=target_n_nd, random_state=self.seed).copy()
                self.df = pd.concat([pos_df, nd_df], axis=0).reset_index(drop=True)
            else:
                self.df = pos_df.reset_index(drop=True)

            # shuffle combined
            self.df = self.df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

            print(f"[Stage3][Dataset] train positives={n_pos}, nd_negatives={target_n_nd}, total={len(self.df)}")
        else:
            # val: keep all rows to measure false positives on ND and non-target defects too
            print(f"[Stage3][Dataset] val total={len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])
        img_path = os.path.join(self.images_dir, img_name)

        # cv2 read (to stay consistent with your existing dataset pipeline)
        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y = row[self.low_labels].to_numpy(dtype=np.float32)
        y = torch.from_numpy(y)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img_name, img, y


def compute_pos_weight_stage3(train_df: pd.DataFrame, low_labels: list[str], clamp: float) -> torch.Tensor:
    """
    pos_weight = (neg+1)/(pos+1), computed over the *actual Stage-3 train dataframe*
    (after adding ND negatives and filtering).
    """
    y = train_df[low_labels].to_numpy(dtype="float32")
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = (neg + 1.0) / (pos + 1.0)
    w = w.clip(max=clamp)
    return torch.tensor(w, dtype=torch.float32)


def init_backbone_from_ckpt(model: DinoV3MultiLabel, ckpt_path: str, device: str):
    """
    Loads backbone weights from a checkpoint (e.g. stage2 best.pt).
    Head weights are ignored if shapes mismatch.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)

    # Keep only backbone.* keys
    filtered = {k: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[Stage3][Init] Loaded backbone from: {ckpt_path}")
    print(f"[Stage3][Init] missing={len(missing)} unexpected={len(unexpected)}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = SimpleTransform(IMG_SIZE, train=True, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    val_tf = SimpleTransform(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)

    # Build datasets
    train_ds = LowLabelStage3Dataset(
        csv_path=TRAIN_CSV,
        images_dir=TRAIN_IMAGES,
        low_labels=LOW_F1_LABELS,
        transform=train_tf,
        nd_label=ND_LABEL,
        mode="train",
        neg_nd_multiplier=NEG_ND_MULTIPLIER,
        neg_nd_cap=NEG_ND_CAP,
        seed=42,
    )
    val_ds = LowLabelStage3Dataset(
        csv_path=VAL_CSV,
        images_dir=VAL_IMAGES,
        low_labels=LOW_F1_LABELS,
        transform=val_tf,
        nd_label=ND_LABEL,
        mode="val",
        seed=42,
    )

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # Model
    model = DinoV3MultiLabel(MODEL_NAME, num_classes=len(LOW_F1_LABELS), pretrained=True).to(device)

    # Optional: initialize backbone from stage2
    if INIT_BACKBONE_FROM_CKPT:
        init_backbone_from_ckpt(model, INIT_BACKBONE_FROM_CKPT, device)

    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Loss
    if USE_POS_WEIGHT:
        # compute pos_weight over the *effective* Stage-3 train set
        w = compute_pos_weight_stage3(train_ds.df, LOW_F1_LABELS, POS_WEIGHT_CLAMP).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=w)
        print(f"[Stage3] pos_weight: {dict(zip(LOW_F1_LABELS, w.detach().cpu().numpy().round(3).tolist()))}")
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    total_steps = EPOCHS * len(train_loader)
    warmup_steps = int(WARMUP_EPOCHS * len(train_loader))

    start_epoch, global_step, best_score, bad_epochs = maybe_resume(
        RESUME_CKPT, model, optimizer, scaler if USE_AMP else None, device
    )

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage3] Epoch {epoch}/{EPOCHS}")
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
                thresholds, macro_f1, micro_f1 = search_thresholds(
                    val_logits, val_targets, strategy="per_class", steps=THRESHOLD_STEPS
                )

            print(f"[Stage3][Epoch {epoch}] macro_f1={macro_f1:.5f} micro_f1={micro_f1:.5f}")

            current = macro_f1 if MONITOR == "macro_f1" else micro_f1
            improved = current > (best_score + MIN_DELTA)

            if improved:
                best_score = current
                bad_epochs = 0
            else:
                bad_epochs += 1

            ckpt = save_checkpoint_multilabel(
                out_dir=OUT_DIR,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler if USE_AMP else None,
                model_name=MODEL_NAME,
                img_size=IMG_SIZE,
                labels=LOW_F1_LABELS,
                thresholds=thresholds,
                macro_f1=macro_f1,
                micro_f1=micro_f1,
                best=improved,
                global_step=global_step,
                best_score=best_score,
                bad_epochs=bad_epochs,
            )
            print((
                      "New BEST. " if improved else "Saved. ") + f"{ckpt} (bad_epochs={bad_epochs}/{EARLY_STOPPING_PATIENCE})")

            if improved:
                with open(os.path.join(OUT_DIR, "best_thresholds.json"), "w") as f:
                    json.dump({lab: float(t) for lab, t in zip(LOW_F1_LABELS, thresholds.tolist())}, f, indent=2)

            if not SAVE_ALL_CHECKPOINTS:
                cleanup_checkpoints(OUT_DIR, keep=MAX_KEEP)

            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping. Best {MONITOR}={best_score:.5f}")
                break

    print(f"[Stage3] Finished. Best {MONITOR}={best_score:.5f}")
    print(f"[Stage3] Best checkpoint: {os.path.join(OUT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
