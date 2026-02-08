import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import SewerMLDataset
from model import DinoV3MultiLabel
from metrics import search_thresholds, f1_from_thresholds
from train_utils import (
    set_seed, cosine_warmup_lr, run_eval,
    SimpleTransform_SEWER_BASE,
    maybe_resume, save_checkpoint_multilabel, cleanup_checkpoints
)

# -------------------------
# New Loss: Asymmetric Loss (ASL)
# -------------------------
class AsymmetricLossMultiLabel(nn.Module):
    """
    Asymmetric Loss (ASL) for multi-label classification.
    Tail-friendly defaults: gamma_neg=4, gamma_pos=1, clip=0.05
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        prob = torch.sigmoid(logits)
        prob_pos = prob
        prob_neg = 1.0 - prob

        if self.clip is not None and self.clip > 0:
            prob_neg = (prob_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(prob_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(prob_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        pt = prob_pos * targets + prob_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal = (1.0 - pt).pow(gamma)

        return -(focal * loss).mean()


# -------------------------
# Training Config (EDIT PATHS IF NEEDED)
# -------------------------
TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

# New output dir so you do not overwrite baseline
OUT_DIR = "outputs_stage2_vit_base_asl_sampler_1epoch_384"

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
RESUME_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\epoch14_macroF1_0.73940_microF1_0.80718.pt"

DEFECT_ONLY = False  # stage2 must be False
SEED = 42

LABELS = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO",
          "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]
NUM_CLASSES = len(LABELS_WO_ND)

FREEZE_BACKBONE = False

IMG_SIZE = 384
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
NUM_WORKERS = 8
end_epoch = 17
# Fine-tune settings
LR = 3.0e-6
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 0.1

USE_AMP = False
GRAD_ACCUM_STEPS = 1

# Threshold search
THRESHOLD_STRATEGY = "per_class"
THRESHOLD_STEPS = 200
EVAL_EVERY_EPOCHS = 1
MONITOR = "macro_f1"  # "macro_f1" or "micro_f1"

EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0001
SAVE_ALL_CHECKPOINTS = True
MAX_KEEP = 5

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]

# Sampler: enable to oversample rare positives
USE_WEIGHTED_SAMPLER = True
SAMPLER_POWER = 0.75  # <1 reduces extreme weights; try 1.0 for stronger oversampling


def make_sample_weights_from_dataset(train_ds: SewerMLDataset, power: float = 1.0) -> torch.Tensor:
    """
    Computes per-sample weights aligned exactly to train_ds ordering.

    Weight per sample = max_c( inv_freq[c] * y[c] ).
    This focuses on the rarest positive label present in the sample.
    """
    y = train_ds.df[train_ds.labels].to_numpy(dtype=np.float32)  # [N,C], labels exclude ND
    pos_counts = y.sum(axis=0) + 1.0
    inv_freq = 1.0 / pos_counts  # rare class => larger

    # Use max to avoid over-favoring multi-positive samples
    w = (y * inv_freq.reshape(1, -1)).max(axis=1)

    # Safety: some rows might have all zeros (rare, but possible). Give them small non-zero weight.
    w = np.clip(w, 1e-3, None)

    if power is not None and power != 1.0:
        w = np.power(w, power)

    return torch.tensor(w, dtype=torch.double)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=True, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    val_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)

    train_ds = SewerMLDataset(TRAIN_CSV, TRAIN_IMAGES, LABELS, transform=train_tf, defect_only=DEFECT_ONLY)
    val_ds = SewerMLDataset(VAL_CSV, VAL_IMAGES, LABELS, transform=val_tf, defect_only=DEFECT_ONLY)

    if USE_WEIGHTED_SAMPLER:
        weights = make_sample_weights_from_dataset(train_ds, power=SAMPLER_POWER)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            batch_size=TRAIN_BATCH_SIZE,
            sampler=sampler,
            shuffle=False,  # must be False when sampler is used
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        print(f"[Sampler] Enabled WeightedRandomSampler. power={SAMPLER_POWER}, n={len(weights)}")
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True
        )
        print("[Sampler] Disabled. Using shuffle=True.")

    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = DinoV3MultiLabel(MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True).to(device)
    if FREEZE_BACKBONE:
        for p in model.backbone.parameters():
            p.requires_grad = False

    # Loss: ASL
    criterion = AsymmetricLossMultiLabel(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # Resume from best
    start_epoch, global_step, best_score, bad_epochs = maybe_resume(
        RESUME_CKPT, model, optimizer, scaler if USE_AMP else None, device
    )

    # Train exactly ONE epoch after resume
    #end_epoch = start_epoch

    # Fresh local schedule for this 1 epoch (avoid LR collapsing due to resumed global_step)
    resume_global_step = global_step
    local_total_steps = len(train_loader) * 1
    local_warmup_steps = int(WARMUP_EPOCHS * len(train_loader))

    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage2-ASL+Sampler] Epoch {epoch}/{end_epoch}")
        optimizer.zero_grad(set_to_none=True)

        for step, (_, x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            local_step = global_step - resume_global_step
            lr = cosine_warmup_lr(local_step, local_total_steps, LR, local_warmup_steps)
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

        # Eval
        if epoch % EVAL_EVERY_EPOCHS == 0:
            val_logits, val_targets = run_eval(model, val_loader, device)

            if THRESHOLD_STRATEGY == "global":
                thresholds, _ = search_thresholds(val_logits, val_targets, strategy="global", steps=THRESHOLD_STEPS)
                macro_f1, micro_f1 = f1_from_thresholds(val_logits, val_targets, thresholds)
            else:
                thresholds, macro_f1, micro_f1 = search_thresholds(
                    val_logits, val_targets, strategy="per_class", steps=THRESHOLD_STEPS
                )

            print(f"[Stage2-ASL+Sampler][Epoch {epoch}] macro_f1={macro_f1:.5f} micro_f1={micro_f1:.5f}")

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
                labels=LABELS_WO_ND,
                thresholds=thresholds,
                macro_f1=macro_f1,
                micro_f1=micro_f1,
                best=improved,
                global_step=global_step,
                best_score=best_score,
                bad_epochs=bad_epochs,
            )
            print(("New BEST. " if improved else "Saved. ") + f"{ckpt} (bad_epochs={bad_epochs}/{EARLY_STOPPING_PATIENCE})")

            if improved:
                with open(os.path.join(OUT_DIR, "best_thresholds.json"), "w") as f:
                    json.dump({label: float(t) for label, t in zip(LABELS_WO_ND, thresholds.tolist())}, f, indent=2)

            if not SAVE_ALL_CHECKPOINTS:
                cleanup_checkpoints(OUT_DIR, keep=MAX_KEEP)

            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping. Best {MONITOR}={best_score:.5f}")
                break

    print(f"[Stage2-ASL+Sampler] Finished. Best {MONITOR}={best_score:.5f}")
    print(f"[Stage2-ASL+Sampler] Best checkpoint: {os.path.join(OUT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
