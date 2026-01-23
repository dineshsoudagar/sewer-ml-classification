import os
import json
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset  # <-- FIX: use Torch Dataset

# Reuse your project modules
from model import DinoV3MultiLabel
from metrics import search_thresholds, f1_from_thresholds
from train_utils import (
    set_seed, cosine_warmup_lr, run_eval,
    SimpleTransform_SEWER_BASE,
    maybe_resume, save_checkpoint_multilabel, cleanup_checkpoints
)

# -------------------------
# Stage-3 Config
# -------------------------
TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"

TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

OUT_DIR = "outputs_stage3_dinov3_small_tail"

MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"
RESUME_CKPT = ""  # set to a ckpt path or leave ""/None

SEED = 42
ND_LABEL = "ND"

LOW_F1_LABELS = ["FO", "RB", "IS", "DE", "IN"]
RARE_EXTRA = []
STAGE3_LABELS = sorted(list(set(LOW_F1_LABELS + RARE_EXTRA)))

ND_NEG_MULT = 1.0
DEFECT_NEG_MULT = 1.0

IMG_SIZE = 384
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
NUM_WORKERS = 8

EPOCHS = 15
LR = 2.0e-5
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1

USE_AMP = False
GRAD_ACCUM_STEPS = 1

USE_ASL = False
USE_POS_WEIGHT = True
POS_WEIGHT_CLAMP = 100.0

THRESHOLD_STRATEGY = "per_class"
THRESHOLD_STEPS = 200

EVAL_EVERY_EPOCHS = 1
MONITOR = "macro_f1"

EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 0.0001

SAVE_ALL_CHECKPOINTS = True
MAX_KEEP = 5

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]

USE_WEIGHTED_SAMPLER = True


# -------------------------
# Optional ASL
# -------------------------
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        prob = torch.sigmoid(logits)
        targets = targets.type_as(logits)

        if self.clip is not None and self.clip > 0:
            prob = torch.clamp(prob, min=self.eps, max=1 - self.eps)
            prob = prob + self.clip
            prob = torch.clamp(prob, max=1.0)

        pos_loss = targets * torch.log(prob.clamp(min=self.eps))
        neg_loss = (1 - targets) * torch.log((1 - prob).clamp(min=self.eps))

        pt_pos = prob
        pt_neg = 1 - prob
        pos_weight = torch.pow(1 - pt_pos, self.gamma_pos)
        neg_weight = torch.pow(1 - pt_neg, self.gamma_neg)

        loss = -(pos_weight * pos_loss + neg_weight * neg_loss)
        return loss.mean()


# -------------------------
# Dataset (Torch)
# -------------------------
class SewerMLDataset_SUBSET(Dataset):
    """
    Multi-label dataset that can optionally KEEP ND rows as negatives while DROPPING ND from targets.
    """
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        labels: list[str],
        transform=None,
        nd_label: str = "ND",
        filter_nd_rows: bool = False,      # for stage3, keep ND rows
        drop_nd_from_labels: bool = True,  # for stage3, don't predict ND
        filename_col: str = "Filename",    # more robust than "first column"
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.nd_label = nd_label

        # Robust filename column selection
        if filename_col in self.df.columns:
            self.image_col = filename_col
        else:
            self.image_col = self.df.columns[0]

        # De-dup columns just in case (prevents row['Filename'] returning a Series)
        if self.df.columns.duplicated().any():
            self.df = self.df.loc[:, ~self.df.columns.duplicated()].copy()

        # Validate columns
        for lab in labels:
            if lab not in self.df.columns:
                raise ValueError(f"Missing label column '{lab}' in {csv_path}")

        if nd_label not in self.df.columns:
            raise ValueError(f"Missing ND label column '{nd_label}' in {csv_path}")

        # Optionally filter ND rows (stage2 behavior). For stage3 we keep ND rows.
        if filter_nd_rows:
            self.df = self.df[self.df[self.nd_label] == 0].reset_index(drop=True)

        # Drop ND from targets (stage3 behavior)
        if drop_nd_from_labels:
            self.labels = [l for l in self.labels if l != self.nd_label]

        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]

        val = row[self.image_col]
        # If duplicated columns slipped in somehow, pandas returns Series
        if isinstance(val, pd.Series):
            val = val.iloc[0]

        img_name = str(val).strip()
        # Prevent accidental multi-line strings from breaking paths
        img_name = img_name.splitlines()[0].strip()

        img_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y = row[self.labels].to_numpy(dtype=np.float32)
        y = torch.from_numpy(y)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img_name, img, y


# -------------------------
# Helpers
# -------------------------
def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nAvailable columns: {list(df.columns)[:30]} ...")


def build_stage3_subset(
    train_csv: str,
    out_csv: str,
    stage3_labels: List[str],
    nd_label: str,
    nd_neg_mult: float,
    defect_neg_mult: float,
    seed: int,
    rb_label: str = "RB",
    rb_only_cap_abs: int = 80000,   # SAFE cap
    rb_only_cap_mult: float = 1.5,  # SAFE cap
) -> Tuple[int, int, int]:
    """
    Positives:
      - keep ALL non-RB positives (any other stage3 label == 1)
      - keep RB-only positives but cap them (safety)
    Negatives:
      - ND==1 sampled ~ nd_neg_mult * n_pos
      - defect negatives (ND==0 and none(stage3_labels)==1) sampled ~ defect_neg_mult * n_pos
    """
    df = pd.read_csv(train_csv)
    _ensure_columns(df, stage3_labels + [nd_label])

    # De-dup columns to avoid any weirdness downstream
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if rb_label not in stage3_labels:
        y = df[stage3_labels].to_numpy(dtype=np.float32)
        pos_mask = (y.sum(axis=1) > 0)
        df_pos = df[pos_mask].copy()
    else:
        non_rb = [l for l in stage3_labels if l != rb_label]
        if len(non_rb) == 0:
            pos_mask = (df[rb_label].to_numpy(dtype=np.int32) == 1)
            df_pos = df[pos_mask].copy()
        else:
            y_non_rb = df[non_rb].to_numpy(dtype=np.float32)
            non_rb_pos_mask = (y_non_rb.sum(axis=1) > 0)

            rb_pos = (df[rb_label].to_numpy(dtype=np.int32) == 1)
            rb_only_mask = rb_pos & (~non_rb_pos_mask)

            df_non_rb_pos = df[non_rb_pos_mask].copy()
            df_rb_only = df[rb_only_mask].copy()

            n_non_rb_pos = len(df_non_rb_pos)

            rb_cap_rel = int(round(rb_only_cap_mult * max(1, n_non_rb_pos)))
            rb_cap = min(rb_only_cap_abs, rb_cap_rel, len(df_rb_only))

            if rb_cap > 0:
                df_rb_only = df_rb_only.sample(n=rb_cap, random_state=seed)
            else:
                df_rb_only = df_rb_only.iloc[0:0]

            df_pos = pd.concat([df_non_rb_pos, df_rb_only], axis=0)

            print(f"[Stage3][Subset] non-RB positives kept: {n_non_rb_pos}")
            print(f"[Stage3][Subset] RB-only available: {int(rb_only_mask.sum())}, kept (cap): {len(df_rb_only)}")

    n_pos = len(df_pos)
    if n_pos == 0:
        raise RuntimeError("No positives found for STAGE3_LABELS. Check your label names / CSV columns.")

    # For defect negatives selection, use full stage3 union mask
    y_all = df[stage3_labels].to_numpy(dtype=np.float32)
    pos_mask_all = (y_all.sum(axis=1) > 0)

    # ND negatives
    df_nd = df[df[nd_label] == 1].copy()
    n_nd_take = int(round(nd_neg_mult * n_pos))
    if n_nd_take > 0:
        n_nd_take = min(n_nd_take, len(df_nd))
        df_nd = df_nd.sample(n=n_nd_take, random_state=seed)
    else:
        df_nd = df_nd.iloc[0:0]
    n_nd = len(df_nd)

    # Other defect negatives
    df_def_neg = df[(df[nd_label] == 0) & (~pos_mask_all)].copy()
    n_def_take = int(round(defect_neg_mult * n_pos))
    if n_def_take > 0:
        n_def_take = min(n_def_take, len(df_def_neg))
        df_def_neg = df_def_neg.sample(n=n_def_take, random_state=seed + 1)
    else:
        df_def_neg = df_def_neg.iloc[0:0]
    n_def = len(df_def_neg)

    df_sub = pd.concat([df_pos, df_nd, df_def_neg], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Ensure no duplicated columns get written
    df_sub = df_sub.loc[:, ~df_sub.columns.duplicated()].copy()

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_sub.to_csv(out_csv, index=False)

    return n_pos, n_nd, n_def


def compute_pos_weight_from_df(df: pd.DataFrame, labels: List[str], clamp: float) -> torch.Tensor:
    _ensure_columns(df, labels)
    y = df[labels].to_numpy(dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = (neg + 1.0) / (pos + 1.0)
    w = np.clip(w, a_min=1.0, a_max=clamp)
    return torch.tensor(w, dtype=torch.float32)


def make_multilabel_sample_weights_from_df(df: pd.DataFrame, labels: list[str], neg_weight: float = 0.2) -> np.ndarray:
    y = df[labels].to_numpy(dtype=np.float32)
    freq = y.sum(axis=0)
    inv = 1.0 / np.clip(freq, a_min=1.0, a_max=None)

    row_sum = y.sum(axis=1)
    w = np.full((len(df),), neg_weight, dtype=np.float32)
    pos_idx = np.where(row_sum > 0)[0]
    if pos_idx.size > 0:
        w[pos_idx] = (y[pos_idx] * inv.reshape(1, -1)).max(axis=1).astype(np.float32)

    w = w / (w.mean() + 1e-8)
    return w


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    subset_csv = os.path.join(OUT_DIR, "stage3_train_subset.csv")
    n_pos, n_nd, n_def = build_stage3_subset(
        train_csv=TRAIN_CSV,
        out_csv=subset_csv,
        stage3_labels=STAGE3_LABELS,
        nd_label=ND_LABEL,
        nd_neg_mult=ND_NEG_MULT,
        defect_neg_mult=DEFECT_NEG_MULT,
        seed=SEED,
        rb_only_cap_abs=80000,
        rb_only_cap_mult=1.5,
    )

    print(f"[Stage3] Labels: {STAGE3_LABELS}")
    print(f"[Stage3] Subset built: {subset_csv}")
    print(f"[Stage3] subset counts: positives={n_pos}, nd_negs={n_nd}, defect_negs={n_def}, total={n_pos + n_nd + n_def}")

    train_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=True, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    val_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)

    labels_for_dataset = STAGE3_LABELS + [ND_LABEL]
    num_classes = len(STAGE3_LABELS)

    train_ds = SewerMLDataset_SUBSET(
        subset_csv, TRAIN_IMAGES, labels_for_dataset, transform=train_tf,
        nd_label=ND_LABEL, filter_nd_rows=False, drop_nd_from_labels=True
    )
    val_ds = SewerMLDataset_SUBSET(
        VAL_CSV, VAL_IMAGES, labels_for_dataset, transform=val_tf,
        nd_label=ND_LABEL, filter_nd_rows=False, drop_nd_from_labels=True
    )

    if USE_WEIGHTED_SAMPLER:
        w = make_multilabel_sample_weights_from_df(train_ds.df, STAGE3_LABELS, neg_weight=0.2)
        sampler = WeightedRandomSampler(torch.from_numpy(w).double(), num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=TRAIN_BATCH_SIZE, sampler=sampler,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=(NUM_WORKERS > 0)
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=(NUM_WORKERS > 0)
        )

    val_loader = DataLoader(
        val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    model = DinoV3MultiLabel(MODEL_NAME, num_classes=num_classes, pretrained=True).to(device)

    if USE_ASL:
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    else:
        if USE_POS_WEIGHT:
            pos_weight = compute_pos_weight_from_df(train_ds.df, STAGE3_LABELS, POS_WEIGHT_CLAMP).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
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
                labels=STAGE3_LABELS,
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
                    json.dump({label: float(t) for label, t in zip(STAGE3_LABELS, thresholds.tolist())}, f, indent=2)

            if not SAVE_ALL_CHECKPOINTS:
                cleanup_checkpoints(OUT_DIR, keep=MAX_KEEP)

            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"[Stage3] Early stopping. Best {MONITOR}={best_score:.5f}")
                break

    print(f"[Stage3] Finished. Best {MONITOR}={best_score:.5f}")
    print(f"[Stage3] Best checkpoint: {os.path.join(OUT_DIR, 'best.pt')}")
    print(f"[Stage3] Thresholds: {os.path.join(OUT_DIR, 'best_thresholds.json')}")


if __name__ == "__main__":
    main()
