# train_stage3_hard_defneg_resume_vitsmall_384.py
# Resume Stage-3 (vit_small) and fix the real issue by mining HARD DEFECT-NEGATIVES:
#   - mine from ND==0 AND sum(tail_labels)==0
#   - train on: positives + hard_defneg (+ optional small ND negatives)
#   - evaluate on VAL ND==0 only

import os
import json
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DinoV3MultiLabel
from metrics import search_thresholds, f1_from_thresholds
from train_utils import (
    set_seed,
    cosine_warmup_lr,
    run_eval,
    SimpleTransform_SEWER_BASE,
    save_checkpoint_multilabel,
)

# -------------------------
# Config
# -------------------------
LOW_F1_LABELS = ["FO", "RB", "IS", "DE", "IN"]
ND_LABEL = "ND"

TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

OUT_DIR = "outputs_stage3_low_labels_384"
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"

# Resume (MODEL WEIGHTS ONLY)
RESUME_CKPT = r"outputs_stage3_low_labels_384\best.pt"

IMG_SIZE = 384
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 64
NUM_WORKERS = 8

# Continue training up to this epoch number
EPOCHS_TOTAL = 15

# LR
LR = 2.0e-5
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1

USE_AMP = False
GRAD_ACCUM_STEPS = 1

# Hard defect-negative mining
# hard_defneg_count = DEFNEG_MULT * n_pos
DEFNEG_MULT = 1.0
DEFNEG_CAP: Optional[int] = None
HARD_SCORE_MODE = "top2"  # "max" or "top2"
MINE_BATCH_SIZE = 64

# Optional small ND negatives for safety (Stage-1 gates ND, so keep small)
ADD_ND_NEG = False
ND_NEG_MULT = 0.0
ND_NEG_CAP: Optional[int] = 0

# Loss
USE_POS_WEIGHT = True
POS_WEIGHT_CLAMP = 50.0

# Threshold search (eval)
THRESHOLD_STRATEGY = "per_class"
THRESHOLD_STEPS = 200

EVAL_EVERY_EPOCHS = 1
MONITOR = "macro_f1"
EARLY_STOPPING_PATIENCE = 3
MIN_DELTA = 1e-4

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


# -------------------------
# Datasets
# -------------------------
class DFImageDataset(Dataset):
    """
    df must contain:
      - first column: filename
      - LOW_F1_LABELS columns
    """

    def __init__(self, df: pd.DataFrame, images_dir: str, labels: List[str], transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.labels = list(labels)
        self.transform = transform
        self.image_col = self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        img_name = str(row[self.image_col]).strip()
        img_path = os.path.join(self.images_dir, img_name)

        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        y = row[self.labels].to_numpy(dtype=np.float32)
        y = torch.from_numpy(y)

        x = self.transform(image=img)["image"]
        return img_name, x, y


class MineDataset(Dataset):
    """returns (row_idx, tensor)"""

    def __init__(self, df: pd.DataFrame, images_dir: str, transform):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_col = self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        img_name = str(row[self.image_col]).strip()
        img_path = os.path.join(self.images_dir, img_name)

        import cv2
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = self.transform(image=img)["image"]
        return idx, x


# -------------------------
# Helpers
# -------------------------
def ensure_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


def load_model_weights_only(ckpt_path: str, model: nn.Module, device: str) -> Tuple[int, float]:
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = float(ckpt.get("best_score", -1.0))
    print(f"[Resume] Loaded model: {ckpt_path}")
    print(f"[Resume] start_epoch={start_epoch}, best_score={best_score:.5f}")
    return start_epoch, best_score


def compute_pos_weight(df: pd.DataFrame, labels: List[str], clamp: float) -> torch.Tensor:
    y = df[labels].to_numpy(dtype="float32")
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    w = (neg + 1.0) / (pos + 1.0)
    w = np.clip(w, a_min=1.0, a_max=clamp)
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def mine_hard_rows(
        df_candidates: pd.DataFrame,
        images_dir: str,
        model: nn.Module,
        transform,
        device: str,
        k: int,
        batch_size: int,
        num_workers: int,
        score_mode: str = "top2",  # "max" or "top2"
) -> pd.DataFrame:
    model.eval()
    ds = MineDataset(df_candidates, images_dir, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    scores = np.zeros((len(ds),), dtype=np.float32)

    for idxs, x in tqdm(dl, desc="[Mine] scoring", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)  # [B, 5]
        p = torch.sigmoid(logits)  # [B, 5]

        if score_mode == "max":
            s = p.max(dim=1).values
        elif score_mode == "top2":
            s = torch.topk(p, k=2, dim=1).values.mean(dim=1)
        else:
            raise ValueError("score_mode must be 'max' or 'top2'")

        scores[idxs.numpy()] = s.detach().cpu().numpy()

    k = min(k, len(scores))
    top_idx = np.argsort(-scores)[:k]

    hard_df = df_candidates.reset_index(drop=True).iloc[top_idx].copy()
    hard_df["_hard_score"] = scores[top_idx]

    print("[Mine] hard score stats:",
          f"min={float(hard_df['_hard_score'].min()):.4f}",
          f"mean={float(hard_df['_hard_score'].mean()):.4f}",
          f"max={float(hard_df['_hard_score'].max()):.4f}")
    return hard_df


def filter_val_nd0(df_val: pd.DataFrame) -> pd.DataFrame:
    return df_val[df_val[ND_LABEL] == 0].reset_index(drop=True)


# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transforms
    train_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=True, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    eval_tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)

    # Load CSVs
    train_all = pd.read_csv(TRAIN_CSV)
    val_all = pd.read_csv(VAL_CSV)

    ensure_cols(train_all, [ND_LABEL] + LOW_F1_LABELS, "TRAIN")
    ensure_cols(val_all, [ND_LABEL] + LOW_F1_LABELS, "VAL")

    # Model (vit_small)
    model = DinoV3MultiLabel(MODEL_NAME, num_classes=len(LOW_F1_LABELS), pretrained=True).to(device)
    start_epoch, best_score = load_model_weights_only(RESUME_CKPT, model, device)

    if start_epoch > EPOCHS_TOTAL:
        raise ValueError(f"start_epoch={start_epoch} > EPOCHS_TOTAL={EPOCHS_TOTAL}. Increase EPOCHS_TOTAL.")

    # -------------------------
    # Build positives
    # -------------------------
    pos_mask = (train_all[LOW_F1_LABELS].sum(axis=1) > 0)
    pos_df = train_all[(train_all[ND_LABEL]==0)& pos_mask].copy()
    n_pos = len(pos_df)

    # -------------------------
    # Mine HARD DEFECT negatives: ND==0 and tail-absent
    # -------------------------
    defneg_cand = train_all[(train_all[ND_LABEL] == 0) & (~pos_mask)].copy()
    target_defneg = int(round(DEFNEG_MULT * n_pos))
    if DEFNEG_CAP is not None:
        target_defneg = min(target_defneg, int(DEFNEG_CAP))
    target_defneg = min(target_defneg, len(defneg_cand))

    print(f"[Stage3] positives={n_pos}")
    print(f"[Stage3] defneg_candidates={len(defneg_cand)}, target_hard_defneg={target_defneg}")

    if target_defneg > 0:
        hard_defneg = mine_hard_rows(
            df_candidates=defneg_cand,
            images_dir=TRAIN_IMAGES,
            model=model,
            transform=eval_tf,  # deterministic mining
            device=device,
            k=target_defneg,
            batch_size=MINE_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            score_mode=HARD_SCORE_MODE,
        )
        # force targets to 0
        for lab in LOW_F1_LABELS:
            hard_defneg[lab] = 0.0
    else:
        hard_defneg = defneg_cand.iloc[0:0].copy()

    # -------------------------
    # Optional ND negatives (small)
    # -------------------------
    if ADD_ND_NEG:
        nd_cand = train_all[train_all[ND_LABEL] == 1].copy()
        target_nd = int(round(ND_NEG_MULT * n_pos))
        if ND_NEG_CAP is not None:
            target_nd = min(target_nd, int(ND_NEG_CAP))
        target_nd = min(target_nd, len(nd_cand))

        if target_nd > 0:
            nd_samp = nd_cand.sample(n=target_nd, random_state=42).copy()
            for lab in LOW_F1_LABELS:
                nd_samp[lab] = 0.0
        else:
            nd_samp = nd_cand.iloc[0:0].copy()
    else:
        nd_samp = train_all.iloc[0:0].copy()

    # Final train df
    train_df = pd.concat([pos_df, hard_defneg, nd_samp], axis=0).sample(frac=1.0, random_state=42).reset_index(
        drop=True)
    print(
        f"[Stage3] train_total={len(train_df)} (pos={len(pos_df)} + hard_defneg={len(hard_defneg)} + nd_neg={len(nd_samp)})")

    # Val ND==0 only
    val_nd0 = filter_val_nd0(val_all)
    print(f"[Stage3] val_nd0_rows={len(val_nd0)}")

    # Loaders
    train_ds = DFImageDataset(train_df, TRAIN_IMAGES, LOW_F1_LABELS, transform=train_tf)
    val_ds = DFImageDataset(val_nd0, VAL_IMAGES, LOW_F1_LABELS, transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=VAL_BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    # Loss
    if USE_POS_WEIGHT:
        pw = compute_pos_weight(train_df, LOW_F1_LABELS, POS_WEIGHT_CLAMP).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
        print("[Stage3] pos_weight:", {k: float(v) for k, v in zip(LOW_F1_LABELS, pw.detach().cpu().numpy())})
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizer (fresh)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    # LR schedule over remaining epochs
    remaining_epochs = EPOCHS_TOTAL - start_epoch + 1
    total_steps = max(1, remaining_epochs * len(train_loader))
    warmup_steps = int(WARMUP_EPOCHS * len(train_loader))
    global_step = 0
    bad_epochs = 0

    # -------------------------
    # Train loop
    # -------------------------
    for epoch in range(start_epoch, EPOCHS_TOTAL + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"[Stage3-HardDefNeg] Epoch {epoch}/{EPOCHS_TOTAL}")
        optimizer.zero_grad(set_to_none=True)

        for step, (_, x, y) in enumerate(pbar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            lr_now = cosine_warmup_lr(global_step, total_steps, LR, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(x)

            loss = criterion(logits.float(), y.float()) / GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()

            if step % GRAD_ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            pbar.set_postfix(loss=float(loss.item() * GRAD_ACCUM_STEPS), lr=float(lr_now))

        # Eval on VAL ND==0
        if epoch % EVAL_EVERY_EPOCHS == 0:
            val_logits, val_targets = run_eval(model, val_loader, device)

            if THRESHOLD_STRATEGY == "global":
                thr, _ = search_thresholds(val_logits, val_targets, strategy="global", steps=THRESHOLD_STEPS)
                macro_f1, micro_f1 = f1_from_thresholds(val_logits, val_targets, thr)
            else:
                thr, macro_f1, micro_f1 = search_thresholds(val_logits, val_targets, strategy="per_class",
                                                            steps=THRESHOLD_STEPS)

            print(f"[Stage3-HardDefNeg][Epoch {epoch}] macro_f1={macro_f1:.5f} micro_f1={micro_f1:.5f}")

            improved = macro_f1 > (best_score + MIN_DELTA)
            if improved:
                best_score = macro_f1
                bad_epochs = 0
            else:
                bad_epochs += 1

            ckpt_path = save_checkpoint_multilabel(
                out_dir=OUT_DIR,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler if USE_AMP else None,
                model_name=MODEL_NAME,
                img_size=IMG_SIZE,
                labels=LOW_F1_LABELS,
                thresholds=thr,
                macro_f1=macro_f1,
                micro_f1=micro_f1,
                best=improved,
                global_step=global_step,
                best_score=best_score,
                bad_epochs=bad_epochs,
            )
            print((
                      "New BEST. " if improved else "Saved. ") + f"{ckpt_path} (bad_epochs={bad_epochs}/{EARLY_STOPPING_PATIENCE})")

            if improved:
                with open(os.path.join(OUT_DIR, "best_thresholds_stage3_nd0.json"), "w") as f:
                    json.dump({lab: float(t) for lab, t in zip(LOW_F1_LABELS, thr.tolist())}, f, indent=2)

            if bad_epochs >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping. Best macro_f1={best_score:.5f}")
                break

    print(f"[Stage3-HardDefNeg] Finished. Best macro_f1={best_score:.5f}")
    print(f"[Stage3-HardDefNeg] Best checkpoint: {os.path.join(OUT_DIR, 'best.pt')}")


if __name__ == "__main__":
    main()
