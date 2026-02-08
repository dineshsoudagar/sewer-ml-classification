import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2

from model import DinoV3MultiLabel
from metrics import search_thresholds, f1_from_thresholds
from train_utils import SimpleTransform_SEWER_BASE

# -------------------------
# EDIT THESE
# -------------------------
SUBSET_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_dinov3_small_tail\stage3_train_subset.csv"
IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"

STAGE3_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_dinov3_small_tail\best.pt"
MODEL_NAME = "vit_small_patch16_dinov3.lvd1689m"

# Must match the model head order used in Stage-3 training
STAGE3_LABELS = ["FO", "RB", "IS", "DE", "IN"]
ND_LABEL = "ND"

IMG_SIZE = 384
BATCH_SIZE = 64
NUM_WORKERS = 8
USE_AMP = True

# Evaluate only on ND==0 rows? Recommended.
EVAL_DEFECT_ONLY = True

# Threshold search
THRESHOLD_STRATEGY = "per_class"  # "per_class" or "global"
THRESHOLD_STEPS = 200

# Dataset stats
SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


# -------------------------


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


class Stage3SubsetDataset(Dataset):
    """
    Loads images + Stage3 labels from subset CSV.
    Can optionally filter to ND==0 only.
    """

    def __init__(self, csv_path: str, images_dir: str, stage3_labels, nd_label="ND",
                 transform=None, defect_only=False):
        self.df = pd.read_csv(csv_path)

        # De-dup columns if needed
        if self.df.columns.duplicated().any():
            self.df = self.df.loc[:, ~self.df.columns.duplicated()].copy()

        if "Filename" in self.df.columns:
            self.image_col = "Filename"
        else:
            self.image_col = self.df.columns[0]

        self.images_dir = images_dir
        self.labels = stage3_labels
        self.nd_label = nd_label
        self.transform = transform

        for c in [self.image_col] + stage3_labels + [nd_label]:
            if c not in self.df.columns:
                raise ValueError(f"Missing column '{c}' in {csv_path}")

        if defect_only:
            self.df = self.df[self.df[nd_label] == 0].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[int(idx)]
        val = row[self.image_col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        img_name = str(val).strip().splitlines()[0].strip()

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


@torch.no_grad()
def infer_logits(model: nn.Module, loader: DataLoader, device: str, use_amp: bool) -> np.ndarray:
    model.eval()
    outs = []

    if device.startswith("cuda"):
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    else:
        amp_ctx = torch.cpu.amp.autocast(enabled=False)

    for _, x, _ in tqdm(loader, desc="Infer", leave=False):
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
        outs.append(logits.float().cpu().numpy())

    return np.concatenate(outs, axis=0)


def f1_pr_recall_per_class(y_true: np.ndarray, y_pred: np.ndarray):
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)

    tp = np.logical_and(yt, yp).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~yt, yp).sum(axis=0).astype(np.float64)
    fn = np.logical_and(yt, ~yp).sum(axis=0).astype(np.float64)

    prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(2 * tp + fp + fn > 0, (2 * tp) / (2 * tp + fp + fn), 0.0)

    macro = float(f1.mean())
    tp_all, fp_all, fn_all = tp.sum(), fp.sum(), fn.sum()
    micro = float((2 * tp_all) / (2 * tp_all + fp_all + fn_all)) if (2 * tp_all + fp_all + fn_all) > 0 else 0.0
    return macro, micro, prec, rec, f1


def load_model(ckpt_path: str, model_name: str, num_classes: int, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = DinoV3MultiLabel(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = SimpleTransform_SEWER_BASE(IMG_SIZE, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    ds = Stage3SubsetDataset(
        SUBSET_CSV, IMAGES_DIR, STAGE3_LABELS, nd_label=ND_LABEL,
        transform=tf, defect_only=EVAL_DEFECT_ONLY
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    print(f"[Eval] rows={len(ds)}  defect_only={EVAL_DEFECT_ONLY}")
    print(f"[Eval] labels={STAGE3_LABELS}")
    print(f"[Eval] ckpt={STAGE3_CKPT}")

    model = load_model(STAGE3_CKPT, MODEL_NAME, num_classes=len(STAGE3_LABELS), device=device)

    logits = infer_logits(model, loader, device, use_amp=USE_AMP)
    probs = _sigmoid_np(logits)

    # Gather targets in same order
    y_true = ds.df[STAGE3_LABELS].to_numpy(dtype=np.int32)

    # Threshold search
    if THRESHOLD_STRATEGY == "global":
        thr, _ = search_thresholds(logits, y_true, strategy="global", steps=THRESHOLD_STEPS)
        macro, micro = f1_from_thresholds(logits, y_true, thr)
        thr_used = thr
    else:
        thr_used, macro, micro = search_thresholds(logits, y_true, strategy="per_class", steps=THRESHOLD_STEPS)

    y_pred = (probs >= thr_used.reshape(1, -1)).astype(np.int32)

    macro2, micro2, prec, rec, f1 = f1_pr_recall_per_class(y_true, y_pred)

    # Diagnostics
    gt_all_zero = int((y_true.sum(axis=1) == 0).sum())
    pred_all_zero = int((y_pred.sum(axis=1) == 0).sum())

    print("\n========== STAGE-3 EVAL ON SUBSET ==========")
    print(f"macro_f1={macro2:.6f}  micro_f1={micro2:.6f}")
    print(f"GT all-zero rows:   {gt_all_zero} / {len(y_true)}")
    print(f"Pred all-zero rows: {pred_all_zero} / {len(y_true)}")
    print("-------------------------------------------")
    print("per-class:")
    for i, lab in enumerate(STAGE3_LABELS):
        sup = int(y_true[:, i].sum())
        print(f"  {lab:>2s}  support={sup:6d}  prec={prec[i]:.4f}  rec={rec[i]:.4f}  f1={f1[i]:.4f}")

    print("-------------------------------------------")
    if THRESHOLD_STRATEGY == "per_class":
        print("thresholds:")
        for lab, t in zip(STAGE3_LABELS, thr_used.tolist()):
            print(f"  {lab}: {float(t):.6f}")

        out_thr = os.path.join(os.path.dirname(STAGE3_CKPT), "eval_thresholds_on_subset.json")
        with open(out_thr, "w") as f:
            json.dump({lab: float(t) for lab, t in zip(STAGE3_LABELS, thr_used.tolist())}, f, indent=2)
        print(f"\nSaved thresholds: {out_thr}")

    print("===========================================")


if __name__ == "__main__":
    main()
