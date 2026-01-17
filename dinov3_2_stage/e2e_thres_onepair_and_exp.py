import os
import re
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DinoV3MultiLabel
from train_utils import SimpleTransform

# =========================
# EDIT THESE
# =========================

# Input CSV/images (use VAL for tuning + scoring; use TEST for final submission export)
CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

# Pick EXACT checkpoints (one each)
STAGE1_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_small_plus\best.pt"  # ND gate (1 logit)
STAGE2_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base\best.pt"  # defects (18 logits)

# Must match training backbone
MODEL_NAME_STAGE_1 = "vit_small_plus_patch16_dinov3.lvd1689m"
MODEL_NAME_STAGE_2 = "vit_base_patch16_dinov3.lvd1689m"

# Label order must match CSV columns
LABELS = ["VA","RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK",
          "ND"]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

# Dataloader
IMG_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 8

# Optimize for Kaggle metric (your leaderboard ~= macro)
MONITOR = "macro"  # "macro" or "micro"

# Threshold tuning
TND_COARSE_STEPS = 200
TND_FINE_STEPS = 400
TND_FINE_WINDOW = 0.05

T2_GLOBAL_STEPS = 200

T2_PERCLASS_COARSE_STEPS = 200
T2_PERCLASS_FINE_STEPS = 400
T2_PERCLASS_FINE_WINDOW = 0.05

# Compute
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP_EVAL = True

# Output root
OUT_ROOT = "e2e_exports_2"


# =========================


def _safe_name(p: str) -> str:
    base = os.path.splitext(os.path.basename(p))[0]
    base = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", base)
    return base


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


class SewerMLFullDataset(Dataset):
    """
    Returns:
      img_name, image_tensor, y (if labels exist in CSV) else zeros
    Keeps row order identical to CSV.
    """

    def __init__(self, csv_path: str, images_dir: str, labels: list[str], transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.image_col = self.df.columns[0]

        self.has_labels = all(lab in self.df.columns for lab in labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])
        img_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.has_labels:
            y = row[self.labels].to_numpy(dtype=np.float32)
        else:
            y = np.zeros((len(self.labels),), dtype=np.float32)
        y = torch.from_numpy(y)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img_name, img, y


@torch.no_grad()
def infer_logits(model: nn.Module, loader: DataLoader, device: str, use_amp: bool) -> np.ndarray:
    model.eval()
    all_logits = []
    amp_ctx = torch.amp.autocast("cuda", enabled=use_amp) if device.startswith("cuda") else torch.cpu.amp.autocast(
        enabled=False)

    for _, x, _ in tqdm(loader, desc="Infer", leave=False):
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
        all_logits.append(logits.float().cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def load_model(ckpt_path: str, num_classes: int, MODEL_NAME) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")  # trusted local
    model = DinoV3MultiLabel(MODEL_NAME, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def f1_macro_micro(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray]:
    """
    y_true, y_pred: [N,C] in {0,1}
    Returns (macro, micro, per_class_f1)
    """
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)

    tp = np.logical_and(yt, yp).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~yt, yp).sum(axis=0).astype(np.float64)
    fn = np.logical_and(yt, ~yp).sum(axis=0).astype(np.float64)

    denom = (2 * tp + fp + fn)
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)

    macro = float(f1.mean())

    tp_all = tp.sum()
    fp_all = fp.sum()
    fn_all = fn.sum()
    denom_micro = (2 * tp_all + fp_all + fn_all)
    micro = float((2 * tp_all) / denom_micro) if denom_micro > 0 else 0.0

    return macro, micro, f1.astype(np.float32)


def build_end2end_preds(
        p_nd: np.ndarray,  # [N]
        p2: np.ndarray,  # [N,18]
        t_nd: float,
        t2_per_class: np.ndarray,  # [18]
) -> np.ndarray:
    """
    Returns y_pred_full [N,19] in {0,1}.
    Rule:
      if p_nd >= t_nd => predict ND=1, others 0
      else => ND=0 and stage2 thresholds apply
    """
    nd_idx = LABELS.index(ND_LABEL)
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]

    pred_nd = (p_nd >= t_nd)  # [N]
    defect_mask = ~pred_nd  # [N]

    pred_others = defect_mask[:, None] & (p2 >= t2_per_class.reshape(1, -1))

    y_pred = np.zeros((p_nd.shape[0], len(LABELS)), dtype=np.int32)
    y_pred[:, nd_idx] = pred_nd.astype(np.int32)
    y_pred[:, other_idx] = pred_others.astype(np.int32)
    return y_pred


def end2end_score(p_nd, p2, y_true_full, t_nd, t2_per_class) -> tuple[float, float]:
    y_pred = build_end2end_preds(p_nd, p2, t_nd, t2_per_class)
    macro, micro, _ = f1_macro_micro(y_true_full, y_pred)
    return macro, micro


def best_threshold_binary_f1_masked(probs: np.ndarray, y: np.ndarray, mask: np.ndarray,
                                    coarse_steps: int, fine_window: float, fine_steps: int) -> float:
    """
    Finds threshold maximizing F1 for one class where prediction is only allowed on mask==True.
    pred = mask & (probs >= t)
    """
    best_t, best_f1 = 0.5, -1.0

    # coarse
    for i in range(coarse_steps + 1):
        t = i / coarse_steps
        pred = mask & (probs >= t)

        tp = np.logical_and(y == 1, pred).sum()
        fp = np.logical_and(y == 0, pred).sum()
        fn = np.logical_and(y == 1, ~pred).sum()
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp) / denom if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1, best_t = f1, t

    # fine
    lo = max(0.0, best_t - fine_window)
    hi = min(1.0, best_t + fine_window)
    for i in range(fine_steps + 1):
        t = lo + (hi - lo) * (i / fine_steps)
        pred = mask & (probs >= t)

        tp = np.logical_and(y == 1, pred).sum()
        fp = np.logical_and(y == 0, pred).sum()
        fn = np.logical_and(y == 1, ~pred).sum()
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp) / denom if denom > 0 else 0.0

        if f1 > best_f1:
            best_f1, best_t = f1, t

    return float(best_t)


def tune_stage2_per_class_thresholds(p_nd: np.ndarray, p2: np.ndarray, y_true_full: np.ndarray,
                                     t_nd: float) -> np.ndarray:
    """
    Per-class thresholds for stage2 given fixed stage1 threshold.
    Prediction allowed only on defect_mask = (p_nd < t_nd).
    """
    nd_idx = LABELS.index(ND_LABEL)
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]

    y_true_others = y_true_full[:, other_idx].astype(np.int32)
    defect_mask = (p_nd < t_nd)  # True where stage2 is active

    t = np.full((p2.shape[1],), 0.5, dtype=np.float32)
    for j in range(p2.shape[1]):
        t[j] = best_threshold_binary_f1_masked(
            probs=p2[:, j],
            y=y_true_others[:, j],
            mask=defect_mask,
            coarse_steps=T2_PERCLASS_COARSE_STEPS,
            fine_window=T2_PERCLASS_FINE_WINDOW,
            fine_steps=T2_PERCLASS_FINE_STEPS,
        )
    return t


def tune_tnd_for_end2end(p_nd: np.ndarray, p2: np.ndarray, y_true_full: np.ndarray, t2_per_class: np.ndarray) -> tuple[
    float, float, float]:
    """
    Tune stage1 ND threshold for best end-to-end MONITOR given fixed stage2 per-class thresholds.
    """
    best = {"tnd": 0.5, "macro": -1.0, "micro": -1.0, "score": -1.0}

    def score_fn(macro, micro):
        return macro if MONITOR == "macro" else micro

    # coarse
    for i in range(TND_COARSE_STEPS + 1):
        tnd = i / TND_COARSE_STEPS
        macro, micro = end2end_score(p_nd, p2, y_true_full, float(tnd), t2_per_class)
        sc = score_fn(macro, micro)
        if sc > best["score"]:
            best.update({"tnd": float(tnd), "macro": float(macro), "micro": float(micro), "score": float(sc)})

    # fine
    t0 = best["tnd"]
    lo = max(0.0, t0 - TND_FINE_WINDOW)
    hi = min(1.0, t0 + TND_FINE_WINDOW)
    for i in range(TND_FINE_STEPS + 1):
        tnd = lo + (hi - lo) * (i / TND_FINE_STEPS)
        macro, micro = end2end_score(p_nd, p2, y_true_full, float(tnd), t2_per_class)
        sc = score_fn(macro, micro)
        if sc > best["score"]:
            best.update({"tnd": float(tnd), "macro": float(macro), "micro": float(micro), "score": float(sc)})

    return best["tnd"], best["macro"], best["micro"]


def tune_stage2_global_threshold(p_nd: np.ndarray, p2: np.ndarray, y_true_full: np.ndarray, t_nd: float) -> tuple[
    float, float, float]:
    """
    For clarity: find best single global threshold for stage2 given fixed stage1 threshold.
    """
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]
    y_true_others = y_true_full[:, other_idx].astype(np.int32)
    defect_mask = (p_nd < t_nd)

    best = {"t": 0.5, "macro": -1.0, "micro": -1.0, "score": -1.0}

    def score_fn(macro, micro):
        return macro if MONITOR == "macro" else micro

    for i in range(T2_GLOBAL_STEPS + 1):
        t = i / T2_GLOBAL_STEPS
        pred_others = defect_mask[:, None] & (p2 >= t)
        y_pred = np.zeros_like(y_true_full, dtype=np.int32)
        y_pred[:, LABELS.index(ND_LABEL)] = (p_nd >= t_nd).astype(np.int32)
        y_pred[:, other_idx] = pred_others.astype(np.int32)

        macro, micro, _ = f1_macro_micro(y_true_full, y_pred)
        sc = score_fn(macro, micro)
        if sc > best["score"]:
            best.update({"t": float(t), "macro": float(macro), "micro": float(micro), "score": float(sc)})

    return best["t"], best["macro"], best["micro"]


def save_predictions_csv(df_in: pd.DataFrame, y_pred_full: np.ndarray, out_csv: str):
    """
    Writes CSV with same first column (image) and same label columns, 0/1.
    """
    image_col = df_in.columns[0]
    out = pd.DataFrame()
    out[image_col] = df_in[image_col].astype(str)

    for i, lab in enumerate(LABELS):
        out[lab] = y_pred_full[:, i].astype(np.int32)

    out.to_csv(out_csv, index=False)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    s1_name = _safe_name(STAGE1_CKPT)
    s2_name = _safe_name(STAGE2_CKPT)

    run_dir = os.path.join(OUT_ROOT, f"e2e__s1_{s1_name}__s2_{s2_name}")
    os.makedirs(run_dir, exist_ok=True)

    # Copy selected checkpoints for reproducibility
    shutil.copyfile(STAGE1_CKPT, os.path.join(run_dir, "stage1_selected.pt"))
    shutil.copyfile(STAGE2_CKPT, os.path.join(run_dir, "stage2_selected.pt"))

    # Load data
    df = pd.read_csv(CSV_PATH)
    has_labels = all(lab in df.columns for lab in LABELS)
    if has_labels:
        y_true_full = df[LABELS].to_numpy(dtype=np.int32)
    else:
        y_true_full = None

    tf = SimpleTransform(IMG_SIZE, train=False)
    ds = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Load models + infer logits
    print("Loading models...")
    m1 = load_model(STAGE1_CKPT, num_classes=1, MODEL_NAME=MODEL_NAME_STAGE_1)
    m2 = load_model(STAGE2_CKPT, num_classes=len(LABELS_WO_ND), MODEL_NAME=MODEL_NAME_STAGE_2)

    print("Inferring Stage-1 logits...")
    logits1 = infer_logits(m1, loader, DEVICE, use_amp=USE_AMP_EVAL).reshape(-1)
    p_nd = _sigmoid_np(logits1)

    print("Inferring Stage-2 logits...")
    logits2 = infer_logits(m2, loader, DEVICE, use_amp=USE_AMP_EVAL)
    p2 = _sigmoid_np(logits2)

    # If no labels in CSV, we cannot tune thresholds
    if not has_labels:
        raise RuntimeError(
            "CSV does not contain label columns, so end-to-end threshold tuning is impossible.\n"
            "Use a labeled VAL CSV for tuning. For TEST submission export, tune thresholds on VAL first,\n"
            "then reuse those thresholds when running on TEST."
        )

    # ---- End-to-end tuning (fixed checkpoints) ----
    # Step 1: initialize t_nd (0.5) and get stage2 per-class thresholds
    print("\nTuning Stage-2 per-class thresholds (given t_nd=0.5)...")
    t_nd_init = 0.5
    t2_pc = tune_stage2_per_class_thresholds(p_nd, p2, y_true_full, t_nd_init)

    # Step 2: tune t_nd for end-to-end score
    print("Tuning Stage-1 ND threshold (end-to-end)...")
    t_nd_best, macro_best, micro_best = tune_tnd_for_end2end(p_nd, p2, y_true_full, t2_pc)

    # Step 3: re-tune stage2 thresholds with best t_nd
    print("Re-tuning Stage-2 per-class thresholds with best t_nd...")
    t2_pc = tune_stage2_per_class_thresholds(p_nd, p2, y_true_full, t_nd_best)

    # Step 4: final score with tuned thresholds
    macro_best, micro_best = end2end_score(p_nd, p2, y_true_full, t_nd_best, t2_pc)

    # Also compute best global threshold for clarity (not used for final preds)
    t2_global_best, macro_g, micro_g = tune_stage2_global_threshold(p_nd, p2, y_true_full, t_nd_best)

    print("\n================ RESULT ================")
    print(f"Stage1 ckpt: {os.path.basename(STAGE1_CKPT)}")
    print(f"Stage2 ckpt: {os.path.basename(STAGE2_CKPT)}")
    print(f"Best t_nd:   {t_nd_best:.6f}")
    print(f"End2end macro_f1: {macro_best:.6f}")
    print(f"End2end micro_f1: {micro_best:.6f}")
    print(f"Best stage2 GLOBAL threshold (for reference): {t2_global_best:.6f}")

    # Build final predictions
    y_pred_full = build_end2end_preds(p_nd, p2, t_nd_best, t2_pc)

    # Save predictions CSV (submit this if CSV_PATH is the Kaggle test CSV)
    pred_csv = os.path.join(run_dir, "predictions_end2end.csv")
    save_predictions_csv(df, y_pred_full, pred_csv)

    # Save thresholds JSON
    thr_json = {
        "monitor": MONITOR,
        "csv_used_for_tuning": CSV_PATH,
        "stage1": {
            "checkpoint": os.path.basename(STAGE1_CKPT),
            "nd_threshold": float(t_nd_best),
            "meaning": "predict ND=1 if p(ND)>=nd_threshold else ND=0 and apply stage2 thresholds",
        },
        "stage2": {
            "checkpoint": os.path.basename(STAGE2_CKPT),
            "global_threshold_reference": float(t2_global_best),
            "per_class_thresholds": {lab: float(t) for lab, t in zip(LABELS_WO_ND, t2_pc.tolist())},
            "labels_order_stage2": LABELS_WO_ND,
        },
        "end_to_end_scores_on_csv": {
            "macro_f1": float(macro_best),
            "micro_f1": float(micro_best),
        },
    }
    with open(os.path.join(run_dir, "thresholds_end2end.json"), "w") as f:
        json.dump(thr_json, f, indent=2)

    # Save a compact summary file
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "stage1_ckpt": os.path.basename(STAGE1_CKPT),
                "stage2_ckpt": os.path.basename(STAGE2_CKPT),
                "t_nd": float(t_nd_best),
                "macro_f1": float(macro_best),
                "micro_f1": float(micro_best),
                "pred_csv": os.path.basename(pred_csv),
            },
            f,
            indent=2,
        )

    print("\nSaved outputs to:", run_dir)
    print("  - stage1_selected.pt")
    print("  - stage2_selected.pt")
    print("  - thresholds_end2end.json")
    print("  - predictions_end2end.csv   <-- submit this if this run used TEST csv")
    print("  - summary.json")


if __name__ == "__main__":
    main()
