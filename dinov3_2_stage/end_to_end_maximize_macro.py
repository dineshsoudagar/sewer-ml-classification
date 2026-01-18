import os
import re
import json
import shutil
import argparse
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

CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

STAGE1_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_small_plus\best.pt"
STAGE2_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base\best.pt"

MODEL_NAME_STAGE_1 = "vit_small_plus_patch16_dinov3.lvd1689m"
MODEL_NAME_STAGE_2 = "vit_base_patch16_dinov3.lvd1689m"

RAW_LOGITS = None
RAW_PROBS = None

# IMPORTANT: must match CSV columns order for y_true_full extraction
LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"
]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

# Stage2 ckpt head order (confirmed)
STAGE2_LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA"
]

IMG_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 8

MONITOR = "macro"  # "macro" or "micro"

TND_COARSE_STEPS = 200
TND_FINE_STEPS = 400
TND_FINE_WINDOW = 0.05

T2_GLOBAL_STEPS = 200

T2_PERCLASS_COARSE_STEPS = 200
T2_PERCLASS_FINE_STEPS = 400
T2_PERCLASS_FINE_WINDOW = 0.05

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP_EVAL = True

OUT_ROOT = "e2e_exports_4"

# --- Policy export set (one run -> all of these saved) ---
POLICY_CONFIGS = [
    {"name": "hard_gate", "policy": "hard_gate"},
    {"name": "union", "policy": "union"},
    # Override: allow stage2 to override stage1 ND when strong stage2 evidence exists
    {"name": "override_m0p05_k1", "policy": "override", "margin": 0.05, "k": 1, "t_lo": None, "t_hi": None},
    # Scaled: down-weight stage2 when p_nd is high
    {"name": "scaled_b2", "policy": "scaled", "beta": 2.0},
]

# Save force variants per policy (creates 3 CSVs per policy: base, forceND, forceONE)
SAVE_FORCE_VARIANTS_PER_POLICY = True

# =========================


def _float_tag(x: float, ndigits: int = 6) -> str:
    return f"{x:.{ndigits}f}".replace(".", "p")


def _safe_name(p: str) -> str:
    base = os.path.splitext(os.path.basename(p))[0]
    base = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", base)
    return base


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def save_raw_probs_csv(
    df_in: pd.DataFrame,
    p_nd: np.ndarray,              # [N] in [0,1]
    p2: np.ndarray,                # [N,18] in [0,1] stage2 order
    out_csv: str,
    stage2_labels: list[str],
    nd_label: str = "ND",
):
    image_col = df_in.columns[0]
    out = pd.DataFrame()
    out[image_col] = df_in[image_col].astype(str)

    for j, lab in enumerate(stage2_labels):
        out[f"p_{lab}"] = p2[:, j].astype(np.float32)

    out[f"p_{nd_label}"] = p_nd.astype(np.float32)
    out.to_csv(out_csv, index=False)


def save_raw_logits_csv(
    df_in: pd.DataFrame,
    logits1: np.ndarray,           # [N]
    logits2: np.ndarray,           # [N,18]
    out_csv: str,
    stage2_labels: list[str],
    nd_label: str = "ND",
):
    image_col = df_in.columns[0]
    out = pd.DataFrame()
    out[image_col] = df_in[image_col].astype(str)

    for j, lab in enumerate(stage2_labels):
        out[f"logit_{lab}"] = logits2[:, j].astype(np.float32)

    out[f"logit_{nd_label}"] = logits1.reshape(-1).astype(np.float32)
    out.to_csv(out_csv, index=False)


def load_raw_probs_for_df(raw_probs_csv: str, df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads p_nd and p2 aligned to df_in row order by image name.
    Expects columns: p_<RB>..p_<VA>, p_ND, plus image column (same first col name as df_in).
    """
    df_raw = pd.read_csv(raw_probs_csv)
    image_col = df_in.columns[0]
    if image_col not in df_raw.columns:
        raise RuntimeError(f"[raw_probs] Missing image column '{image_col}' in {raw_probs_csv}")

    need = [f"p_{lab}" for lab in STAGE2_LABELS] + [f"p_{ND_LABEL}"]
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise RuntimeError(f"[raw_probs] Missing columns in {raw_probs_csv}: {missing}")

    # left-join to preserve df_in order
    df_join = df_in[[image_col]].merge(df_raw[[image_col] + need], on=image_col, how="left", validate="1:1")

    if df_join[need].isna().any().any():
        n_bad = int(df_join[need].isna().any(axis=1).sum())
        raise RuntimeError(f"[raw_probs] {n_bad} rows missing after merge. "
                           f"Ensure raw_probs was generated for the same CSV/images set.")

    p_nd = df_join[f"p_{ND_LABEL}"].to_numpy(dtype=np.float32)
    p2 = np.stack([df_join[f"p_{lab}"].to_numpy(dtype=np.float32) for lab in STAGE2_LABELS], axis=1)
    return p_nd, p2


def load_raw_logits_for_df(raw_logits_csv: str, df_in: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads logits1 and logits2 aligned to df_in row order by image name.
    Expects columns: logit_<RB>..logit_<VA>, logit_ND, plus image column.
    """
    df_raw = pd.read_csv(raw_logits_csv)
    image_col = df_in.columns[0]
    if image_col not in df_raw.columns:
        raise RuntimeError(f"[raw_logits] Missing image column '{image_col}' in {raw_logits_csv}")

    need = [f"logit_{lab}" for lab in STAGE2_LABELS] + [f"logit_{ND_LABEL}"]
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise RuntimeError(f"[raw_logits] Missing columns in {raw_logits_csv}: {missing}")

    df_join = df_in[[image_col]].merge(df_raw[[image_col] + need], on=image_col, how="left", validate="1:1")
    if df_join[need].isna().any().any():
        n_bad = int(df_join[need].isna().any(axis=1).sum())
        raise RuntimeError(f"[raw_logits] {n_bad} rows missing after merge. "
                           f"Ensure raw_logits was generated for the same CSV/images set.")

    logits1 = df_join[f"logit_{ND_LABEL}"].to_numpy(dtype=np.float32).reshape(-1)
    logits2 = np.stack([df_join[f"logit_{lab}"].to_numpy(dtype=np.float32) for lab in STAGE2_LABELS], axis=1)
    return logits1, logits2


def force_nd_if_all_zero(y_pred_full: np.ndarray, labels: list[str], nd_label: str) -> tuple[np.ndarray, np.ndarray]:
    """
    If a row has no positive label at all (sum==0), set ND=1 for that row.
    Returns: (y_pred_forced, mask_rows_forced)
    """
    y = y_pred_full.astype(np.int32, copy=True)
    nd_idx = labels.index(nd_label)

    row_sum = y.sum(axis=1)
    mask = (row_sum == 0)
    if mask.any():
        y[mask, nd_idx] = 1

    return y, mask


def force_one_label_by_maxprob(
    y_pred_full: np.ndarray,
    p_nd: np.ndarray,          # [N]
    p2: np.ndarray,            # [N,18]
    labels_full: list[str],
    labels_stage2: list[str],  # stage2 head order
    nd_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    If a row has all zeros, force exactly one label:
      - ND if p_nd >= max(p2)
      - else the defect argmax
    Returns (y_pred_forced, mask_rows_forced)
    """
    y = y_pred_full.astype(np.int32, copy=True)
    nd_idx = labels_full.index(nd_label)

    row_sum = y.sum(axis=1)
    mask = (row_sum == 0)
    if not mask.any():
        return y, mask

    mask_idx = np.where(mask)[0]
    p2_max = p2[mask].max(axis=1)
    choose_nd = (p_nd[mask] >= p2_max)

    y[mask_idx[choose_nd], nd_idx] = 1

    rest_idx = mask_idx[~choose_nd]
    if rest_idx.size > 0:
        full_index = {lab: labels_full.index(lab) for lab in labels_stage2}
        j = p2[rest_idx].argmax(axis=1)
        for ii, jj in zip(rest_idx.tolist(), j.tolist()):
            lab = labels_stage2[int(jj)]
            y[ii, full_index[lab]] = 1

    return y, mask


class SewerMLFullDataset(Dataset):
    """
    Returns img_name, image_tensor, y (if labels exist) else zeros.
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

    if device.startswith("cuda"):
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    else:
        amp_ctx = torch.cpu.amp.autocast(enabled=False)

    for _, x, _ in tqdm(loader, desc="Infer", leave=False):
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
        all_logits.append(logits.float().cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def load_model(ckpt_path: str, num_classes: int, model_name: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "labels" in ckpt:
        print(f"[CKPT] {model_name} labels:", ckpt["labels"])
    model = DinoV3MultiLabel(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def f1_macro_micro(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray]:
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


def best_threshold_binary_f1_masked(
    probs: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    coarse_steps: int,
    fine_window: float,
    fine_steps: int
) -> float:
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


def tune_stage2_per_class_thresholds(
    p_nd: np.ndarray,
    p2: np.ndarray,
    y_true_full: np.ndarray,
    t_nd: float
) -> np.ndarray:
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]
    y_true_others = y_true_full[:, other_idx].astype(np.int32)
    defect_mask = (p_nd < t_nd)

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


def build_preds_arbitrated(
    p_nd: np.ndarray,            # [N] prob ND
    p2: np.ndarray,              # [N,18] prob defects in STAGE2_LABELS order
    t_nd: float,
    t2_pc: np.ndarray,           # [18]
    policy: str,
    margin: float = 0.05,        # override only
    k: int = 1,                  # override only
    t_lo: float | None = None,   # override only
    t_hi: float | None = None,   # override only
    beta: float = 2.0,           # scaled only
) -> np.ndarray:
    """
    Returns y_pred [N,19] in LABELS order.
    Policies:
      - hard_gate: if p_nd>=t_nd => ND=1, else stage2
      - union: use stage2 everywhere; if any defect fires => ND=0 else ND by stage1
      - override: hard_gate but allow overriding ND when stage2 is clearly confident
      - scaled: p2_scaled = p2*(1-p_nd)^beta then union-like decision
    """
    N = p_nd.shape[0]
    y = np.zeros((N, len(LABELS)), dtype=np.int32)

    nd_idx = LABELS.index(ND_LABEL)
    full_idx = {lab: LABELS.index(lab) for lab in STAGE2_LABELS}

    if policy == "hard_gate":
        pred_nd = (p_nd >= t_nd)
        defect_mask = ~pred_nd
        y[:, nd_idx] = pred_nd.astype(np.int32)

        pred2 = (p2 >= t2_pc.reshape(1, -1))
        if defect_mask.any():
            for j, lab in enumerate(STAGE2_LABELS):
                y[defect_mask, full_idx[lab]] = pred2[defect_mask, j].astype(np.int32)
        return y

    if policy == "union":
        pred2 = (p2 >= t2_pc.reshape(1, -1))
        any_def = pred2.any(axis=1)
        pred_nd = (p_nd >= t_nd) & (~any_def)
        y[:, nd_idx] = pred_nd.astype(np.int32)

        use_def = any_def
        if use_def.any():
            for j, lab in enumerate(STAGE2_LABELS):
                y[use_def, full_idx[lab]] = pred2[use_def, j].astype(np.int32)
        return y

    if policy == "override":
        pred2 = (p2 >= t2_pc.reshape(1, -1))
        pred_nd = (p_nd >= t_nd)

        if t_lo is not None and t_hi is not None:
            band = (p_nd >= t_lo) & (p_nd < t_hi)
        else:
            band = np.ones_like(pred_nd, dtype=bool)

        delta = (p2 - t2_pc.reshape(1, -1)).max(axis=1)   # [N]
        cnt = pred2.sum(axis=1)                           # [N]
        override = pred_nd & band & ((delta >= margin) | (cnt >= k))

        y[:, nd_idx] = (pred_nd & (~override)).astype(np.int32)

        use_def = (~pred_nd) | override
        if use_def.any():
            for j, lab in enumerate(STAGE2_LABELS):
                y[use_def, full_idx[lab]] = pred2[use_def, j].astype(np.int32)
        return y

    if policy == "scaled":
        scale = np.power(1.0 - p_nd.reshape(-1, 1), beta).astype(np.float32)
        p2s = p2 * scale
        pred2s = (p2s >= t2_pc.reshape(1, -1))
        any_def = pred2s.any(axis=1)

        pred_nd = (p_nd >= t_nd) & (~any_def)
        y[:, nd_idx] = pred_nd.astype(np.int32)

        use_def = any_def | (p_nd < t_nd)
        if use_def.any():
            for j, lab in enumerate(STAGE2_LABELS):
                y[use_def, full_idx[lab]] = pred2s[use_def, j].astype(np.int32)
        return y

    raise ValueError(f"Unknown policy: {policy}")


def end2end_score(p_nd, p2, y_true_full, t_nd, t2_per_class) -> tuple[float, float]:
    y_pred = build_preds_arbitrated(p_nd, p2, t_nd, t2_per_class, policy="hard_gate")
    macro, micro, _ = f1_macro_micro(y_true_full, y_pred)
    return macro, micro


def tune_tnd_for_end2end(
    p_nd: np.ndarray,
    p2: np.ndarray,
    y_true_full: np.ndarray,
    t2_per_class: np.ndarray
) -> tuple[float, float, float]:
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


def tune_stage2_global_threshold(
    p_nd: np.ndarray,
    p2: np.ndarray,
    y_true_full: np.ndarray,
    t_nd: float
) -> tuple[float, float, float]:
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]
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

    shutil.copyfile(STAGE1_CKPT, os.path.join(run_dir, "stage1_selected.pt"))
    shutil.copyfile(STAGE2_CKPT, os.path.join(run_dir, "stage2_selected.pt"))

    df = pd.read_csv(CSV_PATH)
    has_labels = all(lab in df.columns for lab in LABELS)
    if not has_labels:
        raise RuntimeError(
            "CSV does not contain label columns, so end-to-end threshold tuning is impossible.\n"
            "Use a labeled VAL CSV for tuning. For TEST submission export, tune thresholds on VAL first,\n"
            "then reuse those thresholds when running on TEST."
        )
    y_true_full = df[LABELS].to_numpy(dtype=np.int32)

    # ------------------------------------------------------------
    # Get p_nd and p2 either from raw files OR from inference
    # ------------------------------------------------------------
    logits1 = None
    logits2 = None

    if RAW_PROBS is not None:
        print("[RAW] Using raw probs:", RAW_PROBS)
        p_nd, p2 = load_raw_probs_for_df(RAW_PROBS, df)
    elif RAW_LOGITS is not None:
        print("[RAW] Using raw logits:", RAW_LOGITS)
        logits1, logits2 = load_raw_logits_for_df(RAW_LOGITS, df)
        p_nd = _sigmoid_np(logits1)
        p2 = _sigmoid_np(logits2)
    else:
        # Run inference
        tf = SimpleTransform(IMG_SIZE, train=False)
        ds = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        print("Loading models...")
        m1 = load_model(STAGE1_CKPT, num_classes=1, model_name=MODEL_NAME_STAGE_1)
        m2 = load_model(STAGE2_CKPT, num_classes=len(LABELS_WO_ND), model_name=MODEL_NAME_STAGE_2)

        print("Inferring Stage-1 logits...")
        logits1 = infer_logits(m1, loader, DEVICE, use_amp=USE_AMP_EVAL).reshape(-1)
        p_nd = _sigmoid_np(logits1)

        print("Inferring Stage-2 logits...")
        logits2 = infer_logits(m2, loader, DEVICE, use_amp=USE_AMP_EVAL)
        p2 = _sigmoid_np(logits2)

        # Save raw outputs for future fast iteration
        raw_probs_csv = os.path.join(run_dir, "raw_probs_stage1_stage2.csv")
        save_raw_probs_csv(df, p_nd, p2, raw_probs_csv, STAGE2_LABELS, nd_label=ND_LABEL)
        print("Saved:", raw_probs_csv)

        raw_logits_csv = os.path.join(run_dir, "raw_logits_stage1_stage2.csv")
        save_raw_logits_csv(df, logits1, logits2, raw_logits_csv, STAGE2_LABELS, nd_label=ND_LABEL)
        print("Saved:", raw_logits_csv)

    # ------------------------------------------------------------
    # Tune thresholds once (same as your best-performing approach)
    # ------------------------------------------------------------
    print("\nTuning Stage-2 per-class thresholds (given t_nd=0.5)...")
    t2_pc = tune_stage2_per_class_thresholds(p_nd, p2, y_true_full, t_nd=0.5)

    print("Tuning Stage-1 ND threshold (end-to-end hard_gate objective)...")
    t_nd_best, _, _ = tune_tnd_for_end2end(p_nd, p2, y_true_full, t2_pc)

    print("Re-tuning Stage-2 per-class thresholds with best t_nd...")
    t2_pc = tune_stage2_per_class_thresholds(p_nd, p2, y_true_full, t_nd=t_nd_best)

    # Reference score for tuned (hard_gate)
    macro_best, micro_best = end2end_score(p_nd, p2, y_true_full, t_nd_best, t2_pc)
    t2_global_best, _, _ = tune_stage2_global_threshold(p_nd, p2, y_true_full, t_nd_best)

    print("\n================ TUNED THRESHOLDS (BASELINE) ================")
    print(f"Stage1 ckpt: {os.path.basename(STAGE1_CKPT)}")
    print(f"Stage2 ckpt: {os.path.basename(STAGE2_CKPT)}")
    print(f"Best t_nd:   {t_nd_best:.6f}")
    print(f"Baseline (hard_gate) macro_f1: {macro_best:.6f}")
    print(f"Baseline (hard_gate) micro_f1: {micro_best:.6f}")
    print(f"Best stage2 GLOBAL threshold (reference): {t2_global_best:.6f}")
    print("=============================================================\n")

    # ------------------------------------------------------------
    # Run ALL policies once and save CSVs
    # ------------------------------------------------------------
    results = []
    best_any = {"macro": -1.0}

    for cfg in POLICY_CONFIGS:
        pol_name = cfg["name"]
        policy = cfg["policy"]

        y_pred = build_preds_arbitrated(
            p_nd=p_nd,
            p2=p2,
            t_nd=t_nd_best,
            t2_pc=t2_pc,
            policy=policy,
            margin=float(cfg.get("margin", 0.05)),
            k=int(cfg.get("k", 1)),
            t_lo=cfg.get("t_lo", None),
            t_hi=cfg.get("t_hi", None),
            beta=float(cfg.get("beta", 2.0)),
        )
        macro_p, micro_p, _ = f1_macro_micro(y_true_full, y_pred)

        csv_base = os.path.join(run_dir, f"pred_{pol_name}__macro_{_float_tag(macro_p)}.csv")
        save_predictions_csv(df, y_pred, csv_base)

        rec = {
            "policy_name": pol_name,
            "policy": policy,
            "params": {k: v for k, v in cfg.items() if k not in ["name", "policy"]},
            "macro_f1": float(macro_p),
            "micro_f1": float(micro_p),
            "csv": os.path.basename(csv_base),
        }

        # Force variants per policy (optional)
        if SAVE_FORCE_VARIANTS_PER_POLICY:
            y_fnd, mask0 = force_nd_if_all_zero(y_pred, LABELS, ND_LABEL)
            macro_fnd, micro_fnd, _ = f1_macro_micro(y_true_full, y_fnd)
            csv_fnd = os.path.join(run_dir, f"pred_{pol_name}__forceND__macro_{_float_tag(macro_fnd)}.csv")
            save_predictions_csv(df, y_fnd, csv_fnd)

            y_f1o, _ = force_one_label_by_maxprob(
                y_pred_full=y_pred,
                p_nd=p_nd,
                p2=p2,
                labels_full=LABELS,
                labels_stage2=STAGE2_LABELS,
                nd_label=ND_LABEL,
            )
            macro_f1o, micro_f1o, _ = f1_macro_micro(y_true_full, y_f1o)
            csv_f1o = os.path.join(run_dir, f"pred_{pol_name}__forceONE__macro_{_float_tag(macro_f1o)}.csv")
            save_predictions_csv(df, y_f1o, csv_f1o)

            rec["all_zero_rows_in_base"] = int(mask0.sum())
            rec["force_nd"] = {
                "macro_f1": float(macro_fnd),
                "micro_f1": float(micro_fnd),
                "csv": os.path.basename(csv_fnd),
            }
            rec["force_one"] = {
                "macro_f1": float(macro_f1o),
                "micro_f1": float(micro_f1o),
                "csv": os.path.basename(csv_f1o),
            }

            # Track best among (base, forceND, forceONE)
            candidates = [
                ("base", macro_p, csv_base),
                ("force_nd", macro_fnd, csv_fnd),
                ("force_one", macro_f1o, csv_f1o),
            ]
            for tag, m, path in candidates:
                if float(m) > float(best_any["macro"]):
                    best_any = {
                        "macro": float(m),
                        "variant": tag,
                        "policy_name": pol_name,
                        "policy": policy,
                        "csv": os.path.basename(path),
                    }
        else:
            if float(macro_p) > float(best_any["macro"]):
                best_any = {
                    "macro": float(macro_p),
                    "variant": "base",
                    "policy_name": pol_name,
                    "policy": policy,
                    "csv": os.path.basename(csv_base),
                }

        results.append(rec)

        print(f"[POLICY] {pol_name:18s} base macro={macro_p:.6f} micro={micro_p:.6f} -> {os.path.basename(csv_base)}")
        if SAVE_FORCE_VARIANTS_PER_POLICY:
            print(f"         all_zero={rec.get('all_zero_rows_in_base', 0)} "
                  f"forceND macro={rec['force_nd']['macro_f1']:.6f} "
                  f"forceONE macro={rec['force_one']['macro_f1']:.6f}")

    print("\n================ BEST EXPORTED CSV ================")
    print(json.dumps(best_any, indent=2))
    print("==================================================\n")

    # ------------------------------------------------------------
    # Save thresholds + summary JSON
    # ------------------------------------------------------------
    thr_json = {
        "monitor": MONITOR,
        "csv_used_for_tuning": CSV_PATH,
        "inputs": {
            "raw_probs_used": RAW_PROBS,
            "raw_logits_used": RAW_LOGITS,
            "note": "If raw_*_used is not null, inference was skipped and values were loaded/aligned by image name.",
        },
        "stage1": {
            "checkpoint": os.path.basename(STAGE1_CKPT),
            "nd_threshold": float(t_nd_best),
            "meaning": "baseline thresholding uses hard_gate; policy exports may override behavior",
        },
        "stage2": {
            "checkpoint": os.path.basename(STAGE2_CKPT),
            "labels_order_stage2": STAGE2_LABELS,
            "global_threshold_reference": float(t2_global_best),
            "per_class_thresholds": {lab: float(t) for lab, t in zip(STAGE2_LABELS, t2_pc.tolist())},
        },
        "baseline_hard_gate_score": {
            "macro_f1": float(macro_best),
            "micro_f1": float(micro_best),
        },
        "policy_exports": results,
        "best_exported": best_any,
    }

    with open(os.path.join(run_dir, "thresholds_end2end.json"), "w") as f:
        json.dump(thr_json, f, indent=2)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "stage1_ckpt": os.path.basename(STAGE1_CKPT),
                "stage2_ckpt": os.path.basename(STAGE2_CKPT),
                "t_nd": float(t_nd_best),
                "best_exported": best_any,
                "policy_count": len(results),
            },
            f,
            indent=2,
        )

    print("Saved outputs to:", run_dir)
    print("  - stage1_selected.pt")
    print("  - stage2_selected.pt")
    print("  - thresholds_end2end.json")
    print("  - summary.json")
    if args.raw_probs is None and args.raw_logits is None:
        print("  - raw_probs_stage1_stage2.csv")
        print("  - raw_logits_stage1_stage2.csv")


if __name__ == "__main__":
    main()
