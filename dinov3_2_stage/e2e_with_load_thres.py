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

CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

STAGE1_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\epoch05_f1_0.92718_acc_0.93508.pt"
STAGE2_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\epoch12_macroF1_0.73492_microF1_0.80464_fn_tnd_on_384.pt"

# ---- Use provided thresholds instead of tuning/searching ----
STAGE1_THRESHOLD_TXT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\best_threshold.txt"
STAGE2_THRESHOLD_JSON = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\best_thresholds_epoch12_macroF1_0.73492_microF1_0.80464_fn_tnd_on_384.json"

# IMPORTANT:
# If your updated Stage-1 model outputs DEFECT_PRESENT (1=defect, 0=ND), set to "DEFECT_PRESENT".
# If your Stage-1 model outputs ND (1=ND, 0=defect), keep as "ND".
STAGE1_OUTPUT_MODE = "ND"  # "ND" or "DEFECT_PRESENT"

MODEL_NAME_STAGE_1 = "vit_small_plus_patch16_dinov3.lvd1689m"
MODEL_NAME_STAGE_2 = "vit_base_patch16_dinov3.lvd1689m"

# IMPORTANT: must match CSV columns order for y_true_full extraction
LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"
]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

# Stage2 ckpt head order (you confirmed this)
STAGE2_LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA"
]

IMG_SIZE = 384
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

OUT_ROOT = "e2e_exports_5_stage1_ep5_n_2_ep12_fn_tnd_on_384"

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

def load_stage1_threshold_txt(path: str) -> float:
    with open(path, "r") as f:
        s = f.read().strip()
    # allow commas/newlines/spaces
    s = s.replace(",", " ").strip()
    t = float(s.split()[0])
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Stage1 threshold out of [0,1]: {t}")
    return t


def load_stage2_thresholds_json(path: str, stage2_labels: list[str]) -> np.ndarray:
    with open(path, "r") as f:
        d = json.load(f)

    if not isinstance(d, dict):
        raise ValueError("Stage2 threshold JSON must be a dict: {label: threshold}")

    missing = [lab for lab in stage2_labels if lab not in d]
    extra = [lab for lab in d.keys() if lab not in stage2_labels]

    if missing:
        raise ValueError(f"Stage2 threshold JSON missing labels: {missing}")
    if extra:
        print(f"[Warn] Stage2 threshold JSON has extra keys not used: {extra}")

    t = np.array([float(d[lab]) for lab in stage2_labels], dtype=np.float32)
    if np.any(t < 0.0) or np.any(t > 1.0):
        bad = [(lab, float(tt)) for lab, tt in zip(stage2_labels, t.tolist()) if not (0.0 <= tt <= 1.0)]
        raise ValueError(f"Stage2 thresholds out of [0,1]: {bad}")
    return t

def force_one_label_by_maxprob(
    y_pred_full: np.ndarray,
    p_nd: np.ndarray,          # [N]
    p2: np.ndarray,            # [N,18]
    labels_full: list[str],
    labels_stage2: list[str],  # stage2 head order (len=18)
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

    # set ND for choose_nd
    y[mask_idx[choose_nd], nd_idx] = 1

    # set defect argmax for remaining
    rest_idx = mask_idx[~choose_nd]
    if rest_idx.size > 0:
        j = p2[rest_idx].argmax(axis=1)  # index in stage2 head space
        full_index = {lab: labels_full.index(lab) for lab in labels_stage2}
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


def build_end2end_preds(p_nd: np.ndarray, p2: np.ndarray, t_nd: float, t2_per_class: np.ndarray) -> np.ndarray:
    nd_idx = LABELS.index(ND_LABEL)
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]

    pred_nd = (p_nd >= t_nd)
    defect_mask = ~pred_nd

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


def tune_stage2_per_class_thresholds(p_nd: np.ndarray, p2: np.ndarray, y_true_full: np.ndarray, t_nd: float) -> np.ndarray:
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


def tune_tnd_for_end2end(p_nd: np.ndarray, p2: np.ndarray, y_true_full: np.ndarray, t2_per_class: np.ndarray) -> tuple[float, float, float]:
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


def tune_stage2_global_threshold(p_nd: np.ndarray, p2: np.ndarray, y_true_full: np.ndarray, t_nd: float) -> tuple[float, float, float]:
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

    tf = SimpleTransform(IMG_SIZE, train=False)
    ds = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("Loading models...")
    m1 = load_model(STAGE1_CKPT, num_classes=1, model_name=MODEL_NAME_STAGE_1)
    m2 = load_model(STAGE2_CKPT, num_classes=len(LABELS_WO_ND), model_name=MODEL_NAME_STAGE_2)

    print("Inferring Stage-1 logits...")
    logits1 = infer_logits(m1, loader, DEVICE, use_amp=USE_AMP_EVAL).reshape(-1)
    p1 = _sigmoid_np(logits1)

    if STAGE1_OUTPUT_MODE == "ND":
        p_nd = p1
    else:
        # stage1 predicts DEFECT_PRESENT
        p_nd = 1.0 - p1

    print("Inferring Stage-2 logits...")
    logits2 = infer_logits(m2, loader, DEVICE, use_amp=USE_AMP_EVAL)
    p2 = _sigmoid_np(logits2)

    # ---- Use provided thresholds (NO tuning/search) ----
    t_stage1 = load_stage1_threshold_txt(STAGE1_THRESHOLD_TXT)
    t2_pc = load_stage2_thresholds_json(STAGE2_THRESHOLD_JSON, STAGE2_LABELS)

    # Stage-1 probability interpretation
    # If Stage-1 outputs ND (1=ND): p_nd = sigmoid(logits1), and threshold is directly t_nd
    # If Stage-1 outputs DEFECT_PRESENT (1=defect): p_defect = sigmoid(logits1),
    # then p_nd = 1 - p_defect, and an equivalent ND-threshold is t_nd = 1 - t_defect
    if STAGE1_OUTPUT_MODE == "ND":
        t_nd_best = float(t_stage1)
        print(f"[Stage1] Using provided threshold for ND: t_nd={t_nd_best:.6f}")
    elif STAGE1_OUTPUT_MODE == "DEFECT_PRESENT":
        # Convert to ND-space so build_end2end_preds() can remain unchanged
        t_nd_best = float(1.0 - t_stage1)
        print(
            f"[Stage1] Using provided threshold for DEFECT_PRESENT: t_defect={t_stage1:.6f} -> derived t_nd={t_nd_best:.6f}")
    else:
        raise ValueError(f"Unknown STAGE1_OUTPUT_MODE: {STAGE1_OUTPUT_MODE}")

    # Evaluate end-to-end score using the provided thresholds (if labels exist)
    macro_best, micro_best = end2end_score(p_nd, p2, y_true_full, t_nd_best, t2_pc)
    t2_global_best = float("nan")  # not computed (no search)

    print("\n================ RESULT ================")
    print(f"Stage1 ckpt: {os.path.basename(STAGE1_CKPT)}")
    print(f"Stage2 ckpt: {os.path.basename(STAGE2_CKPT)}")
    print(f"Best t_nd:   {t_nd_best:.6f}")
    print(f"End2end macro_f1: {macro_best:.6f}")
    print(f"End2end micro_f1: {micro_best:.6f}")
    print(f"Best stage2 GLOBAL threshold (reference): {t2_global_best:.6f}")

    # ---- Build base predictions ----
    y_pred_normal = build_end2end_preds(p_nd, p2, t_nd_best, t2_pc)
    macro_n, micro_n, _ = f1_macro_micro(y_true_full, y_pred_normal)

    # Variant 1: Force ND if all-zero
    y_pred_force_nd, mask_zero = force_nd_if_all_zero(y_pred_normal, LABELS, ND_LABEL)
    macro_fnd, micro_fnd, _ = f1_macro_micro(y_true_full, y_pred_force_nd)

    # Variant 2: Force exactly one label using ND-vs-maxdefect rule
    y_pred_force_one, mask_zero2 = force_one_label_by_maxprob(
        y_pred_full=y_pred_normal,
        p_nd=p_nd,
        p2=p2,
        labels_full=LABELS,
        labels_stage2=STAGE2_LABELS,
        nd_label=ND_LABEL,
    )
    macro_f1o, micro_f1o, _ = f1_macro_micro(y_true_full, y_pred_force_one)

    n_zero = int(mask_zero.sum())
    print("\n========== POST-PROCESS VARIANTS ==========")
    print(f"Rows with ALL-ZERO base predictions: {n_zero}")
    print(f"[Normal]      macro={macro_n:.6f} micro={micro_n:.6f}")
    print(f"[Force ND]    macro={macro_fnd:.6f} micro={micro_fnd:.6f}")
    print(f"[Force One]   macro={macro_f1o:.6f} micro={micro_f1o:.6f}")
    print("==========================================\n")

    # ---- Save CSVs ----
    csv_normal = os.path.join(run_dir, f"pred_end2end__macro_{_float_tag(macro_n)}.csv")
    csv_force_nd = os.path.join(run_dir, f"pred_end2end_forceND__macro_{_float_tag(macro_fnd)}.csv")
    csv_force_one = os.path.join(run_dir, f"pred_end2end_forceONE__macro_{_float_tag(macro_f1o)}.csv")

    save_predictions_csv(df, y_pred_normal, csv_normal)
    save_predictions_csv(df, y_pred_force_nd, csv_force_nd)
    save_predictions_csv(df, y_pred_force_one, csv_force_one)

    # ---- Save thresholds + summary ----
    thr_json = {
        "monitor": MONITOR,
        "csv_used_for_tuning": CSV_PATH,
        "stage1": {
            "checkpoint": os.path.basename(STAGE1_CKPT),
            "nd_threshold": float(t_nd_best),
            "meaning": (
                "Stage1 outputs ND probability; predict ND=1 if p(ND)>=nd_threshold else run stage2"
                if STAGE1_OUTPUT_MODE == "ND"
                else "Stage1 outputs DEFECT_PRESENT probability; internally converted to p(ND)=1-p(defect); predict ND=1 if p(ND)>=nd_threshold else run stage2"
            ),
        },
        "stage2": {
            "checkpoint": os.path.basename(STAGE2_CKPT),
            "labels_order_stage2": STAGE2_LABELS,
            "global_threshold_reference": float(t2_global_best),
            "per_class_thresholds": {lab: float(t) for lab, t in zip(STAGE2_LABELS, t2_pc.tolist())},
        },
        "end_to_end_scores_on_csv": {
            "macro_f1": float(macro_best),
            "micro_f1": float(micro_best),
        },
        "variants": {
            "all_zero_rows_in_base": n_zero,
            "normal": {"macro_f1": float(macro_n), "micro_f1": float(micro_n), "csv": os.path.basename(csv_normal)},
            "force_nd": {"macro_f1": float(macro_fnd), "micro_f1": float(micro_fnd), "csv": os.path.basename(csv_force_nd)},
            "force_one": {"macro_f1": float(macro_f1o), "micro_f1": float(micro_f1o), "csv": os.path.basename(csv_force_one)},
        },
    }

    with open(os.path.join(run_dir, "thresholds_end2end.json"), "w") as f:
        json.dump(thr_json, f, indent=2)

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "stage1_ckpt": os.path.basename(STAGE1_CKPT),
                "stage2_ckpt": os.path.basename(STAGE2_CKPT),
                "t_nd": float(t_nd_best),
                "all_zero_rows_in_base": n_zero,
                "macro_normal": float(macro_n),
                "macro_force_nd": float(macro_fnd),
                "macro_force_one": float(macro_f1o),
                "csv_normal": os.path.basename(csv_normal),
                "csv_force_nd": os.path.basename(csv_force_nd),
                "csv_force_one": os.path.basename(csv_force_one),
            },
            f,
            indent=2,
        )

    print("Saved outputs to:", run_dir)
    print("  - stage1_selected.pt")
    print("  - stage2_selected.pt")
    print("  - thresholds_end2end.json")
    print("  -", os.path.basename(csv_normal), "<-- normal")
    print("  -", os.path.basename(csv_force_nd), "<-- force ND if all-zero")
    print("  -", os.path.basename(csv_force_one), "<-- force one label (ND vs max defect)")
    print("  - summary.json")


if __name__ == "__main__":
    main()
