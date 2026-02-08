import os
import re
import json
import shutil
from typing import Dict, List, Tuple

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
# EDIT THESE PATHS
# =========================
CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

STAGE1_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\epoch05_f1_0.92718_acc_0.93508.pt"
STAGE2_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\epoch14_macroF1_0.73940_microF1_0.80718.pt"
STAGE3_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_low_labels_384\best.pt"

STAGE1_THRESHOLD_TXT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\best_threshold.txt"
STAGE2_THRESHOLD_JSON = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\best_thresholds_epoch14_macroF1_0.73940_microF1_0.80718.json"
# start thresholds for stage3 (baseline); will be tuned
STAGE3_THRESHOLD_JSON_INIT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_low_labels_384\stage3_thresholds_val_nd0.json"
# =========================

# Stage-1 output meaning
STAGE1_OUTPUT_MODE = "ND"  # "ND" or "DEFECT_PRESENT"

MODEL_NAME_STAGE_1 = "vit_small_plus_patch16_dinov3.lvd1689m"
MODEL_NAME_STAGE_2 = "vit_base_patch16_dinov3.lvd1689m"
MODEL_NAME_STAGE_3 = "vit_small_patch16_dinov3.lvd1689m"

# Full label order in CSV (must match your CSV columns)
LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"
]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

# Stage2 head order (your confirmed order)
STAGE2_LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA"
]

# Image sizes per stage (you can change independently)
IMG_SIZE_STAGE1 = 384
IMG_SIZE_STAGE2 = 384
IMG_SIZE_STAGE3 = 384

BATCH_SIZE_STAGE1 = 64
BATCH_SIZE_STAGE2 = 64
BATCH_SIZE_STAGE3 = 64

NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP_EVAL = True

# Tuning config
MONITOR = "macro"   # "macro" or "micro" (for choosing best thresholds)
COARSE_STEPS = 200
FINE_STEPS = 300
FINE_WINDOW = 0.05
CD_PASSES = 3
MIN_IMPROVE = 1e-7

OUT_ROOT = "e2e_tune_stage3_allzero_rescue"

# -------------------------
# Utilities
# -------------------------
def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))

def _safe_name(p: str) -> str:
    base = os.path.splitext(os.path.basename(p))[0]
    base = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", base)
    return base

def load_stage1_threshold_txt(path: str) -> float:
    with open(path, "r") as f:
        s = f.read().strip().replace(",", " ")
    t = float(s.split()[0])
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Stage1 threshold out of [0,1]: {t}")
    return t

def load_thresholds_json(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("Threshold JSON must be a dict: {label: threshold}")
    out = {str(k): float(v) for k, v in d.items()}
    for k, v in out.items():
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold out of [0,1] for {k}: {v}")
    return out

def load_model(ckpt_path: str, num_classes: int, model_name: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "labels" in ckpt:
        print(f"[CKPT] {model_name} labels:", ckpt["labels"])
    model = DinoV3MultiLabel(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model

class ImageIndexDataset(Dataset):
    """
    Uses the same df order, but can restrict to a list of indices.
    Returns (name, tensor).
    """
    def __init__(self, df: pd.DataFrame, images_dir: str, transform, indices: np.ndarray | None = None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_col = self.df.columns[0]
        self.indices = indices if indices is not None else np.arange(len(self.df), dtype=np.int64)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        row = self.df.iloc[idx]
        name = str(row[self.image_col])
        path = os.path.join(self.images_dir, name)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = self.transform(image=img)["image"]
        return name, x

@torch.no_grad()
def infer_probs(model: nn.Module, loader: DataLoader, device: str, use_amp: bool) -> Tuple[List[str], np.ndarray]:
    model.eval()
    names_all: List[str] = []
    logits_all: List[np.ndarray] = []

    if device.startswith("cuda"):
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    else:
        amp_ctx = torch.cpu.amp.autocast(enabled=False)

    for names, x in tqdm(loader, desc="Infer", leave=False):
        names_all.extend(list(names))
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
        logits_all.append(logits.float().cpu().numpy())

    logits = np.concatenate(logits_all, axis=0)
    probs = _sigmoid_np(logits)
    return names_all, probs

def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = (2.0 * tp + fp + fn)
    return float((2.0 * tp) / denom) if denom > 0 else 0.0

def f1_macro_micro(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, np.ndarray]:
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)

    tp = np.logical_and(yt, yp).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~yt, yp).sum(axis=0).astype(np.float64)
    fn = np.logical_and(yt, ~yp).sum(axis=0).astype(np.float64)

    f1 = np.zeros(tp.shape[0], dtype=np.float64)
    denom = (2 * tp + fp + fn)
    ok = denom > 0
    f1[ok] = (2 * tp[ok]) / denom[ok]

    macro = float(f1.mean())

    tp_all = float(tp.sum())
    fp_all = float(fp.sum())
    fn_all = float(fn.sum())
    denom_micro = (2 * tp_all + fp_all + fn_all)
    micro = float((2 * tp_all) / denom_micro) if denom_micro > 0 else 0.0

    return macro, micro, f1.astype(np.float32)

def build_stage1_stage2_preds(
    p_nd: np.ndarray,              # [N]
    p2: np.ndarray,                # [N,18] in STAGE2_LABELS order
    t_nd: float,
    t2_pc: np.ndarray              # [18]
) -> np.ndarray:
    N = p_nd.shape[0]
    y_pred = np.zeros((N, len(LABELS)), dtype=np.int32)
    nd_idx = LABELS.index(ND_LABEL)

    pred_nd = (p_nd >= t_nd)
    defect_mask = ~pred_nd

    # ND output
    y_pred[:, nd_idx] = pred_nd.astype(np.int32)

    # Defect outputs (18)
    full_index = {lab: LABELS.index(lab) for lab in STAGE2_LABELS}
    pred2 = defect_mask[:, None] & (p2 >= t2_pc.reshape(1, -1))  # [N,18] bool
    for j, lab in enumerate(STAGE2_LABELS):
        y_pred[:, full_index[lab]] = pred2[:, j].astype(np.int32)

    return y_pred

def apply_stage3_allzero_rescue_only_adds(
    y_base: np.ndarray,            # [N,19] int
    rescue_idx: np.ndarray,        # [R]
    p3_rescue: np.ndarray,         # [R, K]
    stage3_labels: List[str],      # K labels
    t3: np.ndarray                 # [K]
) -> np.ndarray:
    y = y_base.copy()
    # only adds; y_base at rescue is all-zero defects by construction, but keep OR semantics anyway
    for j, lab in enumerate(stage3_labels):
        col = LABELS.index(lab)
        add = (p3_rescue[:, j] >= float(t3[j])).astype(np.int32)
        y[rescue_idx, col] = np.maximum(y[rescue_idx, col], add)
    return y

def compute_rescue_idx(y_base: np.ndarray) -> np.ndarray:
    nd_idx = LABELS.index(ND_LABEL)
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]
    # stage1 defect path => ND==0 in y_base
    defect_path = (y_base[:, nd_idx] == 0)
    # stage2 predicts all 18 defects 0
    allzero = (y_base[:, other_idx].sum(axis=1) == 0)
    return np.where(defect_path & allzero)[0].astype(np.int64)

# -------------------------
# Threshold tuning (fast, uses only rescue rows)
# -------------------------
def tune_stage3_thresholds_coordinate_descent(
    y_true_full: np.ndarray,          # [N,19] int
    y_base: np.ndarray,               # [N,19] int
    rescue_idx: np.ndarray,           # [R]
    p3_rescue: np.ndarray,            # [R,K]
    stage3_labels: List[str],         # K
    t3_init: np.ndarray,              # [K]
) -> Tuple[np.ndarray, float, float]:
    N = y_true_full.shape[0]
    L = y_true_full.shape[1]
    K = len(stage3_labels)

    yt = y_true_full.astype(bool)
    yb = y_base.astype(bool)

    # Identify affected (stage3) label indices in full LABELS
    affected_full_idx = [LABELS.index(lab) for lab in stage3_labels]
    affected_set = set(affected_full_idx)
    const_idx = [i for i in range(L) if i not in affected_set]

    outside_mask = np.ones((N,), dtype=bool)
    outside_mask[rescue_idx] = False

    # Precompute constant-label contributions (tp/fp/fn and f1) - never changes
    tp_const = fp_const = fn_const = 0.0
    f1_const_sum = 0.0
    for i in const_idx:
        yti = yt[:, i]
        ypi = yb[:, i]
        tp = float(np.logical_and(yti, ypi).sum())
        fp = float(np.logical_and(~yti, ypi).sum())
        fn = float(np.logical_and(yti, ~ypi).sum())
        tp_const += tp
        fp_const += fp
        fn_const += fn
        f1_const_sum += f1_from_counts(tp, fp, fn)

    # Precompute outside counts for affected labels (fixed), and rescue y_true vectors
    outside_counts = []
    rescue_ytrue = []
    for j, full_i in enumerate(affected_full_idx):
        yti_out = yt[outside_mask, full_i]
        ypi_out = yb[outside_mask, full_i]
        tp_o = float(np.logical_and(yti_out, ypi_out).sum())
        fp_o = float(np.logical_and(~yti_out, ypi_out).sum())
        fn_o = float(np.logical_and(yti_out, ~ypi_out).sum())
        outside_counts.append((tp_o, fp_o, fn_o))

        yti_res = yt[rescue_idx, full_i]
        rescue_ytrue.append(yti_res)

    rescue_ytrue = [np.asarray(v, dtype=bool) for v in rescue_ytrue]

    # helper: compute macro/micro for current t3 without building full matrix
    def score_for_t3(t3: np.ndarray) -> Tuple[float, float, List[Tuple[float,float,float,float]]]:
        # returns (macro, micro, per_label_stats) where per_label_stats[j]=(tp,fp,fn,f1)
        tp_var = fp_var = fn_var = 0.0
        f1_var_sum = 0.0
        stats = []

        for j in range(K):
            tp_o, fp_o, fn_o = outside_counts[j]
            pred_res = (p3_rescue[:, j] >= float(t3[j]))
            yt_res = rescue_ytrue[j]

            tp_r = float(np.logical_and(yt_res, pred_res).sum())
            fp_r = float(np.logical_and(~yt_res, pred_res).sum())
            fn_r = float(np.logical_and(yt_res, ~pred_res).sum())

            tp = tp_o + tp_r
            fp = fp_o + fp_r
            fn = fn_o + fn_r
            f1 = f1_from_counts(tp, fp, fn)

            tp_var += tp
            fp_var += fp
            fn_var += fn
            f1_var_sum += f1
            stats.append((tp, fp, fn, f1))

        macro = float((f1_const_sum + f1_var_sum) / float(L))
        denom_micro = (2.0 * (tp_const + tp_var) + (fp_const + fp_var) + (fn_const + fn_var))
        micro = float((2.0 * (tp_const + tp_var)) / denom_micro) if denom_micro > 0 else 0.0
        return macro, micro, stats

    def objective(macro: float, micro: float) -> float:
        return macro if MONITOR == "macro" else micro

    # Baseline score
    best_t3 = t3_init.astype(np.float32).copy()
    best_macro, best_micro, _ = score_for_t3(best_t3)
    best_obj = objective(best_macro, best_micro)

    print(f"[Tune] baseline t3 score: macro={best_macro:.6f} micro={best_micro:.6f}")

    # Coordinate descent
    for cd_pass in range(1, CD_PASSES + 1):
        improved_any = False
        print(f"\n[Tune] Coordinate pass {cd_pass}/{CD_PASSES}")

        # Current per-label stats to speed per-label evaluation
        cur_macro, cur_micro, cur_stats = score_for_t3(best_t3)
        cur_obj = objective(cur_macro, cur_micro)

        for j, lab in enumerate(stage3_labels):
            # contributions from other affected labels fixed
            f1_other_sum = sum(s[3] for k, s in enumerate(cur_stats) if k != j)
            tp_other = sum(s[0] for k, s in enumerate(cur_stats) if k != j)
            fp_other = sum(s[1] for k, s in enumerate(cur_stats) if k != j)
            fn_other = sum(s[2] for k, s in enumerate(cur_stats) if k != j)

            tp_o, fp_o, fn_o = outside_counts[j]
            yt_res = rescue_ytrue[j]
            p_res = p3_rescue[:, j]

            def eval_candidate(t: float) -> Tuple[float,float,float]:
                pred_res = (p_res >= t)
                tp_r = float(np.logical_and(yt_res, pred_res).sum())
                fp_r = float(np.logical_and(~yt_res, pred_res).sum())
                fn_r = float(np.logical_and(yt_res, ~pred_res).sum())

                tp = tp_o + tp_r
                fp = fp_o + fp_r
                fn = fn_o + fn_r
                f1 = f1_from_counts(tp, fp, fn)

                macro = float((f1_const_sum + f1_other_sum + f1) / float(L))
                tp_total = tp_const + tp_other + tp
                fp_total = fp_const + fp_other + fp
                fn_total = fn_const + fn_other + fn
                denom_micro = (2.0 * tp_total + fp_total + fn_total)
                micro = float((2.0 * tp_total) / denom_micro) if denom_micro > 0 else 0.0
                return macro, micro, objective(macro, micro)

            # --- coarse search ---
            best_local_t = float(best_t3[j])
            best_local_obj = cur_obj
            best_local_macro = cur_macro
            best_local_micro = cur_micro

            for i in range(COARSE_STEPS + 1):
                t = float(i) / float(COARSE_STEPS)
                macro, micro, obj = eval_candidate(t)
                if obj > best_local_obj + 1e-15:
                    best_local_obj = obj
                    best_local_t = t
                    best_local_macro = macro
                    best_local_micro = micro

            # --- fine search around coarse best ---
            lo = max(0.0, best_local_t - FINE_WINDOW)
            hi = min(1.0, best_local_t + FINE_WINDOW)
            for i in range(FINE_STEPS + 1):
                t = lo + (hi - lo) * (float(i) / float(FINE_STEPS))
                macro, micro, obj = eval_candidate(float(t))
                if obj > best_local_obj + 1e-15:
                    best_local_obj = obj
                    best_local_t = float(t)
                    best_local_macro = macro
                    best_local_micro = micro

            delta = best_local_obj - cur_obj
            if delta > MIN_IMPROVE:
                print(f"[Tune] {lab}: {best_t3[j]:.6f} -> {best_local_t:.6f} | macro={best_local_macro:.6f} micro={best_local_micro:.6f} (Δobj={delta:.6e})")
                best_t3[j] = best_local_t
                improved_any = True
                # refresh current stats after updating this label
                cur_macro, cur_micro, cur_stats = score_for_t3(best_t3)
                cur_obj = objective(cur_macro, cur_micro)
            else:
                print(f"[Tune] {lab}: keep {best_t3[j]:.6f}")

        new_macro, new_micro, _ = score_for_t3(best_t3)
        new_obj = objective(new_macro, new_micro)
        print(f"[Tune] end of pass {cd_pass}: macro={new_macro:.6f} micro={new_micro:.6f}")

        if not improved_any:
            print("[Tune] no further improvement; stopping early.")
            break

        # update global best
        if new_obj > best_obj + MIN_IMPROVE:
            best_obj = new_obj
            best_macro = new_macro
            best_micro = new_micro

    final_macro, final_micro, _ = score_for_t3(best_t3)
    return best_t3, final_macro, final_micro

# -------------------------
# Main
# -------------------------
def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    run_dir = os.path.join(
        OUT_ROOT,
        f"tune__s1_{_safe_name(STAGE1_CKPT)}__s2_{_safe_name(STAGE2_CKPT)}__s3_{_safe_name(STAGE3_CKPT)}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # Copy artifacts for reproducibility
    shutil.copyfile(STAGE1_CKPT, os.path.join(run_dir, "stage1_selected.pt"))
    shutil.copyfile(STAGE2_CKPT, os.path.join(run_dir, "stage2_selected.pt"))
    shutil.copyfile(STAGE3_CKPT, os.path.join(run_dir, "stage3_selected.pt"))
    shutil.copyfile(STAGE1_THRESHOLD_TXT, os.path.join(run_dir, "stage1_threshold.txt"))
    shutil.copyfile(STAGE2_THRESHOLD_JSON, os.path.join(run_dir, "stage2_thresholds.json"))
    shutil.copyfile(STAGE3_THRESHOLD_JSON_INIT, os.path.join(run_dir, "stage3_thresholds_init.json"))

    df = pd.read_csv(CSV_PATH)
    for lab in LABELS:
        if lab not in df.columns:
            raise ValueError(f"VAL CSV missing column {lab}")
    y_true_full = df[LABELS].to_numpy(dtype=np.int32)

    # Load thresholds (fixed for stage1, stage2)
    t_stage1 = load_stage1_threshold_txt(STAGE1_THRESHOLD_TXT)
    t2_map = load_thresholds_json(STAGE2_THRESHOLD_JSON)
    t2_pc = np.array([float(t2_map[lab]) for lab in STAGE2_LABELS], dtype=np.float32)

    # Stage1 threshold in ND-space
    if STAGE1_OUTPUT_MODE == "ND":
        t_nd = float(t_stage1)
    elif STAGE1_OUTPUT_MODE == "DEFECT_PRESENT":
        t_nd = float(1.0 - t_stage1)
    else:
        raise ValueError("STAGE1_OUTPUT_MODE must be 'ND' or 'DEFECT_PRESENT'")

    print(f"[Stage1] t_nd={t_nd:.6f}")
    print("[Stage2] loaded per-class thresholds for 18 labels")

    # Load Stage-3 labels order directly from ckpt (most reliable)
    ckpt3 = torch.load(STAGE3_CKPT, map_location="cpu")
    if "labels" not in ckpt3:
        raise RuntimeError("Stage-3 ckpt does not contain 'labels'. Add labels to checkpoint or hardcode stage3 order.")
    stage3_labels = list(ckpt3["labels"])
    print(f"[Stage3] labels from ckpt: {stage3_labels}")

    # Load initial Stage-3 thresholds and align
    t3_map_init = load_thresholds_json(STAGE3_THRESHOLD_JSON_INIT)
    missing3 = [lab for lab in stage3_labels if lab not in t3_map_init]
    if missing3:
        raise ValueError(f"Stage3 init threshold JSON missing: {missing3}")
    t3_init = np.array([float(t3_map_init[lab]) for lab in stage3_labels], dtype=np.float32)
    print("[Stage3] loaded initial per-class thresholds")

    # Build transforms/dataloaders for stage1 & stage2 (full set)
    tf1 = SimpleTransform(IMG_SIZE_STAGE1, train=False)
    tf2 = SimpleTransform(IMG_SIZE_STAGE2, train=False)

    ds1 = ImageIndexDataset(df, IMAGES_DIR, transform=tf1)
    ds2 = ImageIndexDataset(df, IMAGES_DIR, transform=tf2)

    dl1 = DataLoader(ds1, batch_size=BATCH_SIZE_STAGE1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    dl2 = DataLoader(ds2, batch_size=BATCH_SIZE_STAGE2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("Loading models...")
    m1 = load_model(STAGE1_CKPT, num_classes=1, model_name=MODEL_NAME_STAGE_1)
    m2 = load_model(STAGE2_CKPT, num_classes=len(LABELS_WO_ND), model_name=MODEL_NAME_STAGE_2)

    print("Inferring Stage-1 probs...")
    names1, p1 = infer_probs(m1, dl1, DEVICE, use_amp=USE_AMP_EVAL)
    p1 = p1.reshape(-1)

    if STAGE1_OUTPUT_MODE == "ND":
        p_nd = p1
    else:
        p_nd = 1.0 - p1

    print("Inferring Stage-2 probs...")
    names2, p2 = infer_probs(m2, dl2, DEVICE, use_amp=USE_AMP_EVAL)

    if names1 != names2:
        for i, (a, b) in enumerate(zip(names1, names2)):
            if a != b:
                raise RuntimeError(f"Stage1/Stage2 order mismatch at {i}: {a} vs {b}")
        raise RuntimeError("Stage1/Stage2 order mismatch (unknown)")

    # Base predictions (Stage1+Stage2 fixed)
    y_base = build_stage1_stage2_preds(p_nd, p2, t_nd, t2_pc)
    base_macro, base_micro, _ = f1_macro_micro(y_true_full, y_base)
    print("\n========== BASE (Stage1+Stage2) ==========")
    print(f"macro_f1={base_macro:.6f} micro_f1={base_micro:.6f}")

    # Rescue subset indices
    rescue_idx = compute_rescue_idx(y_base)
    print(f"\n[Rescue] all-zero defect-path rows: {len(rescue_idx)} / {len(df)}")

    if len(rescue_idx) == 0:
        print("No rescue rows found. Nothing to tune.")
        return

    # Stage3 inference ONLY on rescue rows
    tf3 = SimpleTransform(IMG_SIZE_STAGE3, train=False)
    ds3 = ImageIndexDataset(df, IMAGES_DIR, transform=tf3, indices=rescue_idx)
    dl3 = DataLoader(ds3, batch_size=BATCH_SIZE_STAGE3, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    m3 = load_model(STAGE3_CKPT, num_classes=len(stage3_labels), model_name=MODEL_NAME_STAGE_3)

    print("Inferring Stage-3 probs on rescue rows only...")
    names3, p3_rescue = infer_probs(m3, dl3, DEVICE, use_amp=USE_AMP_EVAL)

    # Baseline fusion using init t3
    y_fused_init = apply_stage3_allzero_rescue_only_adds(
        y_base=y_base,
        rescue_idx=rescue_idx,
        p3_rescue=p3_rescue,
        stage3_labels=stage3_labels,
        t3=t3_init
    )
    init_macro, init_micro, _ = f1_macro_micro(y_true_full, y_fused_init)

    print("\n========== FUSION (init Stage-3 thresholds) ==========")
    print(f"macro_f1={init_macro:.6f} micro_f1={init_micro:.6f}")
    print("t3_init:", {lab: float(t) for lab, t in zip(stage3_labels, t3_init.tolist())})

    # Tune Stage-3 thresholds for best end-to-end macro/micro under rescue-only rule
    t3_best, tuned_macro_fast, tuned_micro_fast = tune_stage3_thresholds_coordinate_descent(
        y_true_full=y_true_full,
        y_base=y_base,
        rescue_idx=rescue_idx,
        p3_rescue=p3_rescue,
        stage3_labels=stage3_labels,
        t3_init=t3_init
    )

    # Build tuned predictions and compute exact macro/micro (sanity check)
    y_fused_best = apply_stage3_allzero_rescue_only_adds(
        y_base=y_base,
        rescue_idx=rescue_idx,
        p3_rescue=p3_rescue,
        stage3_labels=stage3_labels,
        t3=t3_best
    )
    tuned_macro, tuned_micro, f1_per = f1_macro_micro(y_true_full, y_fused_best)

    print("\n========== FUSION (tuned Stage-3 thresholds) ==========")
    print(f"macro_f1={tuned_macro:.6f} micro_f1={tuned_micro:.6f}")
    print("t3_best:", {lab: float(t) for lab, t in zip(stage3_labels, t3_best.tolist())})

    # Save tuned thresholds
    tuned_thr_path = os.path.join(run_dir, "stage3_thresholds_tuned_allzero_rescue.json")
    with open(tuned_thr_path, "w") as f:
        json.dump({lab: float(t) for lab, t in zip(stage3_labels, t3_best.tolist())}, f, indent=2)

    # Save prediction CSV for tuned fusion
    out_csv = os.path.join(run_dir, f"pred_e2e_allzero_rescue_TUNED_macro_{tuned_macro:.6f}.csv")
    out = pd.DataFrame()
    out[df.columns[0]] = df[df.columns[0]].astype(str)
    for i, lab in enumerate(LABELS):
        out[lab] = y_fused_best[:, i].astype(np.int32)
    out.to_csv(out_csv, index=False)

    # Summary
    summary = {
        "csv_used": CSV_PATH,
        "images_dir": IMAGES_DIR,
        "stage1": {"ckpt": os.path.basename(STAGE1_CKPT), "img_size": IMG_SIZE_STAGE1, "t_nd": t_nd, "output_mode": STAGE1_OUTPUT_MODE},
        "stage2": {"ckpt": os.path.basename(STAGE2_CKPT), "img_size": IMG_SIZE_STAGE2, "thresholds": {lab: float(t2_map[lab]) for lab in STAGE2_LABELS}},
        "stage3": {"ckpt": os.path.basename(STAGE3_CKPT), "img_size": IMG_SIZE_STAGE3, "labels": stage3_labels},
        "rescue_rows": int(len(rescue_idx)),
        "scores": {
            "base_stage1_stage2": {"macro_f1": float(base_macro), "micro_f1": float(base_micro)},
            "fusion_init_t3": {"macro_f1": float(init_macro), "micro_f1": float(init_micro), "t3_init": {lab: float(t) for lab, t in zip(stage3_labels, t3_init.tolist())}},
            "fusion_tuned_t3": {"macro_f1": float(tuned_macro), "micro_f1": float(tuned_micro), "t3_best": {lab: float(t) for lab, t in zip(stage3_labels, t3_best.tolist())}},
        },
        "artifacts": {
            "tuned_stage3_thresholds": os.path.basename(tuned_thr_path),
            "pred_csv_tuned": os.path.basename(out_csv),
        }
    }
    with open(os.path.join(run_dir, "summary_tune_stage3_allzero_rescue.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved to:", run_dir)
    print(" -", os.path.basename(tuned_thr_path))
    print(" -", os.path.basename(out_csv))
    print(" - summary_tune_stage3_allzero_rescue.json")

if __name__ == "__main__":
    main()
