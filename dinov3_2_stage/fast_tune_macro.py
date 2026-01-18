import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
GT_CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
PRED_CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\e2e_exports_4\e2e__s1_best__s2_best\raw_probs_stage1_stage2.csv"

ID_COL = "Filename"
PRED_PREFIX = "p_"
ND_LABEL = "ND"

PRED_VALUES_ARE_PROBS = True  # your file is raw_probs

STAGE1_ND_THRESHOLD = 0.43888241052627563
STAGE2_THRESHOLDS = {
  "RB": 0.8389675617218018,
  "OB": 0.5848305821418762,
  "PF": 0.9405169486999512,
  "DE": 0.9484056234359741,
  "FS": 0.4392157196998596,
  "IS": 0.9353083372116089,
  "RO": 0.9139333963394165,
  "IN": 0.8533190488815308,
  "AF": 0.7762061357498169,
  "BE": 0.8643192052841187,
  "FO": 0.8947532176971436,
  "GR": 0.9064573049545288,
  "PH": 0.9561114311218262,
  "PB": 0.9937023520469666,
  "OS": 0.9876817464828491,
  "OP": 0.9939779043197632,
  "OK": 0.6360662579536438,
  "VA": 0.3707858920097351
}

# Sweeps (fast)
TND_SWEEP = np.linspace(0.20, 0.60, 21)   # adjust range if you want
GAMMA_SWEEP = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

# Optional: speed-up by tuning on a subset, then evaluate on full.
# Set to 1.0 to disable subsampling.
TUNE_SUBSAMPLE_FRAC = 1.0  # e.g., 0.25 for quick testing

# =========================
# Fast metrics
# =========================
def fast_macro_micro_f1(y_true: np.ndarray, y_pred: np.ndarray):
    """
    y_true, y_pred: (N, C) binary {0,1}
    Macro-F1: mean over classes of 2TP/(2TP+FP+FN)
    Micro-F1: aggregate over all classes
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = (y_true & y_pred).sum(axis=0)
    fp = ((1 - y_true) & y_pred).sum(axis=0)
    fn = (y_true & (1 - y_pred)).sum(axis=0)

    denom = (2 * tp + fp + fn)
    per_f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    macro = float(per_f1.mean())

    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    micro_denom = (2 * TP + FP + FN)
    micro = float((2 * TP) / micro_denom) if micro_denom > 0 else 0.0
    return macro, micro, per_f1, tp, fp, fn

def enforce_exclusive_nd(pred: np.ndarray, nd_idx: int, defect_idx: np.ndarray):
    any_def = (pred[:, defect_idx].sum(axis=1) > 0).astype(np.int64)
    pred[:, nd_idx] = (1 - any_def)
    return pred

def gate_stats(gt: np.ndarray, p_nd: np.ndarray, t_nd: float, nd_idx: int, defect_idx: np.ndarray):
    gt_any_def = (gt[:, defect_idx].sum(axis=1) > 0).astype(np.int64)
    pred_nd = (p_nd >= t_nd).astype(np.int64)
    gate_fn = int(((gt_any_def == 1) & (pred_nd == 1)).sum())
    gate_fp = int(((gt_any_def == 0) & (pred_nd == 0)).sum())
    n_def = int((gt_any_def == 1).sum())
    n_nd = int((gt_any_def == 0).sum())
    return {
        "t_nd": float(t_nd),
        "gate_fn_rate": gate_fn / n_def if n_def else 0.0,
        "gate_fp_rate": gate_fp / n_nd if n_nd else 0.0,
        "gate_fn": gate_fn,
        "gate_fp": gate_fp,
        "n_def": n_def,
        "n_nd": n_nd,
    }

# =========================
# Build predictions
# =========================
def pred_hard_gate(p_nd: np.ndarray, p_def: np.ndarray, t_nd: float, t_cls: np.ndarray, nd_idx: int, defect_idx: np.ndarray):
    """
    p_def: (N, K) defect probs in same order as defect_labels
    t_cls: (K,) thresholds for defect labels
    """
    pred = np.zeros((p_def.shape[0], 1 + p_def.shape[1]), dtype=np.int64)

    gate_nd = (p_nd >= t_nd)  # True means predict ND at gate => block defects
    # defects only if not gated ND
    pred[:, 1:] = ((~gate_nd)[:, None] & (p_def >= t_cls[None, :])).astype(np.int64)

    pred = enforce_exclusive_nd(pred, nd_idx, defect_idx)
    return pred

def pred_soft_gate(p_nd: np.ndarray, p_def: np.ndarray, gamma: float, t_cls: np.ndarray, nd_idx: int, defect_idx: np.ndarray):
    """
    g = (1 - p_nd)^gamma ; p_final = g * p_def
    """
    pred = np.zeros((p_def.shape[0], 1 + p_def.shape[1]), dtype=np.int64)
    g = np.clip(1.0 - p_nd, 0.0, 1.0) ** float(gamma)
    p_final = g[:, None] * p_def
    pred[:, 1:] = (p_final >= t_cls[None, :]).astype(np.int64)
    pred = enforce_exclusive_nd(pred, nd_idx, defect_idx)
    return pred

# =========================
# Main
# =========================
def main():
    gt = pd.read_csv(GT_CSV_PATH)
    pred = pd.read_csv(PRED_CSV_PATH)

    gt[ID_COL] = gt[ID_COL].astype(str)
    pred[ID_COL] = pred[ID_COL].astype(str)

    defect_labels = list(STAGE2_THRESHOLDS.keys())
    labels = [ND_LABEL] + defect_labels

    merged = gt[[ID_COL] + labels].merge(pred, on=ID_COL, how="inner")
    if len(merged) == 0:
        raise ValueError("No matched rows after merge on Filename.")

    # Optional subsample for quick tuning
    if TUNE_SUBSAMPLE_FRAC < 1.0:
        merged = merged.sample(frac=TUNE_SUBSAMPLE_FRAC, random_state=42).reset_index(drop=True)

    # GT matrix (N, C)
    y_true = merged[labels].values.astype(np.int64)

    # Prediction arrays
    p_nd = merged[f"{PRED_PREFIX}{ND_LABEL}"].values.astype(np.float32)
    p_def = np.stack([merged[f"{PRED_PREFIX}{lab}"].values.astype(np.float32) for lab in defect_labels], axis=1)

    # Threshold arrays aligned with defect_labels order
    t_cls = np.array([STAGE2_THRESHOLDS[lab] for lab in defect_labels], dtype=np.float32)

    nd_idx = 0
    defect_idx = np.arange(1, 1 + len(defect_labels), dtype=np.int64)

    # Baseline: your hard gate
    y_pred0 = pred_hard_gate(p_nd, p_def, STAGE1_ND_THRESHOLD, t_cls, nd_idx, defect_idx)
    macro0, micro0, per_f10, tp0, fp0, fn0 = fast_macro_micro_f1(y_true, y_pred0)
    print("=== Baseline HARD gate (your thresholds) ===")
    print(f"Macro F1: {macro0:.6f} | Micro F1: {micro0:.6f}")
    print("Gate:", gate_stats(y_true, p_nd, STAGE1_ND_THRESHOLD, nd_idx, defect_idx))

    # Sweep t_nd with fixed Stage-2 thresholds
    best = {"macro": -1.0, "micro": None, "t_nd": None}
    print("\n=== Sweep t_nd (HARD gate, Stage-2 thresholds fixed) ===")
    for t_nd in TND_SWEEP:
        y_pred = pred_hard_gate(p_nd, p_def, float(t_nd), t_cls, nd_idx, defect_idx)
        macro, micro, *_ = fast_macro_micro_f1(y_true, y_pred)
        if macro > best["macro"]:
            best = {"macro": macro, "micro": micro, "t_nd": float(t_nd)}
        print(f"t_nd={t_nd:.3f} -> macro={macro:.6f} micro={micro:.6f}")
    print("\nBest HARD sweep:", best)

    # Sweep gamma with soft gate (no stage1 threshold involved)
    bestg = {"macro": -1.0, "micro": None, "gamma": None}
    print("\n=== Sweep gamma (SOFT gate, Stage-2 thresholds fixed) ===")
    for g in GAMMA_SWEEP:
        y_pred = pred_soft_gate(p_nd, p_def, float(g), t_cls, nd_idx, defect_idx)
        macro, micro, *_ = fast_macro_micro_f1(y_true, y_pred)
        if macro > bestg["macro"]:
            bestg = {"macro": macro, "micro": micro, "gamma": float(g)}
        print(f"gamma={g:<4} -> macro={macro:.6f} micro={micro:.6f}")
    print("\nBest SOFT sweep:", bestg)

if __name__ == "__main__":
    main()
