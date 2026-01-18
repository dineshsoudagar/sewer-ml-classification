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

# Your current thresholds
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

# Which defect labels to tune (keep everything else fixed)
TUNE_LABELS = ["FO", "RB", "IS", "DE", "IN"]  # add "AF" if you want

# Sweep t_nd (hard gate). You already found ~0.56 best in coarse sweep.
TND_SWEEP = np.linspace(0.48, 0.62, 15)

# Candidate thresholds for tuned classes: derive from quantiles of predicted probs
CAND_QUANTILES = 80     # more = better but slower
COORD_PASSES = 4        # coordinate descent passes

# =========================
# Fast metrics
# =========================
def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    tp = (y_true & y_pred).sum(axis=0)
    fp = ((1 - y_true) & y_pred).sum(axis=0)
    fn = (y_true & (1 - y_pred)).sum(axis=0)

    denom = 2 * tp + fp + fn
    per_f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return float(per_f1.mean())

def fast_micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    TP = (y_true & y_pred).sum()
    FP = ((1 - y_true) & y_pred).sum()
    FN = (y_true & (1 - y_pred)).sum()
    denom = 2 * TP + FP + FN
    return float((2 * TP) / denom) if denom > 0 else 0.0

# =========================
# Prediction
# =========================
def enforce_exclusive_nd(pred: np.ndarray) -> np.ndarray:
    any_def = (pred[:, 1:].sum(axis=1) > 0).astype(np.int64)
    pred[:, 0] = (1 - any_def)
    return pred

def pred_hard_gate(p_nd: np.ndarray, p_def: np.ndarray, t_nd: float, t_cls: np.ndarray) -> np.ndarray:
    n, k = p_def.shape
    pred = np.zeros((n, 1 + k), dtype=np.int64)

    gate_nd = (p_nd >= t_nd)
    pred[:, 1:] = ((~gate_nd)[:, None] & (p_def >= t_cls[None, :])).astype(np.int64)

    pred = enforce_exclusive_nd(pred)
    return pred

def cand_from_probs(p: np.ndarray, q: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, q)
    vals = np.unique(np.clip(np.quantile(p, qs), 1e-6, 1 - 1e-6))
    if len(vals) < 10:
        vals = np.linspace(0.05, 0.95, 19)
    return vals

# =========================
# Main
# =========================
def main():
    gt = pd.read_csv(GT_CSV_PATH)
    pr = pd.read_csv(PRED_CSV_PATH)

    gt[ID_COL] = gt[ID_COL].astype(str)
    pr[ID_COL] = pr[ID_COL].astype(str)

    defect_labels = list(STAGE2_THRESHOLDS.keys())
    labels = [ND_LABEL] + defect_labels

    merged = gt[[ID_COL] + labels].merge(pr, on=ID_COL, how="inner")
    if len(merged) == 0:
        raise ValueError("No matched rows after merge on Filename.")

    y_true = merged[labels].values.astype(np.int64)
    p_nd = merged[f"{PRED_PREFIX}{ND_LABEL}"].values.astype(np.float32)
    p_def = np.stack([merged[f"{PRED_PREFIX}{lab}"].values.astype(np.float32) for lab in defect_labels], axis=1)

    # thresholds in defect_labels order
    t_base = np.array([STAGE2_THRESHOLDS[lab] for lab in defect_labels], dtype=np.float32)

    # baseline score (your current hard gate)
    y0 = pred_hard_gate(p_nd, p_def, STAGE1_ND_THRESHOLD, t_base)
    base_macro = fast_macro_f1(y_true, y0)
    base_micro = fast_micro_f1(y_true, y0)
    print(f"Baseline (t_nd={STAGE1_ND_THRESHOLD:.6f}) macro={base_macro:.6f} micro={base_micro:.6f}")

    # prepare tuning indices
    tune_set = set(TUNE_LABELS)
    tune_idx = [i for i, lab in enumerate(defect_labels) if lab in tune_set]
    if len(tune_idx) == 0:
        raise ValueError("None of TUNE_LABELS are present in defect_labels.")

    # candidate thresholds for each tuned class based on its predicted probabilities
    cand = {}
    for i in tune_idx:
        lab = defect_labels[i]
        cand[i] = cand_from_probs(p_def[:, i], CAND_QUANTILES)

    best = {"macro": -1.0, "micro": None, "t_nd": None, "t_cls": None}

    for t_nd in TND_SWEEP:
        # start from base thresholds
        t_cls = t_base.copy()

        # coordinate descent only on selected labels
        for _ in range(COORD_PASSES):
            for i in tune_idx:
                best_local = (-1.0, float(t_cls[i]))
                for thr in cand[i]:
                    t_try = t_cls.copy()
                    t_try[i] = float(thr)
                    y_pred = pred_hard_gate(p_nd, p_def, float(t_nd), t_try)
                    macro = fast_macro_f1(y_true, y_pred)
                    if macro > best_local[0]:
                        best_local = (macro, float(thr))
                t_cls[i] = best_local[1]

        y_pred = pred_hard_gate(p_nd, p_def, float(t_nd), t_cls)
        macro = fast_macro_f1(y_true, y_pred)
        micro = fast_micro_f1(y_true, y_pred)

        print(f"t_nd={t_nd:.3f} -> macro={macro:.6f} micro={micro:.6f}")

        if macro > best["macro"]:
            best = {"macro": macro, "micro": micro, "t_nd": float(t_nd), "t_cls": t_cls.copy()}

    print("\n=== BEST RESULT (subset tuning) ===")
    print(f"macro={best['macro']:.6f} micro={best['micro']:.6f} t_nd={best['t_nd']:.6f}")

    # print tuned thresholds only for tuned labels
    print("\nTuned thresholds (only tuned labels shown):")
    for lab in TUNE_LABELS:
        if lab in defect_labels:
            i = defect_labels.index(lab)
            print(f"{lab}: {best['t_cls'][i]:.6f}  (was {t_base[i]:.6f})")

if __name__ == "__main__":
    main()
