"""
analyze_expandai_two_stage.py

Assumes your schema:
- Pred CSV (probs or logits): Filename, p_RB, p_OB, ..., p_VA, p_ND
- GT CSV: Filename, WaterLevel (optional), VA,RB,OB,...,ND, Defect(optional)

ND=1 means no defect.
Final prediction must satisfy exclusivity:
- If any defect label=1 => ND=0
- If all defect labels=0 => ND=1
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

# =========================
# CONFIG (EDIT THESE)
# =========================

GT_CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"         # your validation GT CSV
PRED_CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\e2e_exports_4\e2e__s1_best__s2_best\raw_probs_stage1_stage2.csv"    # your predicted probs OR logits CSV (same columns, different values)

ID_COL = "Filename"

# If PRED_CSV has probabilities in [0,1], set True.
# If it contains raw logits, set False (script will apply sigmoid).
PRED_VALUES_ARE_PROBS = True

ND_LABEL = "ND"
PRED_PREFIX = "p_"  # prediction columns are p_<LABEL>

# Your precomputed thresholds
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

# Optional: re-tune thresholds on this validation set (post-training optimization)
DO_THRESHOLD_TUNING = False
TUNE_PASSES = 3
TUNE_QUANTILES = 60

BOTTOM_K = 10
TOP_K = 10


# =========================
# Utilities
# =========================

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def compute_f1s(y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str]) -> pd.DataFrame:
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    rows = []
    for i, name in enumerate(label_names):
        rows.append({
            "label": name,
            "support": int(sup[i]),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
        })
    per_df = pd.DataFrame(rows).sort_values("f1", ascending=True).reset_index(drop=True)
    return macro, micro, per_df

def enforce_exclusive_nd(pred_bin: np.ndarray, nd_index: int, defect_indices: list[int]) -> np.ndarray:
    # ND = 1 iff no defect predicted
    any_def = (pred_bin[:, defect_indices].sum(axis=1) > 0).astype(int)
    pred_bin[:, nd_index] = (1 - any_def).astype(int)
    return pred_bin

def gate_stats(gt_bin: np.ndarray, s1_nd_prob: np.ndarray, t_nd: float, nd_index: int, defect_indices: list[int]) -> dict:
    gt_any_def = (gt_bin[:, defect_indices].sum(axis=1) > 0).astype(int)
    pred_nd = (s1_nd_prob >= t_nd).astype(int)  # 1 means predicted ND

    gate_fn = int(((gt_any_def == 1) & (pred_nd == 1)).sum())  # defect GT but predicted ND
    gate_fp = int(((gt_any_def == 0) & (pred_nd == 0)).sum())  # ND GT but predicted defect-route

    n_def = int((gt_any_def == 1).sum())
    n_nd = int((gt_any_def == 0).sum())

    return {
        "t_nd": float(t_nd),
        "n_defect_gt": n_def,
        "n_nd_gt": n_nd,
        "gate_fn": gate_fn,
        "gate_fn_rate": float(gate_fn / n_def) if n_def > 0 else 0.0,
        "gate_fp": gate_fp,
        "gate_fp_rate": float(gate_fp / n_nd) if n_nd > 0 else 0.0,
    }

def build_predictions_hard_gate(
    s1_nd_prob: np.ndarray,
    s2_probs: dict[str, np.ndarray],
    labels: list[str],
    nd_label: str,
    t_nd: float,
    t_cls: dict[str, float],
) -> np.ndarray:
    n = len(s1_nd_prob)
    pred = np.zeros((n, len(labels)), dtype=int)

    nd_idx = labels.index(nd_label)
    defect_labels = [l for l in labels if l != nd_label]
    defect_indices = [labels.index(l) for l in defect_labels]

    pred_nd_gate = (s1_nd_prob >= t_nd).astype(int)  # 1 = predict ND at gate

    # If gate says ND, defects forced 0; otherwise threshold Stage-2 probs
    for lab in defect_labels:
        p = s2_probs[lab]
        pred[:, labels.index(lab)] = ((pred_nd_gate == 0) & (p >= t_cls[lab])).astype(int)

    # Enforce ND semantics last
    pred = enforce_exclusive_nd(pred, nd_idx, defect_indices)
    return pred

def coordinate_descent_thresholds(
    gt_bin: np.ndarray,
    s1_nd_prob: np.ndarray,
    s2_probs: dict[str, np.ndarray],
    labels: list[str],
    nd_label: str,
    init_t_nd: float,
    init_t_cls: dict[str, float],
    passes: int,
    candidate_quantiles: int,
):
    y_true = gt_bin.astype(int)

    defect_labels = [l for l in labels if l != nd_label]

    def cand_from_probs(p: np.ndarray) -> np.ndarray:
        qs = np.linspace(0.0, 1.0, candidate_quantiles)
        vals = np.unique(np.clip(np.quantile(p, qs), 1e-6, 1 - 1e-6))
        if len(vals) < 10:
            vals = np.linspace(0.05, 0.95, 19)
        return vals

    cand_nd = cand_from_probs(s1_nd_prob)
    cand_cls = {lab: cand_from_probs(s2_probs[lab]) for lab in defect_labels}

    t_nd = float(init_t_nd)
    t_cls = {lab: float(init_t_cls[lab]) for lab in defect_labels}

    best_macro = -1.0

    for _ in range(passes):
        # optimize t_nd
        best_local = (-1.0, t_nd)
        for t in cand_nd:
            y_pred = build_predictions_hard_gate(s1_nd_prob, s2_probs, labels, nd_label, float(t), t_cls)
            macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
            if macro > best_local[0]:
                best_local = (macro, float(t))
        t_nd = best_local[1]

        # optimize each class threshold
        for lab in defect_labels:
            best_local = (-1.0, t_cls[lab])
            for t in cand_cls[lab]:
                t_try = dict(t_cls)
                t_try[lab] = float(t)
                y_pred = build_predictions_hard_gate(s1_nd_prob, s2_probs, labels, nd_label, t_nd, t_try)
                macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
                if macro > best_local[0]:
                    best_local = (macro, float(t))
            t_cls[lab] = best_local[1]

        y_pred = build_predictions_hard_gate(s1_nd_prob, s2_probs, labels, nd_label, t_nd, t_cls)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        best_macro = max(best_macro, macro)

    return t_nd, t_cls, float(best_macro)


# =========================
# Main
# =========================

def main():
    # Load GT
    gt = pd.read_csv(GT_CSV_PATH)
    if ID_COL not in gt.columns:
        raise ValueError(f"GT missing ID_COL='{ID_COL}'. Columns: {list(gt.columns)[:50]}")

    gt[ID_COL] = gt[ID_COL].astype(str)

    # Determine label columns from GT: intersection with thresholds + ND
    # (Ignore WaterLevel and Defect and any other non-label columns.)
    defect_labels = list(STAGE2_THRESHOLDS.keys())
    labels = [ND_LABEL] + defect_labels  # keep ND first

    missing_gt = [c for c in labels if c not in gt.columns]
    if missing_gt:
        raise ValueError(
            f"GT missing label columns: {missing_gt}. "
            f"GT columns (first 50): {list(gt.columns)[:50]}"
        )

    gt_lab = gt[[ID_COL] + labels].copy()
    for c in labels:
        gt_lab[c] = gt_lab[c].astype(int)

    # Load predictions
    pred = pd.read_csv(PRED_CSV_PATH)
    if ID_COL not in pred.columns:
        raise ValueError(f"Pred CSV missing ID_COL='{ID_COL}'. Columns: {list(pred.columns)[:50]}")

    pred[ID_COL] = pred[ID_COL].astype(str)

    # Merge on Filename
    merged = gt_lab.merge(pred, on=ID_COL, how="inner", suffixes=("_gt", "_pred"))
    if len(merged) == 0:
        raise ValueError("No matched rows after merging GT and pred on Filename.")

    # Extract GT
    gt_bin = merged[labels].values.astype(int)

    # Extract predicted ND + defect probabilities/logits
    nd_pred_col = f"{PRED_PREFIX}{ND_LABEL}"
    if nd_pred_col not in merged.columns:
        raise ValueError(f"Pred CSV missing '{nd_pred_col}'")

    s1_raw = merged[nd_pred_col].values.astype(float)
    s1_nd_prob = s1_raw if PRED_VALUES_ARE_PROBS else sigmoid(s1_raw)

    s2_probs = {}
    for lab in defect_labels:
        col = f"{PRED_PREFIX}{lab}"
        if col not in merged.columns:
            raise ValueError(f"Pred CSV missing '{col}'")
        raw = merged[col].values.astype(float)
        s2_probs[lab] = raw if PRED_VALUES_ARE_PROBS else sigmoid(raw)

    # Evaluate using your thresholds
    t_nd = float(STAGE1_ND_THRESHOLD)
    t_cls = {lab: float(STAGE2_THRESHOLDS[lab]) for lab in defect_labels}

    y_pred = build_predictions_hard_gate(s1_nd_prob, s2_probs, labels, ND_LABEL, t_nd, t_cls)

    macro, micro, per_df = compute_f1s(gt_bin, y_pred, labels)

    print("=== Evaluation using provided thresholds (hard gate + exclusive ND) ===")
    print(f"Macro F1: {macro:.6f}")
    print(f"Micro F1: {micro:.6f}")

    # Gate stats
    nd_idx = labels.index(ND_LABEL)
    defect_indices = [labels.index(l) for l in defect_labels]
    gs = gate_stats(gt_bin, s1_nd_prob, t_nd, nd_idx, defect_indices)
    print("\n=== Stage-1 gate stats at your t_nd ===")
    for k, v in gs.items():
        print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # Per-class
    print(f"\nBottom {BOTTOM_K} classes by F1:")
    print(per_df.head(BOTTOM_K).to_string(index=False))

    print(f"\nTop {TOP_K} classes by F1:")
    print(per_df.sort_values("f1", ascending=False).head(TOP_K).to_string(index=False))

    # Optional tuning
    if DO_THRESHOLD_TUNING:
        print("\n=== Threshold tuning (coordinate descent, macro-F1) ===")
        t_nd2, t_cls2, best_macro = coordinate_descent_thresholds(
            gt_bin=gt_bin,
            s1_nd_prob=s1_nd_prob,
            s2_probs=s2_probs,
            labels=labels,
            nd_label=ND_LABEL,
            init_t_nd=t_nd,
            init_t_cls=t_cls,
            passes=TUNE_PASSES,
            candidate_quantiles=TUNE_QUANTILES,
        )
        print(f"Best tuned macro F1: {best_macro:.6f}")
        print(f"Tuned t_nd: {t_nd2:.6f}")

        y_pred2 = build_predictions_hard_gate(s1_nd_prob, s2_probs, labels, ND_LABEL, t_nd2, t_cls2)
        macro2, micro2, per_df2 = compute_f1s(gt_bin, y_pred2, labels)

        print("\n=== Metrics after tuning ===")
        print(f"Macro F1: {macro2:.6f}")
        print(f"Micro F1: {micro2:.6f}")

        gs2 = gate_stats(gt_bin, s1_nd_prob, t_nd2, nd_idx, defect_indices)
        print("\n=== Stage-1 gate stats at tuned t_nd ===")
        for k, v in gs2.items():
            print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

        print(f"\nBottom {BOTTOM_K} classes by F1 (tuned):")
        print(per_df2.head(BOTTOM_K).to_string(index=False))

if __name__ == "__main__":
    main()
