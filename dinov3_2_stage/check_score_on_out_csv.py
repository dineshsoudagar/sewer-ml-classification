import os
import numpy as np
import pandas as pd

# -------------------------
# Config
# -------------------------
GT_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"          # ground-truth csv
PRED_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_final_predictions.csv"  # your saved predictions csv

LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"
]
ND_LABEL = "ND"


# -------------------------
# Helpers (same logic as your metrics.py for F1)
# -------------------------
def macro_micro_f1_from_preds(preds: np.ndarray, targets: np.ndarray):
    preds = preds.astype(np.int32)
    targets = targets.astype(np.int32)

    tp = (preds & targets).sum(axis=0).astype(np.float64)
    fp = (preds & (1 - targets)).sum(axis=0).astype(np.float64)
    fn = ((1 - preds) & targets).sum(axis=0).astype(np.float64)

    denom = 2.0 * tp + fp + fn
    f1_per_class = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    macro = float(f1_per_class.mean())

    TP = float(tp.sum())
    FP = float(fp.sum())
    FN = float(fn.sum())
    denom_micro = 2.0 * TP + FP + FN
    micro = float((2.0 * TP) / denom_micro) if denom_micro > 0 else 0.0

    return macro, micro, f1_per_class


def binary_metrics(pred: np.ndarray, y: np.ndarray):
    pred = pred.astype(np.int32).reshape(-1)
    y = y.astype(np.int32).reshape(-1)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return acc, f1, precision, recall, (tp, fp, fn, tn)


def main():
    if not os.path.isfile(GT_CSV):
        raise FileNotFoundError(f"GT_CSV not found: {GT_CSV}")
    if not os.path.isfile(PRED_CSV):
        raise FileNotFoundError(f"PRED_CSV not found: {PRED_CSV}")

    df_gt = pd.read_csv(GT_CSV)
    df_pr = pd.read_csv(PRED_CSV)

    img_col_gt = df_gt.columns[0]
    img_col_pr = df_pr.columns[0]

    # Ensure label columns exist
    missing_gt = [c for c in LABELS if c not in df_gt.columns]
    missing_pr = [c for c in LABELS if c not in df_pr.columns]
    if missing_gt:
        raise ValueError(f"GT_CSV missing label columns: {missing_gt}")
    if missing_pr:
        raise ValueError(f"PRED_CSV missing label columns: {missing_pr}")

    # Align rows by image name (robust to ordering differences)
    df_gt = df_gt[[img_col_gt] + LABELS].copy()
    df_pr = df_pr[[img_col_pr] + LABELS].copy()
    df_gt[img_col_gt] = df_gt[img_col_gt].astype(str)
    df_pr[img_col_pr] = df_pr[img_col_pr].astype(str)

    df = df_gt.merge(df_pr, left_on=img_col_gt, right_on=img_col_pr, how="inner", suffixes=("_gt", "_pred"))
    if len(df) == 0:
        raise ValueError("No matching image names between GT_CSV and PRED_CSV.")

    # Cast to int {0,1}
    y_true_all = df[[f"{l}_gt" for l in LABELS]].to_numpy(dtype=np.int32)
    y_pred_all = df[[f"{l}_pred" for l in LABELS]].to_numpy(dtype=np.int32)

    # --- Stage1 ND (binary) ---
    y_nd = df[f"{ND_LABEL}_gt"].to_numpy(dtype=np.int32)
    p_nd = df[f"{ND_LABEL}_pred"].to_numpy(dtype=np.int32)
    nd_acc, nd_f1, nd_prec, nd_rec, nd_counts = binary_metrics(p_nd, y_nd)

    # --- End-to-end all labels ---
    macro_all, micro_all, f1_per = macro_micro_f1_from_preds(y_pred_all, y_true_all)

    # --- Defects only, on GT ND==0 subset ---
    mask_def = (df[f"{ND_LABEL}_gt"].to_numpy(dtype=np.int32) == 0)
    defects = [l for l in LABELS if l != ND_LABEL]
    if mask_def.any():
        y_true_def = df.loc[mask_def, [f"{l}_gt" for l in defects]].to_numpy(dtype=np.int32)
        y_pred_def = df.loc[mask_def, [f"{l}_pred" for l in defects]].to_numpy(dtype=np.int32)
        macro_def, micro_def, f1_per_def = macro_micro_f1_from_preds(y_pred_def, y_true_def)
    else:
        macro_def = micro_def = 0.0
        f1_per_def = np.zeros((len(defects),), dtype=np.float64)

    print("\n========== METRICS FROM SAVED CSV ==========")
    print(f"Matched rows: {len(df)}")
    print(f"[Stage1 ND] acc={nd_acc:.5f} f1={nd_f1:.5f} prec={nd_prec:.5f} rec={nd_rec:.5f} "
          f"tp,fp,fn,tn={nd_counts}")

    print(f"[Defects | GT ND==0 subset] macro_f1={macro_def:.5f} micro_f1={micro_def:.5f}")
    print(f"[End-to-end | All labels] macro_f1={macro_all:.5f} micro_f1={micro_all:.5f}")

    # Optional: per-class F1 (all labels)
    print("\nPer-class F1 (all labels):")
    for lab, f1c in zip(LABELS, f1_per.tolist()):
        print(f"  {lab}: {f1c:.5f}")
    print("===========================================\n")


if __name__ == "__main__":
    main()
