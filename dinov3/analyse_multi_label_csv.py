#!/usr/bin/env python3
"""
analyze_multilabel_csv.py

Single-file CSV analyzer for multi-label datasets (like SewerML).
It computes the key statistics you need to understand imbalance and then prints
a recommended training strategy based on what it finds.

Usage:
  python analyze_multilabel_csv.py --csv /path/to/train.csv
  python analyze_multilabel_csv.py --csv /path/to/train.csv --labels A B C
  python analyze_multilabel_csv.py --csv /path/to/train.csv --labels "A,B,C"
"""

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _infer_labels(df: pd.DataFrame, image_col: str) -> List[str]:
    """
    Heuristic: all columns except image_col that are numeric-ish are labels.
    """
    cand = [c for c in df.columns if c != image_col]
    labels = []
    for c in cand:
        # try coerce to numeric; if many NaNs, likely not a label
        s = pd.to_numeric(df[c], errors="coerce")
        nan_rate = s.isna().mean()
        # keep if mostly numeric
        if nan_rate < 0.2:
            labels.append(c)
    return labels


def _coerce_binary_matrix(df: pd.DataFrame, labels: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns Y float matrix and a dict of unique values per label (to detect non-binary).
    """
    uniques = {}
    Y = np.zeros((len(df), len(labels)), dtype=np.float32)
    for j, lab in enumerate(labels):
        s = pd.to_numeric(df[lab], errors="coerce")
        uniques[lab] = np.sort(pd.Series(s.dropna().unique()).to_numpy())
        # Keep NaNs as 0 for analysis, but report them separately
        s = s.fillna(0.0)
        Y[:, j] = s.to_numpy(dtype=np.float32)
    return Y, uniques


def _top_pairs_from_cooc(C: np.ndarray, labels: List[str], top_k: int = 20) -> List[Tuple[str, str, int]]:
    """
    C: co-occurrence counts (LxL)
    """
    pairs = []
    L = len(labels)
    for i in range(L):
        for j in range(i + 1, L):
            pairs.append((labels[i], labels[j], int(C[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# -----------------------------
# Diagnosis and Recommendation
# -----------------------------
@dataclass
class Diagnosis:
    n_samples: int
    n_labels: int
    min_prevalence: float
    median_prevalence: float
    max_prevalence: float
    max_ir: float
    median_ir: float
    pct_all_zero: float
    mean_labels_per_image: float
    median_labels_per_image: float
    non_binary_labels: List[str]


def diagnose(stats: pd.DataFrame, k_per_sample: np.ndarray, non_binary_labels: List[str]) -> Diagnosis:
    return Diagnosis(
        n_samples=int(stats["n_samples"].iloc[0]),
        n_labels=int(len(stats)),
        min_prevalence=float(stats["prevalence"].min()),
        median_prevalence=float(stats["prevalence"].median()),
        max_prevalence=float(stats["prevalence"].max()),
        max_ir=float(stats["imbalance_ratio_neg_to_pos"].replace([np.inf, -np.inf], np.nan).max()),
        median_ir=float(stats["imbalance_ratio_neg_to_pos"].replace([np.inf, -np.inf], np.nan).median()),
        pct_all_zero=float((k_per_sample == 0).mean()),
        mean_labels_per_image=float(k_per_sample.mean()),
        median_labels_per_image=float(np.median(k_per_sample)),
        non_binary_labels=non_binary_labels,
    )


def recommend(d: Diagnosis) -> str:
    """
    Practical decision rules for multi-label imbalance.
    """
    lines = []
    lines.append("Recommended approach (rule-based):")

    if d.non_binary_labels:
        lines.append(
            f"- Data issue: {len(d.non_binary_labels)} label columns look non-binary. "
            "Fix/clean them to strict 0/1 before training (or explicitly map values)."
        )

    # Severity buckets
    very_imbalanced = (d.min_prevalence < 0.01) or (d.max_ir >= 100)
    moderately_imbalanced = (d.min_prevalence < 0.05) or (d.max_ir >= 20)

    if very_imbalanced:
        lines.append("- Dataset is VERY imbalanced (rare labels / very high neg:pos ratios).")
        lines.append("  Do one of the following (do NOT overcorrect with multiple at once):")
        lines.append("  1) Use a balanced sampling strategy (preferred):")
        lines.append("     - WeightedRandomSampler or per-class balanced batch construction, OR")
        lines.append("     - your SewerMLBalancedDataset approach.")
        lines.append("     - In this case, keep BCEWithLogitsLoss WITHOUT large pos_weight, or clamp pos_weight (e.g. max 20–50).")
        lines.append("  2) If you do NOT use sampling, then use BCEWithLogitsLoss(pos_weight=neg/pos), optionally clamped.")
        lines.append("  Consider stronger losses if rare labels matter most:")
        lines.append("  - Asymmetric Loss (ASL) or Focal-type losses for multi-label (helps rare positives).")
    elif moderately_imbalanced:
        lines.append("- Dataset is moderately imbalanced.")
        lines.append("  Good default: BCEWithLogitsLoss with pos_weight (clamped), OR mild sampling—pick one.")
    else:
        lines.append("- Dataset is relatively balanced per-class.")
        lines.append("  Plain BCEWithLogitsLoss is usually sufficient.")

    # All-zero rate guidance
    if d.pct_all_zero > 0.4:
        lines.append(
            f"- Many images have no positive labels (all-zero rate {d.pct_all_zero:.1%}). "
            "This is common but slows learning of positives; sampling/pos_weight becomes more important."
        )

    # Mixed precision guidance (based on your earlier issue)
    lines.append("- Mixed precision (AMP) stability recommendation:")
    lines.append("  - Keep model forward under autocast, but compute BCE loss in FP32:")
    lines.append("      loss = criterion(logits.float(), y.float())")
    lines.append("  - Prefer BF16 autocast if available (much more stable than FP16).")
    lines.append("  - If using FP16, enable GradScaler and consider grad clipping after scaler.unscale_().")

    # Thresholding
    lines.append("- Evaluation: tune thresholds on a held-out set (global or per-class) after training;")
    lines.append("  do not assume 0.5 is optimal for imbalanced multi-label problems.")

    return "\n".join(lines)


# -----------------------------
# Main analysis
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV")
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help='Label columns. Provide as space-separated: --labels A B C  OR comma-separated: --labels "A,B,C"',
    )
    ap.add_argument("--image-col", default=None, help="Image/name column. Default: first column in CSV.")
    ap.add_argument("--top-k", type=int, default=15, help="How many rare/common classes to print.")
    ap.add_argument("--top-pairs", type=int, default=20, help="How many co-occurrence pairs to print.")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if df.shape[0] == 0:
        print("CSV is empty.")
        sys.exit(1)

    image_col = args.image_col if args.image_col is not None else df.columns[0]

    # Parse labels
    labels: Optional[List[str]] = None
    if args.labels:
        if len(args.labels) == 1 and "," in args.labels[0]:
            labels = [x.strip() for x in args.labels[0].split(",") if x.strip()]
        else:
            labels = args.labels
    else:
        labels = _infer_labels(df, image_col=image_col)

    if not labels:
        print("Could not infer label columns. Please provide --labels explicitly.")
        sys.exit(1)

    missing = [c for c in labels if c not in df.columns]
    if missing:
        print(f"Missing label columns in CSV: {missing}")
        sys.exit(1)

    # Basic integrity
    print("=" * 80)
    print(f"CSV: {args.csv}")
    print(f"Samples: {len(df)}")
    print(f"Image column: {image_col}")
    print(f"Num label columns: {len(labels)}")
    print("=" * 80)

    # NaNs per label
    nan_rates = df[labels].isna().mean().sort_values(ascending=False)
    if nan_rates.max() > 0:
        print("NaN rates (top 10):")
        print(nan_rates.head(10).to_string())
        print("-" * 80)

    # Build Y and detect non-binary
    Y, uniques = _coerce_binary_matrix(df, labels)

    non_binary = []
    for lab in labels:
        u = uniques[lab]
        # allow {0,1} or {0} or {1}; flag others
        # (also flags if values are like 2, -1, 0.5, etc.)
        allowed = set([0.0, 1.0])
        if len(u) == 0:
            continue
        if not set(np.round(u, 6)).issubset(allowed):
            non_binary.append(lab)

    # Per-class statistics
    n = Y.shape[0]
    pos = Y.sum(axis=0)
    neg = n - pos
    prevalence = pos / max(1, n)
    ir = (neg + 1e-9) / (pos + 1e-9)  # neg-to-pos
    pos_weight = ir.copy()

    stats = pd.DataFrame({
        "label": labels,
        "n_samples": n,
        "pos": pos.astype(int),
        "neg": neg.astype(int),
        "prevalence": prevalence,
        "imbalance_ratio_neg_to_pos": ir,
        "suggested_pos_weight": pos_weight,
    }).sort_values("prevalence", ascending=True)

    # Cardinality / density
    k_per_sample = Y.sum(axis=1)
    label_cardinality = float(k_per_sample.mean())
    label_density = float(label_cardinality / len(labels)) if len(labels) > 0 else 0.0

    # Distribution of labels per sample
    k_vals, k_counts = np.unique(k_per_sample.astype(int), return_counts=True)
    k_dist = pd.DataFrame({"k_labels_in_image": k_vals, "count": k_counts})
    k_dist["pct"] = k_dist["count"] / n

    # Duplicates on image column
    dup_rate = float(df[image_col].duplicated().mean())

    # Co-occurrence
    Y_bool = (Y > 0.5)
    C = (Y_bool.T @ Y_bool).astype(int)  # LxL
    np.fill_diagonal(C, 0)
    top_pairs = _top_pairs_from_cooc(C, labels, top_k=args.top_pairs)

    # Print key stats
    print("Per-class prevalence (most rare):")
    print(stats.head(args.top_k).to_string(index=False))
    print("-" * 80)
    print("Per-class prevalence (most common):")
    print(stats.tail(args.top_k).to_string(index=False))
    print("-" * 80)

    print("Prevalence summary:")
    print(stats["prevalence"].describe().to_string())
    print("-" * 80)

    print("Imbalance ratio (neg:pos) summary:")
    print(pd.Series(ir).describe().to_string())
    print("-" * 80)

    print("Labels per image summary:")
    print(pd.Series({
        "mean_labels_per_image": label_cardinality,
        "median_labels_per_image": float(np.median(k_per_sample)),
        "label_density": label_density,
        "pct_all_zero": float((k_per_sample == 0).mean()),
        "pct_exactly_one": float((k_per_sample == 1).mean()),
        "pct_two_or_more": float((k_per_sample >= 2).mean()),
        "max_labels_in_one_image": int(k_per_sample.max()),
    }).to_string())
    print("-" * 80)

    print("Distribution of #labels per image (k):")
    print(k_dist.to_string(index=False))
    print("-" * 80)

    print(f"Duplicate image name rate in '{image_col}': {dup_rate:.2%}")
    if dup_rate > 0:
        print("  Note: duplicates may be valid (augmented views) or may indicate data leakage risk across splits.")
    print("-" * 80)

    if non_binary:
        print(f"Non-binary label columns detected ({len(non_binary)}):")
        # print just first 20 to keep output manageable
        show = non_binary[:20]
        print(show if len(non_binary) <= 20 else show + ["..."])
        print("Example unique values for first few flagged labels:")
        for lab in show[:5]:
            print(f"  {lab}: {uniques[lab]}")
        print("-" * 80)

    print("Top co-occurring label pairs (count):")
    for a, b, c in top_pairs:
        if c <= 0:
            break
        print(f"  {a} + {b}: {c}")
    print("-" * 80)

    # Optional: show “most redundant” pair by Jaccard (can reveal labels that almost always appear together)
    # Compute only for top co-occurring pairs to keep it cheap
    if top_pairs:
        best_j = (None, None, -1.0)
        for a, b, c in top_pairs:
            if c <= 0:
                continue
            ia = labels.index(a)
            ib = labels.index(b)
            j = _jaccard(Y_bool[:, ia], Y_bool[:, ib])
            if j > best_j[2]:
                best_j = (a, b, j)
        if best_j[0] is not None:
            print(f"Highest Jaccard among top pairs: {best_j[0]} vs {best_j[1]} -> {best_j[2]:.3f}")
            print("-" * 80)

    # Diagnose and recommend
    d = diagnose(stats, k_per_sample, non_binary)
    print("=" * 80)
    print("DIAGNOSIS (high-level):")
    print(pd.Series({
        "n_samples": d.n_samples,
        "n_labels": d.n_labels,
        "min_prevalence": d.min_prevalence,
        "median_prevalence": d.median_prevalence,
        "max_prevalence": d.max_prevalence,
        "max_imbalance_ratio_neg_to_pos": d.max_ir,
        "median_imbalance_ratio_neg_to_pos": d.median_ir,
        "pct_all_zero": d.pct_all_zero,
        "mean_labels_per_image": d.mean_labels_per_image,
        "median_labels_per_image": d.median_labels_per_image,
        "non_binary_label_columns": len(d.non_binary_labels),
    }).to_string())
    print("=" * 80)
    print(recommend(d))
    print("=" * 80)

    # Provide a simple “starter config suggestion”
    print("Starter training defaults (practical):")
    if (d.min_prevalence < 0.01) or (d.max_ir >= 100):
        print("- Use balanced sampling OR pos_weight, not both. If you sample, set pos_weight=None or clamp<=20–50.")
        print("- Consider ASL/Focal loss if rare labels dominate your objective.")
    elif (d.min_prevalence < 0.05) or (d.max_ir >= 20):
        print("- Try pos_weight clamped to <=20–50 OR light sampling. Avoid aggressive overcorrection.")
    else:
        print("- Plain BCEWithLogitsLoss, no pos_weight, standard shuffling is likely fine.")

    print("- For AMP: autocast forward, but compute BCE loss in FP32 (logits.float()). Prefer BF16 if supported.")
    print("- Tune thresholds per-class after training for best macro-F1.")
    print("=" * 80)


if __name__ == "__main__":
    main()
