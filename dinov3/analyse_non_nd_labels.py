#!/usr/bin/env python3
"""
analyse_non_nd_labels.py (CORRECTED)

Analyzes all labels EXCEPT a chosen "normal" label (default: ND),
conditioning on ND==0 (defect samples).

Outputs:
- Class-wise % prevalence among ND==0 samples
- Neg:pos ratios and suggested pos_weight (optionally clamped)
- Labels-per-image distribution (excluding ND) for ND==0 subset
- Top co-occurring label pairs (REAL COUNTS; fixed)
- Optional: P(label | FS=1), P(label | VA=1) if those exist

Usage:
  python analyse_non_nd_labels.py --csv D:\path\train.csv
  python analyse_non_nd_labels.py --csv train.csv --normal ND --top-k 15
  python analyse_non_nd_labels.py --csv train.csv --labels "ND,FS,VA,OB,OK,..."
"""

import argparse
import pandas as pd
import numpy as np


def parse_labels_arg(labels_arg):
    if labels_arg is None:
        return None
    if len(labels_arg) == 1 and "," in labels_arg[0]:
        return [x.strip() for x in labels_arg[0].split(",") if x.strip()]
    return labels_arg


def top_pairs_from_cooc(C: np.ndarray, labels, top_k: int = 30, min_count: int = 1):
    pairs = []
    L = len(labels)
    for i in range(L):
        for j in range(i + 1, L):
            c = int(C[i, j])
            if c >= min_count:
                pairs.append((labels[i], labels[j], c))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image-col", default=None, help="default: first column in CSV")
    ap.add_argument("--labels", nargs="*", default=None,
                    help='space-separated or comma-separated: --labels "ND,FS,VA,OB,..."')
    ap.add_argument("--normal", default="ND", help='label to treat as "normal" and exclude (default ND)')
    ap.add_argument("--top-k", type=int, default=15, help="rows to print for rare/common labels")
    ap.add_argument("--top-pairs", type=int, default=25, help="top co-occurring label pairs to print")
    ap.add_argument("--min-pair-count", type=int, default=1000,
                    help="only show co-occurring pairs with at least this many samples (default 1000)")
    ap.add_argument("--clamp-pos-weight", type=float, default=50.0,
                    help="clamp suggested pos_weight to this max (default 50)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    image_col = args.image_col if args.image_col else df.columns[0]

    labels = parse_labels_arg(args.labels)
    if labels is None:
        labels = [c for c in df.columns if c != image_col]

    if args.normal not in labels:
        raise ValueError(f"Normal label '{args.normal}' not found. Available labels: {labels}")

    # Coerce to numeric; treat NaN as 0; binarize by >0.5
    Y = df[labels].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    Yb = (Y > 0.5)

    idx = {lab: i for i, lab in enumerate(labels)}
    normal_i = idx[args.normal]

    # Defect subset: ND == 0
    mask_defect = ~Yb[:, normal_i]
    n_all = Yb.shape[0]
    n_def = int(mask_defect.sum())
    if n_def == 0:
        print(f"No samples with {args.normal}=0 found. Nothing to analyze.")
        return

    # Matrix without ND, restricted to ND==0 samples
    other_labels = [l for l in labels if l != args.normal]
    other_idx = [idx[l] for l in other_labels]
    Y_def_bool = Yb[mask_defect][:, other_idx]  # bool matrix (n_def, L-1)

    print("=" * 100)
    print(f"CSV: {args.csv}")
    print(f"Total samples: {n_all}")
    print(f"Normal label: {args.normal}")
    print(f"Defect subset (where {args.normal}=0): {n_def} ({100.0*n_def/n_all:.3f}% of all)")
    print(f"Analyzed labels (excluding {args.normal}): {len(other_labels)}")
    print("=" * 100)

    # -----------------------
    # Class-wise prevalence among defects
    # -----------------------
    pos = Y_def_bool.sum(axis=0).astype(int)
    neg = n_def - pos
    prev = pos / max(1, n_def)
    ir = (neg + 1e-9) / (pos + 1e-9)  # neg-to-pos

    stats = pd.DataFrame({
        "label": other_labels,
        "pos_count_in_defects": pos,
        "pct_of_defect_samples": prev * 100.0,
        "neg_to_pos_ratio": ir,
        "suggested_pos_weight": np.minimum(ir, args.clamp_pos_weight),
    }).sort_values("pct_of_defect_samples", ascending=True)

    print("Class-wise distribution among DEFECT samples (ND=0):")
    print("\nMost rare (among defects):")
    print(stats.head(args.top_k).to_string(index=False, formatters={"pct_of_defect_samples": "{:.3f}".format}))
    print("\nMost common (among defects):")
    print(stats.tail(args.top_k).to_string(index=False, formatters={"pct_of_defect_samples": "{:.3f}".format}))

    print("\nSummary of % prevalence (among defects):")
    print(stats["pct_of_defect_samples"].describe().to_string())
    print("-" * 100)

    # -----------------------
    # Labels-per-image distribution (excluding ND, within defects)
    # -----------------------
    k = Y_def_bool.sum(axis=1).astype(int)
    k_vals, k_counts = np.unique(k, return_counts=True)
    k_dist = pd.DataFrame({"num_labels_excluding_ND": k_vals, "count": k_counts})
    k_dist["pct"] = k_dist["count"] / n_def * 100.0

    print("Labels per image (excluding ND) within DEFECT samples:")
    print(pd.Series({
        "mean_labels_per_image_excl_ND": float(k.mean()),
        "median_labels_per_image_excl_ND": float(np.median(k)),
        "pct_exactly_1_label": float((k == 1).mean() * 100.0),
        "pct_2_or_more_labels": float((k >= 2).mean() * 100.0),
        "max_labels_in_one_image": int(k.max()),
    }).to_string())
    print("\nDistribution of labels-per-image (excluding ND):")
    print(k_dist.to_string(index=False, formatters={"pct": "{:.3f}".format}))
    print("-" * 100)

    # -----------------------
    # Co-occurrence (excluding ND, within defects) - FIXED
    # IMPORTANT: cast bool -> int BEFORE matmul so we get counts, not logical results.
    # -----------------------
    Y_def_int = Y_def_bool.astype(np.int32)
    C = (Y_def_int.T @ Y_def_int).astype(np.int64)
    np.fill_diagonal(C, 0)

    top_pairs = top_pairs_from_cooc(C, other_labels, top_k=args.top_pairs, min_count=args.min_pair_count)

    print(f"Top {args.top_pairs} co-occurring label pairs (within defects, excluding ND), min_count={args.min_pair_count}:")
    if len(top_pairs) == 0:
        print("  (none met the threshold; lower --min-pair-count)")
    else:
        for a, b, c in top_pairs:
            print(f"  {a} + {b}: {c}")
    print("-" * 100)

    # -----------------------
    # Optional: overlaps with FS/VA if present
    # -----------------------
    for pivot in ["FS", "VA"]:
        if pivot in other_labels:
            p_i = other_labels.index(pivot)
            mask_p = Y_def_bool[:, p_i]
            n_p = int(mask_p.sum())
            if n_p == 0:
                continue

            # For each label: P(label=1 | pivot=1)
            cond = (Y_def_int[mask_p].sum(axis=0) / n_p) * 100.0
            cond_df = pd.DataFrame({"label": other_labels, f"pct_given_{pivot}=1": cond}).sort_values(
                f"pct_given_{pivot}=1", ascending=False
            )

            print(f"Top labels co-occurring with {pivot} (percent of {pivot} samples):")
            print(cond_df.head(12).to_string(index=False, formatters={f"pct_given_{pivot}=1": "{:.3f}".format}))
            print("-" * 100)

    # -----------------------
    # Guidance
    # -----------------------
    min_prev = float(stats["pct_of_defect_samples"].min())
    max_ir = float(stats["neg_to_pos_ratio"].max())

    print("Practical training guidance for Stage-2 (defect-only, excluding ND):")
    if (min_prev < 1.0) or (max_ir >= 100):
        print("- Still VERY imbalanced among defects.")
        print("  Prefer ONE of:")
        print("  (A) balanced sampling / class-aware batching, OR")
        print(f"  (B) BCEWithLogitsLoss(pos_weight=neg/pos) but CLAMP weights (<= {args.clamp_pos_weight}).")
        print("  Consider Asymmetric Loss / Focal variants if rare labels are critical.")
    elif (min_prev < 5.0) or (max_ir >= 20):
        print("- Moderately imbalanced among defects.")
        print("  Use mild sampling or pos_weight (clamped). Avoid overcorrecting with both.")
    else:
        print("- Relatively balanced among defects; plain BCEWithLogitsLoss likely fine.")

    print("- AMP note: keep forward in autocast, but compute BCE loss in FP32: loss = criterion(logits.float(), y.float())")
    print("=" * 100)


if __name__ == "__main__":
    main()
