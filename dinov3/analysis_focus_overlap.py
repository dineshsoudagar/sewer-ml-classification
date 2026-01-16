import argparse
import pandas as pd
import numpy as np

def parse_labels_arg(labels_arg):
    if labels_arg is None:
        return None
    if len(labels_arg) == 1 and "," in labels_arg[0]:
        return [x.strip() for x in labels_arg[0].split(",") if x.strip()]
    return labels_arg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--image-col", default=None, help="default: first column")
    ap.add_argument("--labels", nargs="*", default=None,
                    help='space-separated labels or comma-separated: --labels "A,B,C"')
    ap.add_argument("--focus", default="ND,FS,VA",
                    help='comma-separated focus labels, default "ND,FS,VA"')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    image_col = args.image_col if args.image_col else df.columns[0]

    labels = parse_labels_arg(args.labels)
    if labels is None:
        labels = [c for c in df.columns if c != image_col]

    focus = [x.strip() for x in args.focus.split(",") if x.strip()]
    missing = [c for c in focus if c not in labels]
    if missing:
        raise ValueError(f"Focus labels not found in labels list: {missing}")

    # Coerce to numeric 0/1-ish, treat NaN as 0
    Y = df[labels].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # If your labels are already 0/1, this is fine. If not, anything >0 becomes 1.
    Yb = (Y.to_numpy(dtype=np.float32) > 0.5)
    labels_idx = {lab: i for i, lab in enumerate(labels)}

    n = Yb.shape[0]
    print("=" * 90)
    print(f"CSV: {args.csv}")
    print(f"Samples: {n}")
    print(f"Num labels: {len(labels)}")
    print(f"Focus labels: {focus}")
    print("=" * 90)

    focus_idx = [labels_idx[l] for l in focus]
    non_focus_idx = [i for i in range(len(labels)) if i not in focus_idx]

    # Helper masks
    mask_focus_any = Yb[:, focus_idx].any(axis=1)
    mask_non_focus_any = Yb[:, non_focus_idx].any(axis=1) if non_focus_idx else np.zeros(n, dtype=bool)

    # For each focus label: how many have additional labels?
    for lab in focus:
        i = labels_idx[lab]
        mask_lab = Yb[:, i]

        n_lab = int(mask_lab.sum())
        if n_lab == 0:
            print(f"\n[{lab}] present in 0 samples.")
            continue

        # "Any other label" means: any label besides this one (including other focus labels)
        other_mask = Yb.copy()
        other_mask[:, i] = False
        mask_any_other = other_mask.any(axis=1)

        # Split "other" into: other focus labels vs non-focus labels
        other_focus_cols = [labels_idx[x] for x in focus if x != lab]
        mask_other_focus = Yb[:, other_focus_cols].any(axis=1) if other_focus_cols else np.zeros(n, dtype=bool)
        mask_other_non_focus = Yb[:, non_focus_idx].any(axis=1) if non_focus_idx else np.zeros(n, dtype=bool)

        n_with_any_other = int((mask_lab & mask_any_other).sum())
        n_only_this = n_lab - n_with_any_other

        n_with_other_focus = int((mask_lab & mask_other_focus).sum())
        n_with_non_focus = int((mask_lab & mask_other_non_focus).sum())
        n_with_both = int((mask_lab & mask_other_focus & mask_other_non_focus).sum())

        # Additional label count distribution (excluding this label)
        add_counts = (other_mask.sum(axis=1)).astype(int)
        add_counts_lab = add_counts[mask_lab]
        pct = lambda x: (100.0 * x / n_lab)

        print(f"\n[{lab}]")
        print(f"  present: {n_lab} ({100.0*n_lab/n:.3f}% of all samples)")
        print(f"  ONLY {lab}: {n_only_this} ({pct(n_only_this):.3f}% of {lab} samples)")
        print(f"  {lab} + ANY other label: {n_with_any_other} ({pct(n_with_any_other):.3f}% of {lab} samples)")
        print(f"    overlaps with other focus labels ({','.join([x for x in focus if x != lab])}): "
              f"{n_with_other_focus} ({pct(n_with_other_focus):.3f}%)")
        print(f"    overlaps with non-focus labels (all other 16 labels): "
              f"{n_with_non_focus} ({pct(n_with_non_focus):.3f}%)")
        print(f"    overlaps with BOTH other-focus AND non-focus: "
              f"{n_with_both} ({pct(n_with_both):.3f}%)")

        # show distribution of extra labels
        # bucket additional labels into 0,1,2,3,4,5+
        buckets = {
            "0": int((add_counts_lab == 0).sum()),
            "1": int((add_counts_lab == 1).sum()),
            "2": int((add_counts_lab == 2).sum()),
            "3": int((add_counts_lab == 3).sum()),
            "4": int((add_counts_lab == 4).sum()),
            "5+": int((add_counts_lab >= 5).sum()),
        }
        print("  additional-label-count (excluding this label):")
        for k, v in buckets.items():
            print(f"    {k}: {v} ({pct(v):.3f}%)")

        # Top co-labels with this label (which other labels appear most often with it)
        co_counts = {}
        lab_rows = Yb[mask_lab]
        for j, other in enumerate(labels):
            if other == lab:
                continue
            co_counts[other] = int(lab_rows[:, j].sum())
        top = sorted(co_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("  top co-labels with this label (count, % of this-label samples):")
        for other, c in top:
            if c == 0:
                continue
            print(f"    {other}: {c} ({pct(c):.3f}%)")

    # Now the union: ND or FS or VA
    n_union = int(mask_focus_any.sum())
    if n_union > 0:
        # “other label” relative to the union:
        # any non-focus label present
        n_union_with_non_focus = int((mask_focus_any & mask_non_focus_any).sum())
        n_union_only_focus_labels = n_union - n_union_with_non_focus

        print("\n" + "=" * 90)
        print("[UNION] Samples with ANY of focus labels (ND/FS/VA)")
        print(f"  union size: {n_union} ({100.0*n_union/n:.3f}% of all samples)")
        print(f"  union + at least one NON-focus label: {n_union_with_non_focus} ({100.0*n_union_with_non_focus/n_union:.3f}% of union)")
        print(f"  union with ONLY focus labels (no other labels): {n_union_only_focus_labels} ({100.0*n_union_only_focus_labels/n_union:.3f}% of union)")

        # Overlap pattern among focus labels: 100,010,001,110,101,011,111
        F = Yb[:, focus_idx]  # n x 3
        # encode pattern bits
        code = (F[:, 0].astype(int) << 2) | (F[:, 1].astype(int) << 1) | (F[:, 2].astype(int) << 0)
        names = {
            0: "none",
            1: f"only {focus[2]}",
            2: f"only {focus[1]}",
            3: f"{focus[1]}+{focus[2]}",
            4: f"only {focus[0]}",
            5: f"{focus[0]}+{focus[2]}",
            6: f"{focus[0]}+{focus[1]}",
            7: f"{focus[0]}+{focus[1]}+{focus[2]}",
        }
        print("  focus-label overlap patterns:")
        for k in [4, 2, 1, 6, 5, 3, 7]:  # skip 0, order singles then pairs then triple
            cnt = int((code == k).sum())
            if cnt == 0:
                continue
            print(f"    {names[k]}: {cnt} ({100.0*cnt/n_union:.3f}% of union)")

    print("=" * 90)

if __name__ == "__main__":
    main()
