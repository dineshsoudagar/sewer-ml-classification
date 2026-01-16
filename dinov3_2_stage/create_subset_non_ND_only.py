import os
import pandas as pd

# =========================
# EDIT THESE
# =========================
IN_CSV  = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
OUT_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train_stage2_sanity_5k.csv"

ND_LABEL = "ND"
N_SAMPLES = 5000     # number of ND==0 (defect) rows to sample
SEED = 42
# =========================


def main():
    df = pd.read_csv(IN_CSV)

    if ND_LABEL not in df.columns:
        raise ValueError(f"Column '{ND_LABEL}' not found in CSV. Columns: {list(df.columns)[:10]} ...")

    nd = pd.to_numeric(df[ND_LABEL], errors="coerce").fillna(0).astype(int)

    df_def = df[nd == 0]
    if len(df_def) < N_SAMPLES:
        raise ValueError(f"Not enough ND==0 rows: have {len(df_def)}, need {N_SAMPLES}")

    out = df_def.sample(n=N_SAMPLES, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("=" * 80)
    print("Wrote Stage-2 sanity CSV (defects only, ND==0)")
    print("IN :", IN_CSV)
    print("OUT:", OUT_CSV)
    print(f"Total rows: {len(out)}")
    print(f"ND==0 rows: {(pd.to_numeric(out[ND_LABEL], errors='coerce').fillna(0).astype(int) == 0).sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
