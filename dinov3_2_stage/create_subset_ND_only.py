import os
import pandas as pd

# =========================
# EDIT THESE
# =========================
IN_CSV  = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
OUT_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train_gate_sanity_5k.csv"

ND_LABEL = "ND"
N_ND = 2500        # ND == 1
N_NON_ND = 2500    # ND == 0
SEED = 42
# =========================


def main():
    df = pd.read_csv(IN_CSV)

    if ND_LABEL not in df.columns:
        raise ValueError(f"Column '{ND_LABEL}' not found in CSV. Columns: {list(df.columns)[:10]} ...")

    # Ensure numeric
    nd = pd.to_numeric(df[ND_LABEL], errors="coerce").fillna(0).astype(int)

    df_nd = df[nd == 1]
    df_non = df[nd == 0]

    if len(df_nd) < N_ND:
        raise ValueError(f"Not enough ND==1 rows: have {len(df_nd)}, need {N_ND}")
    if len(df_non) < N_NON_ND:
        raise ValueError(f"Not enough ND==0 rows: have {len(df_non)}, need {N_NON_ND}")

    samp_nd = df_nd.sample(n=N_ND, random_state=SEED)
    samp_non = df_non.sample(n=N_NON_ND, random_state=SEED)

    out = pd.concat([samp_nd, samp_non], axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    print("=" * 80)
    print("Wrote balanced gate sanity CSV")
    print("IN :", IN_CSV)
    print("OUT:", OUT_CSV)
    print(f"Total rows: {len(out)}")
    print(f"ND==1 rows: {(pd.to_numeric(out[ND_LABEL], errors='coerce').fillna(0).astype(int) == 1).sum()}")
    print(f"ND==0 rows: {(pd.to_numeric(out[ND_LABEL], errors='coerce').fillna(0).astype(int) == 0).sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
