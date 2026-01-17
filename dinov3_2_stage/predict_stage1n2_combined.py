import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from model import DinoV3MultiLabel
from train_utils import SimpleTransform, set_seed

# Use your metrics functions
from metrics import binary_metrics_from_logits, f1_from_thresholds

# ============================================================
# Config (edit these)
# ============================================================
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"
OUT_CSV = r"outputs_final_predictions.csv"

# --------------------
# Stage 1 (ND gate)
# --------------------
STAGE1_CKPT = r"outputs_stage1_vit_small_plus\best.pt"
STAGE1_THRESHOLD_FILE = r"\dinov3_2_stage\outputs_stage1_vit_small_plus\best_threshold.txt"  # e.g. r"outputs_stage1_vit_small_plus\best_threshold.txt" or None
STAGE1_MODEL_NAME_FALLBACK = "vit_small_plus_patch16_dinov3.lvd1689m"
STAGE1_IMG_SIZE_FALLBACK = 256
STAGE1_BATCH_SIZE = 64

# --------------------
# Stage 2 (defects)
# --------------------
STAGE2_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base\new best at epoch 8 with thresholds\best__with_thresholds.pt"
STAGE2_THRESHOLDS_FILE = None  # e.g. r"outputs_stage2_vit_base_tesk_5k\best_thresholds.json" or None
STAGE2_MODEL_NAME_FALLBACK = "vit_base_patch16_dinov3.lvd1689m"
STAGE2_THRESHOLD_MODE = "per_class"
STAGE2_IMG_SIZE_FALLBACK = 256
STAGE2_BATCH_SIZE = 32

# --------------------
# Runtime
# --------------------
NUM_WORKERS = 8
USE_AMP = False
SEED = 42

LABELS = ["VA", "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
          "FO", "GR", "PH", "PB", "OS", "OP", "OK", "ND" ]
ND_LABEL = "ND"

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


# ============================================================
# Dataset for inference (keeps ordering and returns image name)
# ============================================================
class SewerMLInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, images_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_col = self.df.columns[0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = str(row[self.image_col])
        img_path = os.path.join(self.images_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img_name, img


# ============================================================
# Threshold loaders (file overrides ckpt)
# ============================================================
def load_stage1_threshold(ckpt: dict) -> float:
    if STAGE1_THRESHOLD_FILE is not None:
        if not os.path.isfile(STAGE1_THRESHOLD_FILE):
            raise FileNotFoundError(f"Stage1 threshold file not found: {STAGE1_THRESHOLD_FILE}")
        with open(STAGE1_THRESHOLD_FILE, "r") as f:
            return float(f.read().strip())

    if "threshold" in ckpt:
        return float(ckpt["threshold"])

    print("[WARN][Stage1] No threshold file and ckpt missing 'threshold'. Using 0.5.")
    return 0.5


def load_stage2_thresholds(ckpt: dict, labels_wo_nd: list[str]) -> np.ndarray:
    """
    Stage2 thresholds loading priority:
      1) If STAGE2_THRESHOLDS_FILE provided -> parse it (supports your report format)
      2) Else -> use ckpt["thresholds"]
      3) Else -> fallback to 0.5
    Returns: thresholds aligned to labels_wo_nd (np.float32 [C])
    """
    # -----------------------------
    # 1) From file (if provided)
    # -----------------------------
    if STAGE2_THRESHOLDS_FILE is not None:
        if not os.path.isfile(STAGE2_THRESHOLDS_FILE):
            raise FileNotFoundError(f"Stage2 thresholds file not found: {STAGE2_THRESHOLDS_FILE}")

        with open(STAGE2_THRESHOLDS_FILE, "r") as f:
            obj = json.load(f)

        # Case A: your "report" format with global/per_class sections
        if isinstance(obj, dict) and ("per_class" in obj or "global" in obj):
            mode = STAGE2_THRESHOLD_MODE
            if mode not in obj:
                # fallback to whichever exists
                mode = "per_class" if "per_class" in obj else "global"

            section = obj[mode]
            if not isinstance(section, dict) or "thresholds" not in section:
                raise ValueError(f"Invalid thresholds file format: missing '{mode}.thresholds'")

            th_dict = section["thresholds"]
            if not isinstance(th_dict, dict):
                raise ValueError(f"Invalid thresholds file format: '{mode}.thresholds' must be a dict")

            th = np.array([float(th_dict[l]) for l in labels_wo_nd], dtype=np.float32)
            if th.shape[0] != len(labels_wo_nd):
                raise ValueError(f"Stage2 thresholds size mismatch from file: {th.shape[0]} vs {len(labels_wo_nd)}")
            return th

        # Case B: flat dict mapping label -> threshold
        if isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values()):
            th = np.array([float(obj[l]) for l in labels_wo_nd], dtype=np.float32)
            if th.shape[0] != len(labels_wo_nd):
                raise ValueError(f"Stage2 thresholds size mismatch from file: {th.shape[0]} vs {len(labels_wo_nd)}")
            return th

        # Case C: dict has a top-level "thresholds" dict
        if isinstance(obj, dict) and "thresholds" in obj and isinstance(obj["thresholds"], dict):
            th_dict = obj["thresholds"]
            th = np.array([float(th_dict[l]) for l in labels_wo_nd], dtype=np.float32)
            if th.shape[0] != len(labels_wo_nd):
                raise ValueError(f"Stage2 thresholds size mismatch from file: {th.shape[0]} vs {len(labels_wo_nd)}")
            return th

        # Case D: list/array thresholds in correct order
        if isinstance(obj, list):
            th = np.asarray(obj, dtype=np.float32).reshape(-1)
            if th.shape[0] != len(labels_wo_nd):
                raise ValueError(f"Stage2 thresholds size mismatch from file: {th.shape[0]} vs {len(labels_wo_nd)}")
            return th

        raise ValueError("Unrecognized Stage2 thresholds file format.")

    # -----------------------------
    # 2) From checkpoint dict
    # -----------------------------
    if "thresholds" in ckpt and ckpt["thresholds"] is not None:
        th = np.asarray(ckpt["thresholds"], dtype=np.float32).reshape(-1)
        if th.shape[0] != len(labels_wo_nd):
            raise ValueError(f"Stage2 thresholds size mismatch in ckpt: {th.shape[0]} vs {len(labels_wo_nd)}")
        return th

    # -----------------------------
    # 3) Fallback
    # -----------------------------
    print("[WARN][Stage2] No thresholds file and ckpt missing 'thresholds'. Using 0.5 for all classes.")
    return np.full((len(labels_wo_nd),), 0.5, dtype=np.float32)


# ============================================================
# Prediction helpers (also return logits for scoring)
# ============================================================
@torch.no_grad()
def predict_stage1_logits(df_in: pd.DataFrame, device: str):
    ckpt = torch.load(STAGE1_CKPT, map_location=device)
    model_name = ckpt.get("model_name", STAGE1_MODEL_NAME_FALLBACK)
    img_size = int(ckpt.get("img_size", STAGE1_IMG_SIZE_FALLBACK))
    thr = load_stage1_threshold(ckpt)

    model = DinoV3MultiLabel(model_name, num_classes=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    tfm = SimpleTransform(img_size, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    ds = SewerMLInferenceDataset(df_in, VAL_IMAGES, transform=tfm)
    loader = DataLoader(ds, batch_size=STAGE1_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    all_names = []
    all_logits = []

    for names, x in tqdm(loader, desc="[Stage1] ND logits"):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(x).float().cpu().numpy()  # [B,1]
        all_names.extend([str(n) for n in names])
        all_logits.append(logits)

    all_logits = np.concatenate(all_logits, axis=0)  # [N,1]
    # predictions
    probs = 1.0 / (1.0 + np.exp(-all_logits.reshape(-1)))
    preds = (probs >= float(thr)).astype(np.int32)

    name_to_pred = {n: int(p) for n, p in zip(all_names, preds)}
    print(f"[Stage1] threshold={thr:.6f} | ckpt={STAGE1_CKPT}")
    return all_names, all_logits, name_to_pred, thr


@torch.no_grad()
def predict_stage2_logits(df_subset: pd.DataFrame, device: str):
    ckpt = torch.load(STAGE2_CKPT, map_location=device)
    model_name = ckpt.get("model_name", STAGE2_MODEL_NAME_FALLBACK)
    img_size = int(ckpt.get("img_size", STAGE2_IMG_SIZE_FALLBACK))

    # labels order for stage2 head
    if "labels" in ckpt and ckpt["labels"] is not None:
        labels_wo_nd = list(ckpt["labels"])
    else:
        labels_wo_nd = [l for l in LABELS if l != ND_LABEL]

    num_classes = len(labels_wo_nd)
    thresholds = load_stage2_thresholds(ckpt, labels_wo_nd)

    model = DinoV3MultiLabel(model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    tfm = SimpleTransform(img_size, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    ds = SewerMLInferenceDataset(df_subset, VAL_IMAGES, transform=tfm)
    loader = DataLoader(ds, batch_size=STAGE2_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    all_names = []
    all_logits = []

    for names, x in tqdm(loader, desc="[Stage2] defect logits"):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(x).float().cpu().numpy()  # [B,C]
        all_names.extend([str(n) for n in names])
        all_logits.append(logits)

    if len(all_logits) == 0:
        return labels_wo_nd, np.zeros((0, num_classes), dtype=np.float32), [], thresholds

    all_logits = np.concatenate(all_logits, axis=0)  # [N,C]
    probs = 1.0 / (1.0 + np.exp(-all_logits))
    preds = (probs >= thresholds[None, :]).astype(np.int32)

    name_to_vec = {n: preds[i] for i, n in enumerate(all_names)}
    print(f"[Stage2] ckpt={STAGE2_CKPT}")
    print(f"[Stage2] labels={labels_wo_nd}")
    return labels_wo_nd, all_logits, all_names, thresholds, name_to_vec


# ============================================================
# Metrics printing using your functions
# ============================================================
def maybe_print_metrics(df_gt: pd.DataFrame, df_pred: pd.DataFrame,
                        stage1_names: list[str], stage1_logits: np.ndarray, stage1_thr: float,
                        labels_wo_nd: list[str], stage2_names: list[str], stage2_logits: np.ndarray,
                        stage2_thresholds: np.ndarray):
    # Need GT columns
    if not all(l in df_gt.columns for l in LABELS):
        print("[METRICS] GT label columns not found in VAL_CSV. Skipping metrics.")
        return

    image_col = df_gt.columns[0]
    gt_map = df_gt.set_index(image_col)

    # ---------------- Stage 1 metrics (ND) ----------------
    y_nd = gt_map.loc[stage1_names, ND_LABEL].to_numpy(dtype=np.float32).reshape(-1, 1)
    m1 = binary_metrics_from_logits(stage1_logits, y_nd, threshold=stage1_thr)
    print("\n========== METRICS ==========")
    print(f"[Stage1 ND] bce={m1['bce']:.5f} acc={m1['acc']:.5f} f1={m1['f1']:.5f} "
          f"prec={m1['precision']:.5f} rec={m1['recall']:.5f} thr={m1['threshold']:.6f}")

    # ---------------- Stage 2 metrics (defects), evaluate on GT ND==0 only ----------------
    if len(stage2_names) > 0:
        gt_def = gt_map.loc[stage2_names, labels_wo_nd].to_numpy(dtype=np.float32)
        macro2, micro2 = f1_from_thresholds(stage2_logits, gt_def, stage2_thresholds)
        print(f"[Stage2 defects | GT ND==0 subset] macro_f1={macro2:.5f} micro_f1={micro2:.5f}")
    else:
        print("[Stage2 defects] No samples were routed to Stage2 (all predicted ND=1).")

    # ---------------- End-to-end metrics on ALL labels (including ND) ----------------
    # Use df_pred (final) and df_gt directly. Compute macro/micro with your internal function path:
    # we can compute from preds/targets by converting to logits-like input not needed; easiest:
    # Build "fake logits" from preds by mapping {0,1} -> {-inf,+inf} is messy.
    # Instead, reuse the same logic as f1_from_thresholds expects logits. We'll compute macro/micro directly here
    # using your metrics.py private logic is not accessible; so we compute macro/micro ourselves in numpy counts (same as your code).
    y_true_all = df_gt[LABELS].to_numpy(dtype=np.int32)
    y_pred_all = df_pred[LABELS].to_numpy(dtype=np.int32)

    tp = (y_pred_all & y_true_all).sum(axis=0).astype(np.float64)
    fp = (y_pred_all & (1 - y_true_all)).sum(axis=0).astype(np.float64)
    fn = ((1 - y_pred_all) & y_true_all).sum(axis=0).astype(np.float64)

    denom = 2.0 * tp + fp + fn
    f1_per = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    macro = float(f1_per.mean())

    TP = float(tp.sum())
    FP = float(fp.sum())
    FN = float(fn.sum())
    denom_micro = 2.0 * TP + FP + FN
    micro = float((2.0 * TP) / denom_micro) if denom_micro > 0 else 0.0

    print(f"[End-to-end all labels] macro_f1={macro:.5f} micro_f1={micro:.5f}")
    print("=============================\n")


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(VAL_CSV):
        raise FileNotFoundError(f"VAL_CSV not found: {VAL_CSV}")
    if not os.path.isdir(VAL_IMAGES):
        raise FileNotFoundError(f"VAL_IMAGES not found: {VAL_IMAGES}")
    if not os.path.isfile(STAGE1_CKPT):
        raise FileNotFoundError(f"STAGE1_CKPT not found: {STAGE1_CKPT}")
    if not os.path.isfile(STAGE2_CKPT):
        raise FileNotFoundError(f"STAGE2_CKPT not found: {STAGE2_CKPT}")

    df_in = pd.read_csv(VAL_CSV)
    image_col = df_in.columns[0]

    # ---------- Stage 1 ----------
    s1_names, s1_logits, name_to_nd, s1_thr = predict_stage1_logits(df_in, device)

    # Route to Stage 2 where predicted ND==0
    df_def = df_in[df_in[image_col].astype(str).map(lambda n: name_to_nd.get(str(n), 0) == 0)].copy()

    # ---------- Stage 2 ----------
    labels_wo_nd, s2_logits, s2_names, s2_thr, name_to_vec = predict_stage2_logits(df_def, device)

    # ---------- Build final output (same format as input) ----------
    df_out = df_in.copy()
    cols = list(df_out.columns)

    # If label columns exist, overwrite them; otherwise create them (still saves predictions).
    for lab in LABELS:
        if lab not in df_out.columns:
            df_out[lab] = 0

    # Set all labels to 0
    for lab in LABELS:
        df_out[lab] = 0

    # Fill ND
    df_out[ND_LABEL] = df_out[image_col].astype(str).map(lambda n: name_to_nd.get(str(n), 0)).astype(int)

    # Fill defects only if ND==0
    label_to_j = {lab: j for j, lab in enumerate(labels_wo_nd)}

    def _get_def_pred(name: str, lab: str) -> int:
        if name_to_nd.get(name, 0) == 1:
            return 0
        vec = name_to_vec.get(name, None)
        if vec is None:
            return 0
        return int(vec[label_to_j[lab]])

    for lab in LABELS:
        if lab == ND_LABEL:
            continue
        if lab not in label_to_j:
            df_out[lab] = 0
        else:
            df_out[lab] = df_out[image_col].astype(str).map(lambda n: _get_def_pred(str(n), lab)).astype(int)

    # Reorder columns to match input exactly if input already had labels
    # (If input did not have labels, keep original + appended labels)
    if all(l in cols for l in LABELS):
        df_out = df_out[cols]  # exact same order as input

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"[FINAL] Saved: {OUT_CSV}")

    # ---------- Metrics (only if GT exists) ----------
    maybe_print_metrics(
        df_gt=df_in,
        df_pred=df_out,
        stage1_names=s1_names,
        stage1_logits=s1_logits,
        stage1_thr=s1_thr,
        labels_wo_nd=labels_wo_nd,
        stage2_names=s2_names,
        stage2_logits=s2_logits,
        stage2_thresholds=s2_thr,
    )


if __name__ == "__main__":
    main()
