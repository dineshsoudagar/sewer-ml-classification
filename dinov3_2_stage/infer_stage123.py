import os
import re
import json
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import DinoV3MultiLabel
from train_utils import SimpleTransform

# ============================================================
# EDIT THESE PATHS / OPTIONS
# ============================================================

CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

STAGE1_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\epoch05_f1_0.92718_acc_0.93508.pt"
STAGE2_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\epoch12_macroF1_0.73492_microF1_0.80464_fn_tnd_on_384.pt"
STAGE3_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_low_labels_384\best.pt"

STAGE1_THRESHOLD_TXT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\best_threshold.txt"
STAGE2_THRESHOLD_JSON = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\best_thresholds_epoch12_macroF1_0.73492_microF1_0.80464_fn_tnd_on_384.json"
STAGE3_THRESHOLD_JSON = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_low_labels_384\best_thresholds.json"

# Stage-1 output interpretation:
# - "ND": stage1 sigmoid output is p(ND)
# - "DEFECT_PRESENT": stage1 sigmoid output is p(defect); we convert p(ND)=1-p(defect)
STAGE1_OUTPUT_MODE = "ND"  # "ND" or "DEFECT_PRESENT"

MODEL_NAME_STAGE_1 = "vit_small_plus_patch16_dinov3.lvd1689m"
MODEL_NAME_STAGE_2 = "vit_base_patch16_dinov3.lvd1689m"
MODEL_NAME_STAGE_3 = "vit_small_patch16_dinov3.lvd1689m"

# Full label order (CSV + submission order)
LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"
]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

# Stage-2 head order (must match your stage2 ckpt head)
STAGE2_LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA"
]

# Stage-3 head order (must match your stage3 ckpt head + best_thresholds.json keys)
# Set to your specialist/tail labels.
STAGE3_LABELS = ["FO", "RB", "IS", "DE", "IN"]  # edit as needed

# Image size + batch per stage
IMG_SIZE_STAGE1 = 384
IMG_SIZE_STAGE2 = 384
IMG_SIZE_STAGE3 = 384

BATCH_SIZE_STAGE1 = 32
BATCH_SIZE_STAGE2 = 64
BATCH_SIZE_STAGE3 = 64

NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP_EVAL = True

OUT_ROOT = "e2e_exports_stage1_stage2_stage3"

# ------------------------------
# Stage-3 fusion options
# ------------------------------
# 1) If Stage-1 predicts ND, but Stage-3 predicts ANY tail label >= t3, un-gate ND and emit those tail labels.
APPLY_STAGE3_UNGATE_ND = True

# 2) Per-class override rule for tail labels:
#    if p3(c) >= t3(c) and p2(c) < t2(c) then set final(c)=1 (even if stage2 missed)
APPLY_STAGE3_OVERRIDE = True

# 3) If Stage-2 (after Stage-1 defect path) produces ALL-ZERO defects, apply Stage-3 and use those tail preds.
APPLY_STAGE3_IF_STAGE2_ALL_ZERO = True

# Optional: if you want Stage-3 to only ever add labels (never remove), keep True.
STAGE3_ONLY_ADDS = True

# ============================================================


def _float_tag(x: float, ndigits: int = 6) -> str:
    return f"{x:.{ndigits}f}".replace(".", "p")


def _safe_name(p: str) -> str:
    base = os.path.splitext(os.path.basename(p))[0]
    base = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", base)
    return base


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def load_stage1_threshold_txt(path: str) -> float:
    with open(path, "r") as f:
        s = f.read().strip()
    s = s.replace(",", " ").strip()
    t = float(s.split()[0])
    if not (0.0 <= t <= 1.0):
        raise ValueError(f"Stage1 threshold out of [0,1]: {t}")
    return t


def load_thresholds_json(path: str, labels_order: list[str]) -> np.ndarray:
    with open(path, "r") as f:
        d = json.load(f)

    if not isinstance(d, dict):
        raise ValueError("Threshold JSON must be a dict: {label: threshold}")

    missing = [lab for lab in labels_order if lab not in d]
    extra = [lab for lab in d.keys() if lab not in labels_order]

    if missing:
        raise ValueError(f"Threshold JSON missing labels: {missing}")
    if extra:
        print(f"[Warn] Threshold JSON has extra keys not used: {extra}")

    t = np.array([float(d[lab]) for lab in labels_order], dtype=np.float32)
    if np.any(t < 0.0) or np.any(t > 1.0):
        bad = [(lab, float(tt)) for lab, tt in zip(labels_order, t.tolist()) if not (0.0 <= tt <= 1.0)]
        raise ValueError(f"Thresholds out of [0,1]: {bad}")
    return t


class SewerMLFullDataset(Dataset):
    """
    Returns img_name, image_tensor, y (if labels exist) else zeros.
    Keeps row order identical to CSV.
    """
    def __init__(self, csv_path: str, images_dir: str, labels: list[str], transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.image_col = self.df.columns[0]
        self.has_labels = all(lab in self.df.columns for lab in labels)

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

        if self.has_labels:
            y = row[self.labels].to_numpy(dtype=np.float32)
        else:
            y = np.zeros((len(self.labels),), dtype=np.float32)
        y = torch.from_numpy(y)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img_name, img, y


@torch.no_grad()
def infer_logits_and_names(model: nn.Module, loader: DataLoader, device: str, use_amp: bool) -> tuple[list[str], np.ndarray]:
    """
    Returns (names, logits) where names preserves concatenation order.
    """
    model.eval()
    all_logits = []
    all_names = []

    if device.startswith("cuda"):
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    else:
        amp_ctx = torch.cpu.amp.autocast(enabled=False)

    for names, x, _ in tqdm(loader, desc="Infer", leave=False):
        all_names.extend(list(names))
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
        all_logits.append(logits.float().cpu().numpy())

    return all_names, np.concatenate(all_logits, axis=0)


def load_model_and_labels(ckpt_path: str, num_classes: int, model_name: str) -> tuple[nn.Module, list[str] | None]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt_labels = ckpt.get("labels", None)
    if ckpt_labels is not None:
        print(f"[CKPT] {os.path.basename(ckpt_path)} labels:", ckpt_labels)

    model = DinoV3MultiLabel(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model, ckpt_labels


def f1_macro_micro(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, np.ndarray]:
    yt = y_true.astype(bool)
    yp = y_pred.astype(bool)

    tp = np.logical_and(yt, yp).sum(axis=0).astype(np.float64)
    fp = np.logical_and(~yt, yp).sum(axis=0).astype(np.float64)
    fn = np.logical_and(yt, ~yp).sum(axis=0).astype(np.float64)

    denom = (2 * tp + fp + fn)
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)

    macro = float(f1.mean())

    tp_all = tp.sum()
    fp_all = fp.sum()
    fn_all = fn.sum()
    denom_micro = (2 * tp_all + fp_all + fn_all)
    micro = float((2 * tp_all) / denom_micro) if denom_micro > 0 else 0.0

    return macro, micro, f1.astype(np.float32)


def build_stage12_preds(p_nd: np.ndarray, p2: np.ndarray, t_nd: float, t2_per_class: np.ndarray) -> np.ndarray:
    """
    Base pipeline: Stage-1 gates ND, Stage-2 predicts defect labels only when not ND.
    """
    nd_idx = LABELS.index(ND_LABEL)
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]

    pred_nd = (p_nd >= t_nd)                       # [N]
    defect_mask = ~pred_nd                         # [N]
    pred_others = defect_mask[:, None] & (p2 >= t2_per_class.reshape(1, -1))  # [N,18]

    y_pred = np.zeros((p_nd.shape[0], len(LABELS)), dtype=np.int32)
    y_pred[:, nd_idx] = pred_nd.astype(np.int32)
    y_pred[:, other_idx] = pred_others.astype(np.int32)
    return y_pred


def enforce_nd_exclusive(y_pred_full: np.ndarray) -> np.ndarray:
    """
    Enforce dataset rule: ND is exclusive and must be 1 iff no defect labels are predicted.
    Guarantees no all-zero rows.
    """
    y = y_pred_full.astype(np.int32, copy=True)
    nd_idx = LABELS.index(ND_LABEL)
    defect_cols = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]
    defect_sum = y[:, defect_cols].sum(axis=1)
    y[:, nd_idx] = (defect_sum == 0).astype(np.int32)
    return y


def apply_stage3_fusion(
    y_base: np.ndarray,
    p_nd: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t_nd: float,
    t2: np.ndarray,
    t3: np.ndarray,
    stage2_labels: list[str],
    stage3_labels: list[str],
) -> tuple[np.ndarray, dict]:
    """
    Implements:
      (A) un-gate ND if stage3 predicts any tail label >= t3
      (B) per-class override: if p3(c) >= t3(c) and p2(c) < t2(c), set final(c)=1
      (C) if stage2 path yields all-zero defect labels (ND=0 and no defects), apply stage3 tail preds
    """
    y = y_base.astype(np.int32, copy=True)

    nd_idx = LABELS.index(ND_LABEL)
    full_idx = {lab: LABELS.index(lab) for lab in LABELS}
    s2_idx = {lab: i for i, lab in enumerate(stage2_labels)}
    s3_idx = {lab: i for i, lab in enumerate(stage3_labels)}

    pred_nd = (p_nd >= t_nd)           # [N]
    pred2 = (p2 >= t2.reshape(1, -1))  # [N,18]
    pred3 = (p3 >= t3.reshape(1, -1))  # [N,K]

    stats = {
        "ungated_by_stage3": 0,
        "override_hits": 0,
        "all_zero_stage2_rows": 0,
        "all_zero_rows_filled_by_stage3": 0,
    }

    # (A) Un-gate ND if Stage-3 fires on any tail label
    if APPLY_STAGE3_UNGATE_ND:
        any3 = (pred3.sum(axis=1) > 0)
        ungate_mask = pred_nd & any3
        if ungate_mask.any():
            stats["ungated_by_stage3"] = int(ungate_mask.sum())
            y[ungate_mask, nd_idx] = 0
            # emit stage3 tail labels
            for lab in stage3_labels:
                j3 = s3_idx[lab]
                fi = full_idx[lab]
                y[ungate_mask, fi] = np.maximum(y[ungate_mask, fi], pred3[ungate_mask, j3].astype(np.int32))

    # (B) Per-class override rule for tail labels
    if APPLY_STAGE3_OVERRIDE:
        override_total = 0
        for lab in stage3_labels:
            if lab not in s2_idx:
                continue  # if stage3 label not part of stage2 head (unlikely here)
            j2 = s2_idx[lab]
            j3 = s3_idx[lab]
            fi = full_idx[lab]

            override_mask = pred3[:, j3] & (~pred2[:, j2])
            if override_mask.any():
                override_total += int(override_mask.sum())
                y[override_mask, fi] = 1
                y[override_mask, nd_idx] = 0
        stats["override_hits"] = int(override_total)

    # (C) If Stage-2 produced all-zero defect labels (while ND=0), apply Stage-3 tail preds
    if APPLY_STAGE3_IF_STAGE2_ALL_ZERO:
        defect_cols = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]
        all_zero_defects = (y[:, nd_idx] == 0) & (y[:, defect_cols].sum(axis=1) == 0)
        stats["all_zero_stage2_rows"] = int(all_zero_defects.sum())

        if all_zero_defects.any():
            # fill with stage3 predictions (tail only)
            before_sum = y[all_zero_defects][:, defect_cols].sum(axis=1)
            for lab in stage3_labels:
                j3 = s3_idx[lab]
                fi = full_idx[lab]
                if STAGE3_ONLY_ADDS:
                    y[all_zero_defects, fi] = np.maximum(y[all_zero_defects, fi], pred3[all_zero_defects, j3].astype(np.int32))
                else:
                    y[all_zero_defects, fi] = pred3[all_zero_defects, j3].astype(np.int32)
            after_sum = y[all_zero_defects][:, defect_cols].sum(axis=1)
            stats["all_zero_rows_filled_by_stage3"] = int((after_sum > before_sum).sum())

    # Final enforcement: ND exclusive and no all-zero
    y = enforce_nd_exclusive(y)
    return y, stats


def save_predictions_csv(df_in: pd.DataFrame, y_pred_full: np.ndarray, out_csv: str):
    image_col = df_in.columns[0]
    out = pd.DataFrame()
    out[image_col] = df_in[image_col].astype(str)
    for i, lab in enumerate(LABELS):
        out[lab] = y_pred_full[:, i].astype(np.int32)
    out.to_csv(out_csv, index=False)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    s1_name = _safe_name(STAGE1_CKPT)
    s2_name = _safe_name(STAGE2_CKPT)
    s3_name = _safe_name(STAGE3_CKPT)
    run_dir = os.path.join(OUT_ROOT, f"e2e__s1_{s1_name}__s2_{s2_name}__s3_{s3_name}")
    os.makedirs(run_dir, exist_ok=True)

    shutil.copyfile(STAGE1_CKPT, os.path.join(run_dir, "stage1_selected.pt"))
    shutil.copyfile(STAGE2_CKPT, os.path.join(run_dir, "stage2_selected.pt"))
    shutil.copyfile(STAGE3_CKPT, os.path.join(run_dir, "stage3_selected.pt"))

    df = pd.read_csv(CSV_PATH)
    has_labels = all(lab in df.columns for lab in LABELS)
    y_true_full = df[LABELS].to_numpy(dtype=np.int32) if has_labels else None

    # Transforms + datasets per stage (different image sizes)
    tf_s1 = SimpleTransform(IMG_SIZE_STAGE1, train=False)
    tf_s2 = SimpleTransform(IMG_SIZE_STAGE2, train=False)
    tf_s3 = SimpleTransform(IMG_SIZE_STAGE3, train=False)

    ds_s1 = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=tf_s1)
    ds_s2 = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=tf_s2)
    ds_s3 = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=tf_s3)

    loader_s1 = DataLoader(ds_s1, batch_size=BATCH_SIZE_STAGE1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    loader_s2 = DataLoader(ds_s2, batch_size=BATCH_SIZE_STAGE2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    loader_s3 = DataLoader(ds_s3, batch_size=BATCH_SIZE_STAGE3, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print("Loading models...")
    m1, ckpt_labels_s1 = load_model_and_labels(STAGE1_CKPT, num_classes=1, model_name=MODEL_NAME_STAGE_1)
    m2, ckpt_labels_s2 = load_model_and_labels(STAGE2_CKPT, num_classes=len(LABELS_WO_ND), model_name=MODEL_NAME_STAGE_2)
    m3, ckpt_labels_s3 = load_model_and_labels(STAGE3_CKPT, num_classes=len(STAGE3_LABELS), model_name=MODEL_NAME_STAGE_3)

    # If stage3 ckpt stored labels, you can enforce consistency:
    if ckpt_labels_s3 is not None:
        # You can comment this out if you prefer manual STAGE3_LABELS.
        if list(ckpt_labels_s3) != list(STAGE3_LABELS):
            raise RuntimeError(
                f"STAGE3_LABELS mismatch.\n"
                f"Config STAGE3_LABELS: {STAGE3_LABELS}\n"
                f"CKPT labels:          {ckpt_labels_s3}\n"
                f"Fix STAGE3_LABELS to match the stage3 checkpoint head order."
            )

    print(f"Inferring Stage-1 logits @ img_size={IMG_SIZE_STAGE1} ...")
    names1, logits1 = infer_logits_and_names(m1, loader_s1, DEVICE, use_amp=USE_AMP_EVAL)
    logits1 = logits1.reshape(-1)
    p1 = _sigmoid_np(logits1)

    if STAGE1_OUTPUT_MODE == "ND":
        p_nd = p1
    elif STAGE1_OUTPUT_MODE == "DEFECT_PRESENT":
        p_nd = 1.0 - p1
    else:
        raise ValueError(f"Unknown STAGE1_OUTPUT_MODE: {STAGE1_OUTPUT_MODE}")

    print(f"Inferring Stage-2 logits @ img_size={IMG_SIZE_STAGE2} ...")
    names2, logits2 = infer_logits_and_names(m2, loader_s2, DEVICE, use_amp=USE_AMP_EVAL)
    p2 = _sigmoid_np(logits2)

    print(f"Inferring Stage-3 logits @ img_size={IMG_SIZE_STAGE3} ...")
    names3, logits3 = infer_logits_and_names(m3, loader_s3, DEVICE, use_amp=USE_AMP_EVAL)
    p3 = _sigmoid_np(logits3)

    # Safety: ensure consistent ordering
    if names1 != names2 or names1 != names3:
        for i in range(min(len(names1), len(names2), len(names3))):
            if not (names1[i] == names2[i] == names3[i]):
                raise RuntimeError(
                    "Loaders produced different sample order.\n"
                    f"First mismatch at index {i}:\n"
                    f"  stage1={names1[i]}\n"
                    f"  stage2={names2[i]}\n"
                    f"  stage3={names3[i]}\n"
                    "Ensure all loaders use shuffle=False and identical CSV/order."
                )
        raise RuntimeError("Names mismatch detected (unknown).")

    # Load thresholds
    t_stage1 = load_stage1_threshold_txt(STAGE1_THRESHOLD_TXT)
    t2_pc = load_thresholds_json(STAGE2_THRESHOLD_JSON, STAGE2_LABELS)
    t3_pc = load_thresholds_json(STAGE3_THRESHOLD_JSON, STAGE3_LABELS)

    if STAGE1_OUTPUT_MODE == "ND":
        t_nd = float(t_stage1)
        print(f"[Stage1] Using provided ND threshold: t_nd={t_nd:.6f}")
    else:
        t_nd = float(1.0 - t_stage1)
        print(f"[Stage1] Using provided DEFECT threshold t_defect={t_stage1:.6f} -> derived t_nd={t_nd:.6f}")

    # -------- Base Stage1+Stage2 preds --------
    y_pred_base = build_stage12_preds(p_nd, p2, t_nd, t2_pc)
    y_pred_base = enforce_nd_exclusive(y_pred_base)

    # -------- Stage3 fusion preds --------
    y_pred_s3, stats = apply_stage3_fusion(
        y_base=y_pred_base,
        p_nd=p_nd,
        p2=p2,
        p3=p3,
        t_nd=t_nd,
        t2=t2_pc,
        t3=t3_pc,
        stage2_labels=STAGE2_LABELS,
        stage3_labels=STAGE3_LABELS,
    )

    # -------- Evaluation (if labels exist) --------
    macro_base = micro_base = None
    macro_s3 = micro_s3 = None
    if has_labels:
        macro_base, micro_base, _ = f1_macro_micro(y_true_full, y_pred_base)
        macro_s3, micro_s3, _ = f1_macro_micro(y_true_full, y_pred_s3)

        print("\n================ RESULTS (VAL) ================")
        print(f"Stage1: {os.path.basename(STAGE1_CKPT)}  img={IMG_SIZE_STAGE1}")
        print(f"Stage2: {os.path.basename(STAGE2_CKPT)}  img={IMG_SIZE_STAGE2}")
        print(f"Stage3: {os.path.basename(STAGE3_CKPT)}  img={IMG_SIZE_STAGE3}")
        print(f"t_nd:   {t_nd:.6f}")
        print("----------------------------------------------")
        print(f"[Base S1+S2]   macro_f1={macro_base:.6f}  micro_f1={micro_base:.6f}")
        print(f"[With Stage3]  macro_f1={macro_s3:.6f}  micro_f1={micro_s3:.6f}")
        print("----------------------------------------------")
        print("Stage3 fusion stats:", stats)
        print("==============================================\n")
    else:
        print("\n[Info] CSV has no labels; skipping F1 evaluation and exporting predictions only.\n")
        print("Stage3 fusion stats:", stats)

    # -------- Save CSV outputs --------
    csv_base = os.path.join(run_dir, f"pred_base_s1s2__macro_{_float_tag(macro_base or 0.0)}.csv")
    csv_s3   = os.path.join(run_dir, f"pred_fused_s1s2s3__macro_{_float_tag(macro_s3 or 0.0)}.csv")

    save_predictions_csv(df, y_pred_base, csv_base)
    save_predictions_csv(df, y_pred_s3, csv_s3)

    # -------- Save run metadata --------
    meta = {
        "csv": CSV_PATH,
        "images_dir": IMAGES_DIR,
        "device": DEVICE,
        "amp_eval": USE_AMP_EVAL,
        "image_sizes": {"stage1": IMG_SIZE_STAGE1, "stage2": IMG_SIZE_STAGE2, "stage3": IMG_SIZE_STAGE3},
        "batch_sizes": {"stage1": BATCH_SIZE_STAGE1, "stage2": BATCH_SIZE_STAGE2, "stage3": BATCH_SIZE_STAGE3},
        "stage1": {"ckpt": os.path.basename(STAGE1_CKPT), "output_mode": STAGE1_OUTPUT_MODE, "t_nd": float(t_nd)},
        "stage2": {
            "ckpt": os.path.basename(STAGE2_CKPT),
            "labels_order": STAGE2_LABELS,
            "thresholds": {lab: float(t) for lab, t in zip(STAGE2_LABELS, t2_pc.tolist())},
        },
        "stage3": {
            "ckpt": os.path.basename(STAGE3_CKPT),
            "labels_order": STAGE3_LABELS,
            "thresholds": {lab: float(t) for lab, t in zip(STAGE3_LABELS, t3_pc.tolist())},
            "fusion": {
                "ungate_nd": APPLY_STAGE3_UNGATE_ND,
                "override_rule": APPLY_STAGE3_OVERRIDE,
                "apply_if_stage2_all_zero": APPLY_STAGE3_IF_STAGE2_ALL_ZERO,
                "only_adds": STAGE3_ONLY_ADDS,
                "stats": stats,
            },
        },
        "outputs": {
            "base_csv": os.path.basename(csv_base),
            "fused_csv": os.path.basename(csv_s3),
        },
        "val_scores": None if not has_labels else {
            "base": {"macro_f1": float(macro_base), "micro_f1": float(micro_base)},
            "fused": {"macro_f1": float(macro_s3), "micro_f1": float(micro_s3)},
        },
    }

    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved outputs to:", run_dir)
    print("  - stage1_selected.pt")
    print("  - stage2_selected.pt")
    print("  - stage3_selected.pt")
    print("  -", os.path.basename(csv_base), "<-- base S1+S2")
    print("  -", os.path.basename(csv_s3), "<-- fused S1+S2+S3")
    print("  - run_meta.json")


if __name__ == "__main__":
    main()
