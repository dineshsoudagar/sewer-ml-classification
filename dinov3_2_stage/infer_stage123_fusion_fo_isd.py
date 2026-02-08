# eval_e2e_stage1_stage2_stage3_fusion.py
# Stage1 + Stage2 + Stage3 fusion evaluation on VAL (labeled CSV).
# NEW POLICY (two-tier Stage-3):
#   A) Rescue rows (Stage1=defect path AND Stage2 predicts ALL-ZERO): apply Stage-3 with ALL 5 tail labels using normal t3.
#   B) Global add on entire dataset: Stage-3 may add ONLY FO+IS, but at a VERY HIGH threshold (t3_hi).

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

# =========================
# EDIT THESE
# =========================

CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

# ---- Stage 1 ----
STAGE1_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\epoch05_f1_0.92718_acc_0.93508.pt"
STAGE1_THRESHOLD_TXT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage1_vit_samll_plus_img_384\best_threshold.txt"
STAGE1_OUTPUT_MODE = "ND"  # "ND" or "DEFECT_PRESENT"
MODEL_NAME_STAGE_1 = "vit_small_plus_patch16_dinov3.lvd1689m"
IMG_SIZE_STAGE1 = 384
BATCH_SIZE_STAGE1 = 64

# ---- Stage 2 ----
STAGE2_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\epoch14_macroF1_0.73940_microF1_0.80718.pt"
STAGE2_THRESHOLD_JSON = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage2_vit_base_fn_tnd_on_384\best_thresholds_epoch14_macroF1_0.73940_microF1_0.80718.json"
MODEL_NAME_STAGE_2 = "vit_base_patch16_dinov3.lvd1689m"
IMG_SIZE_STAGE2 = 384
BATCH_SIZE_STAGE2 = 64

# ---- Stage 3 ---- (tail specialist)
STAGE3_CKPT = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_low_labels_384\best.pt"
STAGE3_THRESHOLD_JSON = r"D:\expandAI-hiring\expandai-hiring-sewer\sewer-ml-classification\dinov3_2_stage\outputs_stage3_low_labels_384\stage3_thresholds_val_nd0.json"
MODEL_NAME_STAGE_3 = "vit_small_patch16_dinov3.lvd1689m"
IMG_SIZE_STAGE3 = 384
BATCH_SIZE_STAGE3 = 64

# Stage-3 label order (must match stage3 head order)
STAGE3_LABELS = ["FO", "RB", "IS", "DE", "IN"]

# Full label space (for scoring/export)
LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"
]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

# Stage2 head order (confirmed)
STAGE2_LABELS = [
    "RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE",
    "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA"
]

NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP_EVAL = True

# Output folder
OUT_ROOT = "e2e_exports_stage1_stage2_stage3_fusion__all5_rescue__global_FOIS_hi"

# ------------------------------
# NEW: Global Stage-3 add policy (applies on entire dataset)
# ------------------------------
STAGE3_GLOBAL_ADD_LABELS = ["FO", "IS"]
STAGE3_GLOBAL_HI_FLOOR = 0.99               # try 0.98 / 0.99 / 0.995
STAGE3_GLOBAL_REQUIRE_STAGE2_MISS = True    # safer: only add if stage2 missed that class
STAGE3_GLOBAL_APPLY_ON_ALL_ROWS = True      # entire dataset (includes stage1 ND-gated rows)

# =========================


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


def load_thresholds_json(path: str, labels: list[str]) -> np.ndarray:
    with open(path, "r") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("Threshold JSON must be dict: {label: threshold}")

    missing = [lab for lab in labels if lab not in d]
    extra = [lab for lab in d.keys() if lab not in labels]
    if missing:
        raise ValueError(f"Threshold JSON missing labels: {missing}")
    if extra:
        print(f"[Warn] Threshold JSON has extra keys not used: {extra}")

    t = np.array([float(d[lab]) for lab in labels], dtype=np.float32)
    if np.any(t < 0.0) or np.any(t > 1.0):
        bad = [(lab, float(tt)) for lab, tt in zip(labels, t.tolist()) if not (0.0 <= tt <= 1.0)]
        raise ValueError(f"Thresholds out of [0,1]: {bad}")
    return t


class SewerMLFullDataset(Dataset):
    """
    Returns img_name, image_tensor, y
    Keeps row order identical to CSV.
    """
    def __init__(self, csv_path: str, images_dir: str, labels: list[str], transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.image_col = self.df.columns[0]
        self.has_labels = all(lab in self.df.columns for lab in labels)
        if not self.has_labels:
            raise RuntimeError("CSV must contain label columns for evaluation.")

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

        y = row[self.labels].to_numpy(dtype=np.float32)
        y = torch.from_numpy(y)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img_name, img, y


@torch.no_grad()
def infer_logits(model: nn.Module, loader: DataLoader, device: str, use_amp: bool) -> np.ndarray:
    model.eval()
    all_logits = []

    if device.startswith("cuda"):
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
    else:
        amp_ctx = torch.cpu.amp.autocast(enabled=False)

    for _, x, _ in tqdm(loader, desc="Infer", leave=False):
        x = x.to(device, non_blocking=True)
        with amp_ctx:
            logits = model(x)
        all_logits.append(logits.float().cpu().numpy())

    return np.concatenate(all_logits, axis=0)


def load_model(ckpt_path: str, num_classes: int, model_name: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "labels" in ckpt:
        print(f"[CKPT] {model_name} labels:", ckpt["labels"])
    model = DinoV3MultiLabel(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


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


def build_stage1_stage2_preds(p_nd: np.ndarray, p2: np.ndarray, t_nd: float, t2_pc: np.ndarray) -> np.ndarray:
    """
    Base pipeline:
      if p_nd >= t_nd => ND=1, others=0
      else => ND=0, others=(p2 >= t2_pc)
    """
    nd_idx = LABELS.index(ND_LABEL)
    other_idx = [i for i, lab in enumerate(LABELS) if lab != ND_LABEL]

    pred_nd = (p_nd >= t_nd)
    defect_mask = ~pred_nd
    pred_others = defect_mask[:, None] & (p2 >= t2_pc.reshape(1, -1))

    y_pred = np.zeros((p_nd.shape[0], len(LABELS)), dtype=np.int32)
    y_pred[:, nd_idx] = pred_nd.astype(np.int32)
    y_pred[:, other_idx] = pred_others.astype(np.int32)
    return y_pred


def fuse_stage3(
    y_pred_base: np.ndarray,      # [N,19] base preds from stage1+stage2
    p_nd: np.ndarray,             # [N]
    p2: np.ndarray,               # [N,18]
    p3: np.ndarray,               # [N,5]
    t_nd: float,
    t2: np.ndarray,               # [18]
    t3: np.ndarray,               # [5] normal thresholds
    only_adds: bool,
    ungate_nd: bool,
    override: bool,
    stage2_all_zero_rescue: bool,
    require_stage1_defect_for_stage3: bool = True,
    # NEW:
    t3_hi: np.ndarray | None = None,             # [5] high-confidence thresholds
    global_add_labels: list[str] | None = None,  # e.g. ["FO","IS"]
    global_apply_on_all_rows: bool = True,
    global_require_stage2_miss: bool = True,
) -> np.ndarray:
    """
    Two-tier Stage-3:
      - Rescue: if stage1 defect path AND stage2 all-zero -> apply stage3 ALL 5 labels using t3.
      - Global add: add ONLY subset labels using very high thresholds t3_hi (optionally on all rows).

    Existing optional policies:
      - ungate_nd:
          if stage1 says ND but stage3 fires any tail label -> set ND=0 and emit tail labels (riskier)
      - override:
          if p3(c)>=t3(c) and stage2 missed c -> set c=1 (adds missing tail labels)
      - only_adds:
          stage3 never removes a stage2 positive (recommended True)
      - require_stage1_defect_for_stage3:
          prevents stage3 touching ND-gated rows (except global add if enabled)
    """
    out = y_pred_base.copy().astype(np.int32)

    nd_idx = LABELS.index(ND_LABEL)
    idx_full = {lab: i for i, lab in enumerate(LABELS)}
    idx2 = {lab: i for i, lab in enumerate(STAGE2_LABELS)}
    idx3 = {lab: i for i, lab in enumerate(STAGE3_LABELS)}

    pred_nd = (p_nd >= t_nd)
    defect_mask = ~pred_nd

    # Stage2 per-class predictions (for comparisons)
    stage2_pred = (p2 >= t2.reshape(1, -1))

    # Optionally prevent stage3 from touching ND-gated rows at all (for normal stage3 ops)
    stage3_allowed_mask = defect_mask.copy()
    if not require_stage1_defect_for_stage3:
        stage3_allowed_mask = np.ones_like(defect_mask, dtype=bool)

    # 1) UNGATE ND (riskier)
    if ungate_nd:
        tail_fire = (p3 >= t3.reshape(1, -1)).any(axis=1)
        ungate_mask = pred_nd & tail_fire
        if ungate_mask.any():
            out[ungate_mask, nd_idx] = 0
            for lab in STAGE3_LABELS:
                j3 = idx3[lab]
                jF = idx_full[lab]
                out[ungate_mask, jF] = (p3[ungate_mask, j3] >= t3[j3]).astype(np.int32)

    # 2) STAGE2 ALL-ZERO rescue on defect path: apply ALL 5 Stage3 labels with normal t3
    if stage2_all_zero_rescue:
        stage2_any = stage2_pred.any(axis=1)
        rescue_mask = stage3_allowed_mask & (~stage2_any)
        if rescue_mask.any():
            for lab in STAGE3_LABELS:
                j3 = idx3[lab]
                jF = idx_full[lab]
                out[rescue_mask, jF] = (p3[rescue_mask, j3] >= t3[j3]).astype(np.int32)
            out[rescue_mask, nd_idx] = 0

    # 3) Per-class override (adds missing tail labels) using normal t3
    if override:
        for lab in STAGE3_LABELS:
            j2 = idx2.get(lab, None)
            if j2 is None:
                continue
            j3 = idx3[lab]
            jF = idx_full[lab]

            missed_by_s2 = ~stage2_pred[:, j2]
            fire_s3 = (p3[:, j3] >= t3[j3])
            add_mask = stage3_allowed_mask & missed_by_s2 & fire_s3

            if only_adds:
                out[add_mask, jF] = 1
            else:
                out[add_mask, jF] = 1

            out[add_mask, nd_idx] = 0

    # 4) NEW: Global high-confidence add for FO/IS only, using t3_hi
    if (t3_hi is not None) and (global_add_labels is not None) and (len(global_add_labels) > 0):
        if global_apply_on_all_rows:
            global_mask = np.ones_like(defect_mask, dtype=bool)  # entire dataset (includes ND-gated)
        else:
            global_mask = stage3_allowed_mask  # only defect path

        for lab in global_add_labels:
            if lab not in idx3:
                continue
            j3 = idx3[lab]
            jF = idx_full[lab]

            fire_hi = (p3[:, j3] >= t3_hi[j3])

            if global_require_stage2_miss and (lab in idx2):
                missed = ~stage2_pred[:, idx2[lab]]
                add_mask = global_mask & missed & fire_hi
            else:
                add_mask = global_mask & fire_hi

            out[add_mask, jF] = 1
            out[add_mask, nd_idx] = 0  # if we add any defect -> ND must be 0

    # Safety: if any defect label set on a row, ND must be 0
    any_def = out[:, :nd_idx].sum(axis=1)  # ND is last, so this is all defects
    out[any_def > 0, nd_idx] = 0

    return out


def save_predictions_csv(df_in: pd.DataFrame, y_pred_full: np.ndarray, out_csv: str):
    image_col = df_in.columns[0]
    out = pd.DataFrame()
    out[image_col] = df_in[image_col].astype(str)
    for i, lab in enumerate(LABELS):
        out[lab] = y_pred_full[:, i].astype(np.int32)
    out.to_csv(out_csv, index=False)


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    run_dir = os.path.join(
        OUT_ROOT,
        f"e2e__s1_{_safe_name(STAGE1_CKPT)}__s2_{_safe_name(STAGE2_CKPT)}__s3_{_safe_name(STAGE3_CKPT)}"
    )
    os.makedirs(run_dir, exist_ok=True)

    # copy artifacts
    shutil.copyfile(STAGE1_CKPT, os.path.join(run_dir, "stage1_selected.pt"))
    shutil.copyfile(STAGE2_CKPT, os.path.join(run_dir, "stage2_selected.pt"))
    shutil.copyfile(STAGE3_CKPT, os.path.join(run_dir, "stage3_selected.pt"))

    df = pd.read_csv(CSV_PATH)
    y_true_full = df[LABELS].to_numpy(dtype=np.int32)

    # ---- load thresholds (no search) ----
    t_stage1 = load_stage1_threshold_txt(STAGE1_THRESHOLD_TXT)
    t2_pc = load_thresholds_json(STAGE2_THRESHOLD_JSON, STAGE2_LABELS)
    t3_pc = load_thresholds_json(STAGE3_THRESHOLD_JSON, STAGE3_LABELS)

    # Stage1 threshold in ND space
    if STAGE1_OUTPUT_MODE == "ND":
        t_nd = float(t_stage1)
    elif STAGE1_OUTPUT_MODE == "DEFECT_PRESENT":
        t_nd = float(1.0 - t_stage1)
    else:
        raise ValueError(f"Unknown STAGE1_OUTPUT_MODE: {STAGE1_OUTPUT_MODE}")

    # NEW: build t3_hi for global FO/IS add
    t3_hi = t3_pc.copy()
    for lab in STAGE3_GLOBAL_ADD_LABELS:
        j = STAGE3_LABELS.index(lab)
        t3_hi[j] = max(float(t3_hi[j]), float(STAGE3_GLOBAL_HI_FLOOR))

    print(f"[Stage1] t_nd={t_nd:.6f}")
    print(f"[Stage2] loaded per-class thresholds for {len(STAGE2_LABELS)} labels")
    print(f"[Stage3] loaded per-class thresholds for {len(STAGE3_LABELS)} labels")
    print(f"[Stage3] global-add labels={STAGE3_GLOBAL_ADD_LABELS} hi_floor={STAGE3_GLOBAL_HI_FLOOR} require_s2_miss={STAGE3_GLOBAL_REQUIRE_STAGE2_MISS} all_rows={STAGE3_GLOBAL_APPLY_ON_ALL_ROWS}")
    print(f"[Stage3] t3 (normal): { {lab: float(t) for lab, t in zip(STAGE3_LABELS, t3_pc.tolist())} }")
    print(f"[Stage3] t3_hi:      { {lab: float(t) for lab, t in zip(STAGE3_LABELS, t3_hi.tolist())} }")

    # ---- loaders per stage (supports different image sizes / batch sizes) ----
    ds_s1 = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=SimpleTransform(IMG_SIZE_STAGE1, train=False))
    ds_s2 = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=SimpleTransform(IMG_SIZE_STAGE2, train=False))
    ds_s3 = SewerMLFullDataset(CSV_PATH, IMAGES_DIR, LABELS, transform=SimpleTransform(IMG_SIZE_STAGE3, train=False))

    dl_s1 = DataLoader(ds_s1, batch_size=BATCH_SIZE_STAGE1, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    dl_s2 = DataLoader(ds_s2, batch_size=BATCH_SIZE_STAGE2, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    dl_s3 = DataLoader(ds_s3, batch_size=BATCH_SIZE_STAGE3, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # ---- load models ----
    print("Loading models...")
    m1 = load_model(STAGE1_CKPT, num_classes=1, model_name=MODEL_NAME_STAGE_1)
    m2 = load_model(STAGE2_CKPT, num_classes=len(LABELS_WO_ND), model_name=MODEL_NAME_STAGE_2)
    m3 = load_model(STAGE3_CKPT, num_classes=len(STAGE3_LABELS), model_name=MODEL_NAME_STAGE_3)

    # ---- infer ----
    print("Inferring Stage-1 logits...")
    logits1 = infer_logits(m1, dl_s1, DEVICE, use_amp=USE_AMP_EVAL).reshape(-1)
    p1 = _sigmoid_np(logits1)
    p_nd = p1 if STAGE1_OUTPUT_MODE == "ND" else (1.0 - p1)

    print("Inferring Stage-2 logits...")
    logits2 = infer_logits(m2, dl_s2, DEVICE, use_amp=USE_AMP_EVAL)
    p2 = _sigmoid_np(logits2)  # [N,18]

    print("Inferring Stage-3 logits...")
    logits3 = infer_logits(m3, dl_s3, DEVICE, use_amp=USE_AMP_EVAL)
    p3 = _sigmoid_np(logits3)  # [N,5]

    # ---- base preds ----
    y_base = build_stage1_stage2_preds(p_nd, p2, t_nd, t2_pc)
    macro_base, micro_base, _ = f1_macro_micro(y_true_full, y_base)
    print("\n========== BASE (Stage1+Stage2) ==========")
    print(f"macro_f1={macro_base:.6f} micro_f1={micro_base:.6f}")

    # ---- evaluate fusion variants ----
    variants = []

    def add_variant(name, only_adds, ungate_nd, override, allzero):
        variants.append({
            "name": name,
            "only_adds": only_adds,
            "ungate_nd": ungate_nd,
            "override": override,
            "allzero": allzero,
            "require_stage1_defect_for_stage3": True,  # keep True for normal stage3 ops
        })

    # Minimal / safe core set (your best earlier was usually allzero rescue)
    add_variant("S3_allzero_rescue_only_adds", True, False, False, True)

    # Optional comparisons:
    add_variant("S3_ungate_only", True, True, False, False)
    add_variant("S3_ungate+allzero", True, True, False, True)
    add_variant("S3_override_only_adds", True, False, True, False)
    add_variant("S3_override+allzero_only_adds", True, False, True, True)
    add_variant("S3_ungate+override", True, True, True, False)
    add_variant("S3_ungate+override+allzero", True, True, True, True)

    results = []
    topk_to_save = 5

    for v in variants:
        y_fused = fuse_stage3(
            y_pred_base=y_base,
            p_nd=p_nd, p2=p2, p3=p3,
            t_nd=t_nd, t2=t2_pc, t3=t3_pc,
            only_adds=v["only_adds"],
            ungate_nd=v["ungate_nd"],
            override=v["override"],
            stage2_all_zero_rescue=v["allzero"],
            require_stage1_defect_for_stage3=v["require_stage1_defect_for_stage3"],
            # NEW global add:
            t3_hi=t3_hi,
            global_add_labels=STAGE3_GLOBAL_ADD_LABELS,
            global_apply_on_all_rows=STAGE3_GLOBAL_APPLY_ON_ALL_ROWS,
            global_require_stage2_miss=STAGE3_GLOBAL_REQUIRE_STAGE2_MISS,
        )
        macro, micro, _ = f1_macro_micro(y_true_full, y_fused)

        results.append({
            "name": v["name"],
            "macro_f1": float(macro),
            "micro_f1": float(micro),
            "only_adds": v["only_adds"],
            "ungate_nd": v["ungate_nd"],
            "override": v["override"],
            "allzero": v["allzero"],
        })

    results_sorted = sorted(results, key=lambda r: r["macro_f1"], reverse=True)

    print("\n========== FUSION VARIANTS (ranked by macro_f1) ==========")
    for r in results_sorted:
        print(
            f"{r['macro_f1']:.6f}  micro={r['micro_f1']:.6f}  "
            f"{r['name']}  (ungate={r['ungate_nd']}, override={r['override']}, allzero={r['allzero']})"
        )
    print("==========================================================\n")

    # Save best K CSVs and summary json
    best_pack = []
    for i, r in enumerate(results_sorted[:topk_to_save]):
        v = next(x for x in variants if x["name"] == r["name"])
        y_fused = fuse_stage3(
            y_pred_base=y_base,
            p_nd=p_nd, p2=p2, p3=p3,
            t_nd=t_nd, t2=t2_pc, t3=t3_pc,
            only_adds=v["only_adds"],
            ungate_nd=v["ungate_nd"],
            override=v["override"],
            stage2_all_zero_rescue=v["allzero"],
            require_stage1_defect_for_stage3=v["require_stage1_defect_for_stage3"],
            # NEW global add:
            t3_hi=t3_hi,
            global_add_labels=STAGE3_GLOBAL_ADD_LABELS,
            global_apply_on_all_rows=STAGE3_GLOBAL_APPLY_ON_ALL_ROWS,
            global_require_stage2_miss=STAGE3_GLOBAL_REQUIRE_STAGE2_MISS,
        )

        out_csv = os.path.join(run_dir, f"pred_e2e_{i+1:02d}_{r['name']}_macro_{_float_tag(r['macro_f1'])}.csv")
        save_predictions_csv(df, y_fused, out_csv)

        best_pack.append({
            "rank": i + 1,
            "name": r["name"],
            "macro_f1": r["macro_f1"],
            "micro_f1": r["micro_f1"],
            "csv": os.path.basename(out_csv),
            "config": v,
        })

    summary = {
        "csv_used": CSV_PATH,
        "stage1": {
            "ckpt": os.path.basename(STAGE1_CKPT),
            "img_size": IMG_SIZE_STAGE1,
            "t_nd": float(t_nd),
            "output_mode": STAGE1_OUTPUT_MODE,
        },
        "stage2": {
            "ckpt": os.path.basename(STAGE2_CKPT),
            "img_size": IMG_SIZE_STAGE2,
            "labels": STAGE2_LABELS,
            "thresholds": {lab: float(t) for lab, t in zip(STAGE2_LABELS, t2_pc.tolist())},
        },
        "stage3": {
            "ckpt": os.path.basename(STAGE3_CKPT),
            "img_size": IMG_SIZE_STAGE3,
            "labels": STAGE3_LABELS,
            "thresholds_t3": {lab: float(t) for lab, t in zip(STAGE3_LABELS, t3_pc.tolist())},
            "thresholds_t3_hi": {lab: float(t) for lab, t in zip(STAGE3_LABELS, t3_hi.tolist())},
            "global_add": {
                "labels": STAGE3_GLOBAL_ADD_LABELS,
                "hi_floor": float(STAGE3_GLOBAL_HI_FLOOR),
                "require_stage2_miss": bool(STAGE3_GLOBAL_REQUIRE_STAGE2_MISS),
                "apply_on_all_rows": bool(STAGE3_GLOBAL_APPLY_ON_ALL_ROWS),
            },
        },
        "base_stage1_stage2": {
            "macro_f1": float(macro_base),
            "micro_f1": float(micro_base),
        },
        "ranked_variants": results_sorted,
        "saved_topk": best_pack,
    }

    with open(os.path.join(run_dir, "summary_stage1_stage2_stage3.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved outputs to:", run_dir)
    print("  - summary_stage1_stage2_stage3.json")
    for bp in best_pack:
        print(f"  - {bp['csv']}")


if __name__ == "__main__":
    main()
