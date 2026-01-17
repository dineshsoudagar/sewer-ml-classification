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


# -------------------------
# Predict Config (Stage 2)
# -------------------------
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train_stage2_sanity_5k.csv"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"

OUT_CSV = r"outputs_stage2_predictions.csv"

CKPT_PATH = r"outputs_stage2_vit_base_tesk_5k\best.pt"  # <- set this

# Optional: used only if checkpoint does not contain model_name/img_size/labels/thresholds
MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
IMG_SIZE = 256

BATCH_SIZE = 32
NUM_WORKERS = 8

USE_AMP = False
SEED = 42

LABELS = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA", "ND"]
ND_LABEL = "ND"

SEWER_MEAN = [0.523, 0.453, 0.345]
SEWER_STD = [0.210, 0.199, 0.154]


class SewerMLInferenceDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, transform=None):
        self.df = pd.read_csv(csv_path)
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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_thresholds_from_ckpt_or_file(out_dir: str, ckpt: dict, labels_wo_nd: list[str]) -> np.ndarray:
    # Prefer ckpt thresholds (saved as numpy array in your save_checkpoint_multilabel)
    if "thresholds" in ckpt and ckpt["thresholds"] is not None:
        th = np.asarray(ckpt["thresholds"], dtype=np.float32).reshape(-1)
        if th.shape[0] != len(labels_wo_nd):
            raise ValueError(f"Checkpoint thresholds size mismatch: {th.shape[0]} vs labels {len(labels_wo_nd)}")
        return th

    # Fallback to best_thresholds.json (your training script writes it)
    js = os.path.join(out_dir, "best_thresholds.json")
    if os.path.isfile(js):
        with open(js, "r") as f:
            d = json.load(f)
        th = np.array([float(d[l]) for l in labels_wo_nd], dtype=np.float32)
        return th

    # Final fallback
    return np.full((len(labels_wo_nd),), 0.5, dtype=np.float32)


@torch.no_grad()
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device)

    ckpt_model_name = ckpt.get("model_name", MODEL_NAME)
    ckpt_img_size = int(ckpt.get("img_size", IMG_SIZE))

    # Determine defect labels from checkpoint if present, else from LABELS
    if "labels" in ckpt and ckpt["labels"] is not None:
        labels_wo_nd = list(ckpt["labels"])
    else:
        labels_wo_nd = [l for l in LABELS if l != ND_LABEL]

    num_classes = len(labels_wo_nd)

    model = DinoV3MultiLabel(ckpt_model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    out_dir = os.path.dirname(CKPT_PATH)
    thresholds = _load_thresholds_from_ckpt_or_file(out_dir, ckpt, labels_wo_nd)

    tfm = SimpleTransform(ckpt_img_size, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    ds = SewerMLInferenceDataset(VAL_CSV, VAL_IMAGES, transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_names = []
    all_logits = []

    for names, x in tqdm(loader, desc="[Predict Stage2]"):
        x = x.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(x)  # [B,C]
        all_names.extend(list(names))
        all_logits.append(logits.float().cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)  # [N,C]
    probs = _sigmoid_np(all_logits)
    preds = (probs >= thresholds[None, :]).astype(np.int32)  # [N,C]

    # Map image -> predicted vector
    name_to_vec = {n: preds[i] for i, n in enumerate(all_names)}

    # --- Save in same format/columns as input VAL_CSV ---
    df_in = pd.read_csv(VAL_CSV)
    image_col = df_in.columns[0]
    cols = list(df_in.columns)

    df_out = df_in.copy()

    # If CSV has label columns, overwrite them with predictions while preserving exact column order.
    # We keep "same format as input", so we only touch columns that exist in df_in.
    # For Stage2-only inference: ND is set to 0 (because stage2 assumes defect images).
    for c in cols[1:]:
        if c == ND_LABEL:
            df_out[c] = 0
        elif c in labels_wo_nd:
            j = labels_wo_nd.index(c)
            df_out[c] = df_out[image_col].map(lambda n: int(name_to_vec.get(str(n), np.zeros((num_classes,), dtype=np.int32))[j])).astype(int)
        else:
            # Column exists in input but is not part of stage2 labels (leave as-is or set to 0).
            # To keep output strictly "predictions", set to 0:
            df_out[c] = 0

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"[Stage2] Saved: {OUT_CSV}")
    print(f"[Stage2] labels={labels_wo_nd}")
    print(f"[Stage2] ckpt={CKPT_PATH}")


if __name__ == "__main__":
    main()
