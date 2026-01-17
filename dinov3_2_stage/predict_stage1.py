import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from model import DinoV3MultiLabel
from train_utils import SimpleTransform, set_seed


# -------------------------
# Predict Config (Stage 1)
# -------------------------
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

OUT_CSV = r"outputs_stage1_predictions.csv"

CKPT_PATH = r"outputs_stage1_vit_small_plus\best.pt"  # <- set this

# Optional: used only if checkpoint does not contain model_name/img_size/threshold
MODEL_NAME = "vit_small_plus_patch16_dinov3.lvd1689m"
IMG_SIZE = 256

BATCH_SIZE = 64
NUM_WORKERS = 8

USE_AMP = False
SEED = 42

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


def _load_threshold_from_ckpt_or_file(out_dir: str, ckpt: dict) -> float:
    if "threshold" in ckpt:
        return float(ckpt["threshold"])

    # Fallback to best_threshold.txt if present
    txt = os.path.join(out_dir, "best_threshold.txt")
    if os.path.isfile(txt):
        with open(txt, "r") as f:
            return float(f.read().strip())

    # Final fallback
    return 0.5


@torch.no_grad()
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    ckpt = torch.load(CKPT_PATH, map_location=device)
    ckpt_model_name = ckpt.get("model_name", MODEL_NAME)
    ckpt_img_size = int(ckpt.get("img_size", IMG_SIZE))

    # Stage-1 is always 1 logit (ND)
    model = DinoV3MultiLabel(ckpt_model_name, num_classes=1, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    out_dir = os.path.dirname(CKPT_PATH)
    threshold = _load_threshold_from_ckpt_or_file(out_dir, ckpt)

    tfm = SimpleTransform(ckpt_img_size, train=False, SEWER_MEAN=SEWER_MEAN, SEWER_STD=SEWER_STD)
    ds = SewerMLInferenceDataset(VAL_CSV, VAL_IMAGES, transform=tfm)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_names = []
    all_preds = []

    for names, x in tqdm(loader, desc="[Predict Stage1]"):
        x = x.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(x)  # [B,1]
        probs = torch.sigmoid(logits).squeeze(1)  # [B]
        preds = (probs >= threshold).to(torch.int32).cpu().numpy()

        all_names.extend(list(names))
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)
    name_to_pred = {n: int(p) for n, p in zip(all_names, all_preds)}

    # --- Save in same format/columns as input VAL_CSV ---
    df_in = pd.read_csv(VAL_CSV)
    image_col = df_in.columns[0]
    cols = list(df_in.columns)

    if ND_LABEL not in df_in.columns:
        raise ValueError(f"Input CSV must contain '{ND_LABEL}' column to keep same format. Got columns: {cols}")

    df_out = df_in.copy()

    # Set all label columns (except image) to 0 first, then fill ND with prediction
    for c in cols[1:]:
        df_out[c] = 0

    df_out[ND_LABEL] = df_out[image_col].map(lambda n: name_to_pred.get(str(n), 0)).astype(int)

    os.makedirs(os.path.dirname(OUT_CSV) or ".", exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"[Stage1] Saved: {OUT_CSV}")
    print(f"[Stage1] threshold={threshold:.6f} | ckpt={CKPT_PATH}")


if __name__ == "__main__":
    main()
