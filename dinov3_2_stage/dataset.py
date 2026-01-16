import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SewerMLDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        labels: list[str],
        transform=None,
        defect_only: bool = False,  # True => gate (ND only, keep all rows). False => stage-2 (exclude ND rows + ND column)
        nd_label: str = "ND",
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform
        self.defect_only = defect_only
        self.nd_label = nd_label

        # First column is image name
        self.image_col = self.df.columns[0]

        for lab in labels:
            if lab not in self.df.columns:
                raise ValueError(f"Missing label column '{lab}' in {csv_path}")

        if nd_label not in self.df.columns:
            raise ValueError(f"Missing ND label column '{nd_label}' in {csv_path}")

        # If defect_only is False => stage-2 training: exclude ND==1 rows + drop ND from labels
        if not self.defect_only:
            self.df = self.df[self.df[self.nd_label] == 0].reset_index(drop=True)
            self.labels = [l for l in self.labels if l != self.nd_label]
        else:
            # defect_only True => gate training: keep all rows, target is ONLY ND column
            self.labels = [self.nd_label]

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

        # If gate mode (ND only), return scalar instead of shape (1,)
        if self.defect_only:
            y = y.squeeze(0)

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        return img_name, img, y


if __name__ == "__main__":
    # ---- set these paths ----
    CSV_PATH = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
    IMAGES_DIR = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
    ND_LABEL = "ND"

    # Infer labels from CSV header: all columns except first (Filename)
    df_head = pd.read_csv(CSV_PATH, nrows=1)
    image_col = df_head.columns[0]
    LABELS = [c for c in df_head.columns if c != image_col]

    # ---- TEST 1: gate mode (ND only) ----
    ds_gate = SewerMLDataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        labels=LABELS,
        transform=None,
        defect_only=True,
        nd_label=ND_LABEL,
    )

    print("=" * 80)
    print("TEST 1: gate mode (ND only, all rows)")
    print("len(ds_gate) =", len(ds_gate))
    print("target labels =", ds_gate.labels)
    for i in range(3):
        name, img, y = ds_gate[i]
        print(f"[{i}] {name} | img={img.shape} | ND={y.item()}")

    # ---- TEST 2: stage-2 mode (ND==0 only, ND removed from y) ----
    ds_def = SewerMLDataset(
        csv_path=CSV_PATH,
        images_dir=IMAGES_DIR,
        labels=LABELS,
        transform=None,
        defect_only=False,
        nd_label=ND_LABEL,
    )

    print("=" * 80)
    print("TEST 2: stage-2 mode (ND==0 rows only, ND excluded from y)")
    print("len(ds_def) =", len(ds_def))
    print("target labels =", ds_def.labels)
    for i in range(3):
        name, img, y = ds_def[i]
        print(f"[{i}] {name} | img={img.shape} | y_shape={tuple(y.shape)} | y_sum={float(y.sum().item()):.0f}")
    print("=" * 80)
