import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SewerMLDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str, labels: list[str], transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform

        # First column is image name by your description
        self.image_col = self.df.columns[0]
        for lab in labels:
            if lab not in self.df.columns:
                raise ValueError(f"Missing label column '{lab}' in {csv_path}")

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
