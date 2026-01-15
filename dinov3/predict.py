import os
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from dataset import SewerMLDataset
from metrics import sigmoid
from train import SimpleTransform  # reuse same preprocessing
from model import DinoV3MultiLabel

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="src/config.yaml")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    labels = cfg["labels"]

    ckpt = torch.load(args.ckpt, map_location="cpu")
    thresholds = ckpt.get("thresholds", np.array([0.5] * len(labels), dtype=np.float32))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tf = SimpleTransform(cfg["img_size"], train=False)
    ds = SewerMLDataset(args.csv, args.images, labels, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    model = DinoV3MultiLabel(cfg["model_name"], num_classes=len(labels), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    image_names = []
    probs_all = []

    for names, x, _ in tqdm(dl, desc="Predict"):
        x = x.to(device, non_blocking=True)
        logits = model(x).float().cpu().numpy()
        probs = sigmoid(logits)
        image_names.extend(list(names))
        probs_all.append(probs)

    probs_all = np.concatenate(probs_all, axis=0)
    preds = (probs_all >= thresholds[None, :]).astype(np.int32)

    # Output format: first col image_name, then label columns (0/1)
    out = pd.DataFrame({"image_name": image_names})
    for i, lab in enumerate(labels):
        out[lab] = preds[:, i]
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
