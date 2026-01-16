import os
import math
import random
import shutil
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Works for both:
      - binary: logits [N,1], targets [N] or [N,1]
      - multilabel: logits [N,C], targets [N,C]
    Returns numpy arrays.
    """
    model.eval()
    all_logits = []
    all_targets = []
    for _, x, y in tqdm(loader, desc="Eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(y.float().cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def cleanup_checkpoints(out_dir: str, keep: int = 5, prefix: str = "epoch", suffix: str = ".pt"):
    files = [f for f in os.listdir(out_dir) if f.endswith(suffix) and f.startswith(prefix)]
    if len(files) <= keep:
        return

    # Keep most recent N by filename sort (simple)
    files.sort()
    to_delete = files[:-keep]
    for f in to_delete:
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass


def save_checkpoint_binary(
        out_dir: str,
        epoch: int,
        model: nn.Module,
        cfg: dict,
        labels: List[str],
        threshold: float,
        f1: float,
        acc: float,
        best: bool = False,
) -> str:
    fname = f"epoch{epoch:02d}_f1_{f1:.5f}_acc_{acc:.5f}.pt"
    path = os.path.join(out_dir, fname)

    payload = {
        "epoch": epoch,
        "model_name": cfg["model_name"],
        "labels": labels,
        "img_size": cfg["img_size"],
        "state_dict": model.state_dict(),
        "threshold": float(threshold),
        "f1": float(f1),
        "acc": float(acc),
        "config": cfg,
    }
    torch.save(payload, path)

    if best:
        best_path = os.path.join(out_dir, "best.pt")
        shutil.copyfile(path, best_path)

    return path


def save_checkpoint_multilabel(
        out_dir: str,
        epoch: int,
        model: nn.Module,
        cfg: dict,
        labels: List[str],
        thresholds: np.ndarray,
        macro_f1: float,
        micro_f1: float,
        best: bool = False,
) -> str:
    fname = f"epoch{epoch:02d}_macroF1_{macro_f1:.5f}_microF1_{micro_f1:.5f}.pt"
    path = os.path.join(out_dir, fname)

    payload = {
        "epoch": epoch,
        "model_name": cfg["model_name"],
        "labels": labels,
        "img_size": cfg["img_size"],
        "state_dict": model.state_dict(),
        "thresholds": thresholds,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "config": cfg,
    }
    torch.save(payload, path)

    if best:
        best_path = os.path.join(out_dir, "best.pt")
        shutil.copyfile(path, best_path)

    return path


# Transform
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


class SimpleTransform:
    def __init__(self, img_size: int, train: bool, rotate_degrees=(0, 90, 180), SEWER_MEAN=None,
                 SEWER_STD=None):
        if SEWER_MEAN is None:
            SEWER_MEAN = [0.523, 0.453, 0.345]
        if SEWER_STD is None:
            SEWER_STD = [0.210, 0.199, 0.154]
        self.img_size = img_size
        self.train = train
        self.rotate_degrees = rotate_degrees
        self.mean = torch.tensor(SEWER_MEAN).view(3, 1, 1)
        self.std = torch.tensor(SEWER_STD).view(3, 1, 1)

    def __call__(self, image):
        x = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        x = F.resize(x, [self.img_size, self.img_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if self.train:
            deg = random.choice(self.rotate_degrees)
            if deg != 0:
                x = F.rotate(x, deg, interpolation=InterpolationMode.BILINEAR, expand=False)
        x = (x - self.mean) / self.std
        return {"image": x}
