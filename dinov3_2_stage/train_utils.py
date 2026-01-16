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

# Transform
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


# -------------------------
# Reproducibility
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# LR schedule
# -------------------------
def cosine_warmup_lr(step: int, total_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# -------------------------
# Eval (returns numpy arrays; matches your metrics.py)
# -------------------------
@torch.no_grad()
def run_eval(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
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


# -------------------------
# Checkpoint housekeeping
# -------------------------
def cleanup_checkpoints(out_dir: str, keep: int = 5, prefix: str = "epoch", suffix: str = ".pt"):
    files = [f for f in os.listdir(out_dir) if f.endswith(suffix) and f.startswith(prefix)]
    if len(files) <= keep:
        return
    files.sort()  # simple: keep most recent by filename
    for f in files[:-keep]:
        try:
            os.remove(os.path.join(out_dir, f))
        except OSError:
            pass


# -------------------------
# Resume helpers
# -------------------------
def maybe_resume(
    resume_ckpt: Optional[str],
    model: nn.Module,
    optimizer,
    scaler,
    device: str,
) -> Tuple[int, int, float, int]:
    """
    Returns: start_epoch, global_step, best_score, bad_epochs
      - best_score is best_f1 for stage1, or best monitored metric for stage2
    """
    if not resume_ckpt:
        return 1, 0, -1.0, 0

    ckpt = torch.load(resume_ckpt, map_location=device)

    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    global_step = int(ckpt.get("global_step", 0))
    best_score = float(ckpt.get("best_score", -1.0))
    bad_epochs = int(ckpt.get("bad_epochs", 0))

    print(f"[Resume] Loaded: {resume_ckpt}")
    print(f"[Resume] start_epoch={start_epoch}, global_step={global_step}, best_score={best_score:.5f}, bad_epochs={bad_epochs}")
    return start_epoch, global_step, best_score, bad_epochs


# -------------------------
# Save checkpoints (Stage1 / Stage2)
# -------------------------
def save_checkpoint_binary(
    out_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    model_name: str,
    img_size: int,
    labels: List[str],
    threshold: float,
    f1: float,
    acc: float,
    best: bool,
    global_step: int,
    best_score: float,
    bad_epochs: int,
) -> str:
    fname = f"epoch{epoch:02d}_f1_{f1:.5f}_acc_{acc:.5f}.pt"
    path = os.path.join(out_dir, fname)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_score": float(best_score),
        "bad_epochs": int(bad_epochs),

        "model_name": model_name,
        "labels": labels,
        "img_size": int(img_size),

        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,

        "threshold": float(threshold),
        "f1": float(f1),
        "acc": float(acc),
    }

    os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, path)

    if best:
        shutil.copyfile(path, os.path.join(out_dir, "best.pt"))

    return path


def save_checkpoint_multilabel(
    out_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer,
    scaler,
    model_name: str,
    img_size: int,
    labels: List[str],
    thresholds: np.ndarray,
    macro_f1: float,
    micro_f1: float,
    best: bool,
    global_step: int,
    best_score: float,
    bad_epochs: int,
) -> str:
    fname = f"epoch{epoch:02d}_macroF1_{macro_f1:.5f}_microF1_{micro_f1:.5f}.pt"
    path = os.path.join(out_dir, fname)

    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_score": float(best_score),
        "bad_epochs": int(bad_epochs),

        "model_name": model_name,
        "labels": labels,
        "img_size": int(img_size),

        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,

        "thresholds": thresholds.astype(np.float32),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }

    os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, path)

    if best:
        shutil.copyfile(path, os.path.join(out_dir, "best.pt"))

    return path


# -------------------------
# Transform (unchanged behavior)
# -------------------------
class SimpleTransform:
    def __init__(self, img_size: int, train: bool, rotate_degrees=(0, 90, 180), SEWER_MEAN=None, SEWER_STD=None):
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
