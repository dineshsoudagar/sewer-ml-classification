import os
import json
import re
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SewerMLDataset
from model import DinoV3MultiLabel
from train_utils import SimpleTransform, run_eval
from metrics import search_thresholds, f1_from_thresholds

TRAIN_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\train.csv"
VAL_CSV = r"D:\expandAI-hiring\expandai-hiring-sewer\SewerML_Val_jpg.csv"
TRAIN_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\train_images"
VAL_IMAGES = r"D:\expandAI-hiring\expandai-hiring-sewer\test_images"

OUT_DIR = "outputs_stage2_vit_base"

MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"

LABELS = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS", "OP", "OK", "VA",
          "ND"]
ND_LABEL = "ND"
LABELS_WO_ND = [l for l in LABELS if l != ND_LABEL]

IMG_SIZE = 256
BATCH_SIZE = 64
NUM_WORKERS = 8

THRESHOLD_STEPS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Select "best" using per-class thresholds
MONITOR = "macro_f1"  # "macro_f1" or "micro_f1"

# Backtrack stopping rule
PATIENCE_BACKTRACK = 2  # stop after this many consecutive non-improvements
MIN_DELTA = 0.0  # require at least this improvement to reset patience


# =========================


def _epoch_num_from_name(fname: str) -> int:
    m = re.search(r"epoch(\d+)", fname)
    return int(m.group(1)) if m else -1


def load_model_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")  # trusted local checkpoint
    model = DinoV3MultiLabel(MODEL_NAME, num_classes=len(LABELS_WO_ND), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def write_thresholds_artifacts(
        src_ckpt_path: str,
        labels_wo_nd: list[str],
        global_thresholds: np.ndarray,
        global_macro: float,
        global_micro: float,
        per_class_thresholds: np.ndarray,
        per_class_macro: float,
        per_class_micro: float,
):
    """
    Option-1: Do NOT overwrite. Write:
      - augmented checkpoint: <base>__with_thresholds.pt
      - JSON sidecar:         <base>__thresholds.json
    """
    ckpt = torch.load(src_ckpt_path, map_location="cpu")

    ckpt["thresholding"] = {
        "val_csv": VAL_CSV,
        "threshold_steps": int(THRESHOLD_STEPS),
        "labels": labels_wo_nd,
        "global": {
            "thresholds": global_thresholds.astype(np.float32),
            "macro_f1": float(global_macro),
            "micro_f1": float(global_micro),
        },
        "per_class": {
            "thresholds": per_class_thresholds.astype(np.float32),
            "macro_f1": float(per_class_macro),
            "micro_f1": float(per_class_micro),
        },
    }

    base = os.path.splitext(os.path.basename(src_ckpt_path))[0]
    aug_ckpt_name = f"{base}__with_thresholds.pt"
    aug_ckpt_path = os.path.join(os.path.dirname(src_ckpt_path), aug_ckpt_name)
    torch.save(ckpt, aug_ckpt_path)

    global_map = {lab: float(t) for lab, t in zip(labels_wo_nd, global_thresholds.reshape(-1).tolist())}
    per_class_map = {lab: float(t) for lab, t in zip(labels_wo_nd, per_class_thresholds.reshape(-1).tolist())}

    json_name = f"{base}__thresholds.json"
    json_path = os.path.join(os.path.dirname(src_ckpt_path), json_name)
    with open(json_path, "w") as f:
        json.dump(
            {
                "checkpoint": os.path.basename(src_ckpt_path),
                "val_csv": VAL_CSV,
                "threshold_steps": int(THRESHOLD_STEPS),
                "labels": labels_wo_nd,
                "global": {
                    "thresholds": global_map,
                    "macro_f1": float(global_macro),
                    "micro_f1": float(global_micro),
                    "threshold": float(global_thresholds.reshape(-1)[0]),
                },
                "per_class": {
                    "thresholds": per_class_map,
                    "macro_f1": float(per_class_macro),
                    "micro_f1": float(per_class_micro),
                },
            },
            f,
            indent=2,
        )

    return aug_ckpt_path, json_path


def _evaluate_and_cache(ckpt_path: str, val_loader: DataLoader):
    model = load_model_from_ckpt(ckpt_path)
    val_logits, val_targets = run_eval(model, val_loader, DEVICE)

    # GLOBAL thresholds
    global_thresholds, _ = search_thresholds(val_logits, val_targets, strategy="global", steps=THRESHOLD_STEPS)
    global_macro, global_micro = f1_from_thresholds(val_logits, val_targets, global_thresholds)

    # PER-CLASS thresholds
    per_class_thresholds, per_class_macro, per_class_micro = search_thresholds(
        val_logits, val_targets, strategy="per_class", steps=THRESHOLD_STEPS
    )

    aug_ckpt_path, json_path = write_thresholds_artifacts(
        ckpt_path,
        LABELS_WO_ND,
        global_thresholds,
        global_macro,
        global_micro,
        per_class_thresholds,
        per_class_macro,
        per_class_micro,
    )

    score = per_class_macro if MONITOR == "macro_f1" else per_class_micro

    print(
        f"{os.path.basename(ckpt_path)} | "
        f"GLOBAL macro={global_macro:.5f} micro={global_micro:.5f} thr={float(global_thresholds[0]):.3f} | "
        f"PER-CLASS macro={per_class_macro:.5f} micro={per_class_micro:.5f} | "
        f"monitor={score:.5f} | saved: {os.path.basename(aug_ckpt_path)}"
    )

    return {
        "ckpt": ckpt_path,
        "aug_ckpt": aug_ckpt_path,
        "json": json_path,
        "score": float(score),
        "global_macro": float(global_macro),
        "global_micro": float(global_micro),
        "global_thr": float(global_thresholds.reshape(-1)[0]),
        "per_class_macro": float(per_class_macro),
        "per_class_micro": float(per_class_micro),
    }


def main():
    val_tf = SimpleTransform(IMG_SIZE, train=False)
    val_ds = SewerMLDataset(VAL_CSV, VAL_IMAGES, LABELS, transform=val_tf, defect_only=False)
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    best_pt_path = os.path.join(OUT_DIR, "best.pt")
    if not os.path.exists(best_pt_path):
        raise RuntimeError(f"best.pt not found in {OUT_DIR}. Expected: {best_pt_path}")

    # Collect epoch checkpoints, sorted by epoch number
    epoch_ckpts = [
        os.path.join(OUT_DIR, f)
        for f in os.listdir(OUT_DIR)
        if f.startswith("epoch") and f.endswith(".pt") and "__with_thresholds" not in f
    ]
    if not epoch_ckpts:
        raise RuntimeError(f"No epoch checkpoints found in {OUT_DIR}")

    epoch_ckpts.sort(key=lambda p: _epoch_num_from_name(os.path.basename(p)))

    # Determine which epoch best.pt corresponds to (if present in payload)
    best_payload = torch.load(best_pt_path, map_location="cpu")
    best_epoch = int(best_payload.get("epoch", -1))

    # Fallback: if epoch not stored, we still evaluate best.pt then backtrack all epochs
    print("Evaluating best.pt first...")
    best_info = _evaluate_and_cache(best_pt_path, val_loader)

    best_score = best_info["score"]
    best_choice = best_info

    ranking = []
    ranking.append({
        "source": "best.pt",
        "ckpt": os.path.basename(best_info["ckpt"]),
        "monitor_score": best_info["score"],
        "global": {"macro_f1": best_info["global_macro"], "micro_f1": best_info["global_micro"],
                   "threshold": best_info["global_thr"]},
        "per_class": {"macro_f1": best_info["per_class_macro"], "micro_f1": best_info["per_class_micro"]},
        "augmented_ckpt": os.path.basename(best_info["aug_ckpt"]),
        "thresholds_json": os.path.basename(best_info["json"]),
    })

    # Build the backtrack list: epochs older than best_epoch (if known), else all epochs (newest->oldest)
    if best_epoch >= 0:
        older = [p for p in epoch_ckpts if _epoch_num_from_name(os.path.basename(p)) < best_epoch]
        ckpts_to_check = list(reversed(older))  # newest->oldest among older epochs
        print(f"best.pt epoch={best_epoch}. Backtracking through {len(ckpts_to_check)} older epoch checkpoints...")
    else:
        ckpts_to_check = list(reversed(epoch_ckpts))
        print(f"best.pt epoch unknown. Backtracking through {len(ckpts_to_check)} epoch checkpoints...")

    bad_streak = 0

    for ckpt_path in ckpts_to_check:
        info = _evaluate_and_cache(ckpt_path, val_loader)

        ranking.append({
            "source": "epoch",
            "ckpt": os.path.basename(info["ckpt"]),
            "epoch": _epoch_num_from_name(os.path.basename(info["ckpt"])),
            "monitor_score": info["score"],
            "global": {"macro_f1": info["global_macro"], "micro_f1": info["global_micro"],
                       "threshold": info["global_thr"]},
            "per_class": {"macro_f1": info["per_class_macro"], "micro_f1": info["per_class_micro"]},
            "augmented_ckpt": os.path.basename(info["aug_ckpt"]),
            "thresholds_json": os.path.basename(info["json"]),
        })

        if info["score"] > (best_score + MIN_DELTA):
            best_score = info["score"]
            best_choice = info
            bad_streak = 0
        else:
            bad_streak += 1

        if bad_streak >= PATIENCE_BACKTRACK:
            print(f"\nStopping backtrack early: {bad_streak} consecutive non-improvements.")
            break

    ranking_path = os.path.join(OUT_DIR, "checkpoint_ranking_with_thresholds__best_then_backtrack.json")
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2)

    print("\n================ BEST (by per-class thresholds) ================")
    print(f"Chosen ckpt:        {os.path.basename(best_choice['ckpt'])}")
    print(f"Augmented ckpt:     {os.path.basename(best_choice['aug_ckpt'])}")
    print(f"Thresholds JSON:    {os.path.basename(best_choice['json'])}")
    print(
        f"Best GLOBAL macro:  {best_choice['global_macro']:.5f}  micro: {best_choice['global_micro']:.5f}  thr: {best_choice['global_thr']:.3f}")
    print(f"Best PER-CLASS macro:{best_choice['per_class_macro']:.5f} micro: {best_choice['per_class_micro']:.5f}")
    print(f"Ranking saved:      {ranking_path}")

    # Convenience aliases for deployment
    alias_ckpt = os.path.join(OUT_DIR, "best__with_thresholds.pt")
    alias_json = os.path.join(OUT_DIR, "best__thresholds.json")

    with open(best_choice["aug_ckpt"], "rb") as rf, open(alias_ckpt, "wb") as wf:
        wf.write(rf.read())
    with open(best_choice["json"], "r", encoding="utf-8") as rf, open(alias_json, "w", encoding="utf-8") as wf:
        wf.write(rf.read())

    print("\nConvenience copies written:")
    print(f"  {alias_ckpt}")
    print(f"  {alias_json}")


if __name__ == "__main__":
    main()
