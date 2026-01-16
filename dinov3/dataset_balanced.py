import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_balanced_multilabel_subset(
    df: pd.DataFrame,
    image_col: str,
    labels: list[str],
    target_size: int = 5000,
    min_pos_per_class: int = 200,
    seed: int = 42,
):
    """
    Multi-label balancing heuristic:
    1) Greedily pick samples to reach min_pos_per_class positives for each class.
    2) If still below target_size, fill remaining with samples that add the most remaining positives,
       then random fill.

    Returns a filtered dataframe (subset).
    """

    rng = np.random.default_rng(seed)
    y = df[labels].to_numpy(dtype=np.int32)
    n, c = y.shape

    # Indices for each class positive
    pos_idx = [np.where(y[:, j] == 1)[0] for j in range(c)]

    # Shuffle indices per class to avoid bias
    for j in range(c):
        rng.shuffle(pos_idx[j])

    selected = set()
    pos_counts = np.zeros(c, dtype=np.int32)

    # --- Phase 1: satisfy min_pos_per_class ---
    # Iterate classes and add positives until each meets target.
    # Since multi-label, one sample can satisfy multiple classes.
    class_order = np.argsort([len(pos_idx[j]) for j in range(c)])  # start with rare classes

    for j in class_order:
        needed = max(0, min_pos_per_class - pos_counts[j])
        if needed == 0:
            continue
        # add up to needed samples from this class
        for idx in pos_idx[j]:
            if idx in selected:
                continue
            selected.add(idx)
            pos_counts += y[idx]
            needed = max(0, min_pos_per_class - pos_counts[j])
            if needed == 0:
                break

    # --- Phase 2: fill up to target_size prioritizing remaining positive coverage ---
    if len(selected) < target_size:
        remaining = np.array([i for i in range(n) if i not in selected], dtype=np.int32)
        rng.shuffle(remaining)

        # Score each candidate by how many "still-needed" positives it would add
        need_mask = (pos_counts < min_pos_per_class).astype(np.int32)

        # Greedy add candidates that add positives where still needed
        # Stop if no unmet needs or reach target_size.
        for idx in remaining:
            if len(selected) >= target_size:
                break
            if need_mask.sum() == 0:
                break

            gain = int((y[idx] * need_mask).sum())
            if gain > 0:
                selected.add(int(idx))
                pos_counts += y[idx]
                need_mask = (pos_counts < min_pos_per_class).astype(np.int32)

        # Random fill to reach target_size
        if len(selected) < target_size:
            remaining2 = np.array([i for i in range(n) if i not in selected], dtype=np.int32)
            rng.shuffle(remaining2)
            take = min(target_size - len(selected), len(remaining2))
            selected.update(map(int, remaining2[:take]))

    selected = sorted(list(selected))
    sub_df = df.iloc[selected].reset_index(drop=True)

    # Diagnostics
    sub_y = sub_df[labels].to_numpy(dtype=np.int32)
    pos_per_class = sub_y.sum(axis=0)
    any_pos = (sub_y.sum(axis=1) > 0).sum()

    print("Balanced subset built:")
    print(f"  target_size={target_size}, actual_size={len(sub_df)}")
    print(f"  rows_with_any_positive={any_pos}/{len(sub_df)}")
    print(f"  positives_per_class={pos_per_class.tolist()}")

    return sub_df


class SewerMLBalancedDataset(Dataset):
    """
    Like your SewerMLDataset but optionally builds a balanced subset at init.

    CSV schema:
      - first column: image name
      - remaining columns include 'labels' in {0,1}
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        labels: list[str],
        transform=None,
        # balancing params
        use_balanced_subset: bool = True,
        subset_size: int = 5000,
        min_pos_per_class: int = 200,
        seed: int = 42,
    ):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.labels = labels
        self.transform = transform

        self.image_col = self.df.columns[0]
        for lab in labels:
            if lab not in self.df.columns:
                raise ValueError(f"Missing label column '{lab}' in {csv_path}")

        if use_balanced_subset:
            self.df = build_balanced_multilabel_subset(
                df=self.df,
                image_col=self.image_col,
                labels=self.labels,
                target_size=subset_size,
                min_pos_per_class=min_pos_per_class,
                seed=seed,
            )

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
