import numpy as np
from sklearn.metrics import f1_score

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def search_thresholds(
    logits: np.ndarray,
    targets: np.ndarray,
    strategy: str = "per_class",
    steps: int = 200,
):
    """
    Finds thresholds to maximize macro F1 on the provided validation set.
    - logits: [N, C]
    - targets: [N, C] in {0,1}
    """
    probs = sigmoid(logits)
    n, c = probs.shape

    if strategy == "global":
        best_t, best_f1 = 0.5, -1.0
        for i in range(steps + 1):
            t = i / steps
            preds = (probs >= t).astype(np.int32)
            f1 = f1_score(targets, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return np.array([best_t] * c, dtype=np.float32), best_f1

    if strategy == "per_class":
        th = np.full((c,), 0.5, dtype=np.float32)
        # Greedy per-class max F1 (macro proxy)
        # For stable behavior, maximize each class F1 independently.
        for j in range(c):
            best_t, best_f1c = 0.5, -1.0
            pj = probs[:, j]
            yj = targets[:, j]
            for i in range(steps + 1):
                t = i / steps
                predj = (pj >= t).astype(np.int32)
                f1c = f1_score(yj, predj, average="binary", zero_division=0)
                if f1c > best_f1c:
                    best_f1c, best_t = f1c, t
            th[j] = best_t

        preds = (probs >= th[None, :]).astype(np.int32)
        macro = f1_score(targets, preds, average="macro", zero_division=0)
        micro = f1_score(targets, preds, average="micro", zero_division=0)
        return th, macro, micro

    raise ValueError(f"Unknown strategy: {strategy}")

def f1_from_thresholds(logits: np.ndarray, targets: np.ndarray, thresholds: np.ndarray):
    probs = sigmoid(logits)
    preds = (probs >= thresholds[None, :]).astype(np.int32)
    macro = f1_score(targets, preds, average="macro", zero_division=0)
    micro = f1_score(targets, preds, average="micro", zero_division=0)
    return macro, micro
