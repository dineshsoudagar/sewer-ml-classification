import numpy as np


# -------------------------
# Numerically-stable sigmoid
# -------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid to avoid overflow
    x = x.astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


# -------------------------
# Confusion counts + metrics
# -------------------------
def _binary_counts(y_true: np.ndarray, y_pred: np.ndarray):
    # y_true/y_pred are {0,1} int arrays of same shape
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return tp, fp, fn, tn


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else 0.0


def _binary_precision_recall_f1_from_counts(tp: int, fp: int, fn: int):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _binary_accuracy_from_counts(tp: int, fp: int, fn: int, tn: int):
    return _safe_div(tp + tn, tp + fp + fn + tn)


# --------------------------------
# Stable BCEWithLogits (vectorized)
# --------------------------------
def bce_with_logits_np(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    Stable BCEWithLogits in numpy.
    logits, targets: [N] or [N,1] or [N,C]
    Returns scalar mean loss.
    loss = max(x,0) - x*y + log(1 + exp(-abs(x)))
    """
    x = logits.astype(np.float64, copy=False)
    y = targets.astype(np.float64, copy=False)
    loss = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
    return float(loss.mean())


# ----------------------------------------
# Binary metrics for stage-1 (ND gate)
# ----------------------------------------
def binary_metrics_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    logits: [N] or [N,1]
    targets: [N] or [N,1] in {0,1}
    Returns: dict with bce, acc, f1, precision, recall
    """
    logits = logits.reshape(-1)
    y_true = targets.reshape(-1).astype(np.int32, copy=False)

    probs = sigmoid(logits)
    y_pred = (probs >= threshold).astype(np.int32)

    tp, fp, fn, tn = _binary_counts(y_true, y_pred)
    precision, recall, f1 = _binary_precision_recall_f1_from_counts(tp, fp, fn)
    acc = _binary_accuracy_from_counts(tp, fp, fn, tn)

    return {
        "bce": bce_with_logits_np(logits, y_true),
        "acc": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,  # helpful for debugging
    }


def binary_search_threshold_for_f1(
    logits: np.ndarray,
    targets: np.ndarray,
    steps: int = 200,
) -> tuple[float, float]:
    """
    Sweep threshold in [0,1] to maximize binary F1.
    Returns (best_threshold, best_f1)
    """
    logits = logits.reshape(-1)
    y_true = targets.reshape(-1).astype(np.int32, copy=False)
    probs = sigmoid(logits)

    best_t, best_f1 = 0.5, -1.0
    for i in range(steps + 1):
        t = i / steps
        y_pred = (probs >= t).astype(np.int32)
        tp, fp, fn, tn = _binary_counts(y_true, y_pred)
        _, _, f1 = _binary_precision_recall_f1_from_counts(tp, fp, fn)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    return float(best_t), float(best_f1)


# ----------------------------------------
# Multi-label thresholding (no sklearn)
# ----------------------------------------
def _macro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    y_true/y_pred: [N,C] in {0,1}
    macro F1 = mean over classes of binary F1.
    """
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)

    # per-class counts
    tp = (y_true & y_pred).sum(axis=0).astype(np.float64)
    fp = ((1 - y_true) & y_pred).sum(axis=0).astype(np.float64)
    fn = (y_true & (1 - y_pred)).sum(axis=0).astype(np.float64)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    return float(f1.mean())


def _micro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    micro F1 = global TP/FP/FN over all classes and samples.
    """
    y_true = y_true.astype(np.int32, copy=False)
    y_pred = y_pred.astype(np.int32, copy=False)

    tp = int((y_true & y_pred).sum())
    fp = int(((1 - y_true) & y_pred).sum())
    fn = int((y_true & (1 - y_pred)).sum())

    precision, recall, f1 = _binary_precision_recall_f1_from_counts(tp, fp, fn)
    return float(f1)


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
    Returns:
      - global: (thresholds[C], best_macro_f1)
      - per_class: (thresholds[C], macro_f1, micro_f1)
    """
    probs = sigmoid(logits)
    targets = targets.astype(np.int32, copy=False)
    n, c = probs.shape

    if strategy == "global":
        best_t, best_macro = 0.5, -1.0
        for i in range(steps + 1):
            t = i / steps
            preds = (probs >= t).astype(np.int32)
            macro = _macro_f1_from_preds(targets, preds)
            if macro > best_macro:
                best_macro, best_t = macro, t
        return np.array([best_t] * c, dtype=np.float32), float(best_macro)

    if strategy == "per_class":
        th = np.full((c,), 0.5, dtype=np.float32)

        # Optimize each class threshold for that class' F1
        for j in range(c):
            pj = probs[:, j]
            yj = targets[:, j]

            best_t, best_f1c = 0.5, -1.0
            for i in range(steps + 1):
                t = i / steps
                predj = (pj >= t).astype(np.int32)
                tp, fp, fn, tn = _binary_counts(yj, predj)
                _, _, f1c = _binary_precision_recall_f1_from_counts(tp, fp, fn)
                if f1c > best_f1c:
                    best_f1c, best_t = f1c, t
            th[j] = best_t

        preds = (probs >= th[None, :]).astype(np.int32)
        macro = _macro_f1_from_preds(targets, preds)
        micro = _micro_f1_from_preds(targets, preds)
        return th.astype(np.float32), float(macro), float(micro)

    raise ValueError(f"Unknown strategy: {strategy}")


def f1_from_thresholds(logits: np.ndarray, targets: np.ndarray, thresholds: np.ndarray):
    probs = sigmoid(logits)
    targets = targets.astype(np.int32, copy=False)
    preds = (probs >= thresholds[None, :]).astype(np.int32)
    macro = _macro_f1_from_preds(targets, preds)
    micro = _micro_f1_from_preds(targets, preds)
    return float(macro), float(micro)
