import numpy as np


# -------------------------
# Numerically-stable sigmoid
# -------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


# -------------------------
# Stable BCEWithLogits (numpy)
# -------------------------
def bce_with_logits_np(logits: np.ndarray, targets: np.ndarray) -> float:
    """
    logits, targets can be shape [N] or [N,1] or [N,C].
    Returns scalar mean loss.
    loss = max(x,0) - x*y + log(1 + exp(-abs(x)))
    """
    x = logits.astype(np.float64)
    y = targets.astype(np.float64)
    loss = np.maximum(x, 0.0) - x * y + np.log1p(np.exp(-np.abs(x)))
    return float(loss.mean())


# -------------------------
# Binary metrics helpers (numpy)
# -------------------------
def _binary_counts(pred: np.ndarray, y: np.ndarray):
    # pred,y: [N] int {0,1}
    pred = pred.astype(np.int32)
    y = y.astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def _binary_metrics_from_counts(tp: int, fp: int, fn: int, tn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    return float(acc), float(f1), float(precision), float(recall)


# -------------------------
# Exact best threshold for binary F1 (no grid)
# -------------------------
def _best_threshold_f1_exact_1d(probs: np.ndarray, y: np.ndarray):
    """
    probs: [N] float in [0,1]
    y:     [N] int {0,1}
    Returns: (best_threshold, best_f1)
    """
    probs = probs.astype(np.float32).reshape(-1)
    y = y.astype(np.int32).reshape(-1)

    pos_total = int(y.sum())
    if pos_total == 0:
        # no positives => any threshold gives F1=0; choose 1.0 (predict none)
        return 1.0, 0.0

    # sort probs desc
    order = np.argsort(probs)[::-1]
    p_sorted = probs[order]
    y_sorted = y[order]

    # If we predict top-k as positive:
    tp = np.cumsum(y_sorted).astype(np.float64)
    k = np.arange(1, len(y_sorted) + 1, dtype=np.float64)
    fp = k - tp
    fn = float(pos_total) - tp

    denom = 2.0 * tp + fp + fn
    f1 = np.where(denom > 0, (2.0 * tp) / denom, 0.0)

    best_idx = int(np.argmax(f1))
    best_f1 = float(f1[best_idx])

    # threshold that reproduces this cutoff:
    # choose mid-point between p_sorted[best_idx] and next prob if possible.
    if best_idx < len(p_sorted) - 1 and p_sorted[best_idx] != p_sorted[best_idx + 1]:
        thr = 0.5 * (float(p_sorted[best_idx]) + float(p_sorted[best_idx + 1]))
    else:
        thr = float(p_sorted[best_idx])

    # clamp for safety
    thr = float(np.clip(thr, 0.0, 1.0))
    return thr, best_f1


# -------------------------
# Multi-label F1 (macro/micro) from preds/targets (numpy)
# -------------------------
def _f1_macro_micro_from_preds(preds: np.ndarray, targets: np.ndarray):
    """
    preds, targets: [N,C] int {0,1}
    Returns (macro_f1, micro_f1)
    """
    preds = preds.astype(np.int32)
    targets = targets.astype(np.int32)

    tp = (preds & targets).sum(axis=0).astype(np.float64)
    fp = (preds & (1 - targets)).sum(axis=0).astype(np.float64)
    fn = ((1 - preds) & targets).sum(axis=0).astype(np.float64)

    denom = 2.0 * tp + fp + fn
    f1_per_class = np.where(denom > 0, (2.0 * tp) / denom, 0.0)
    macro = float(f1_per_class.mean())

    TP = float(tp.sum())
    FP = float(fp.sum())
    FN = float(fn.sum())
    denom_micro = 2.0 * TP + FP + FN
    micro = float((2.0 * TP) / denom_micro) if denom_micro > 0 else 0.0

    return macro, micro


# -------------------------
# Public API: search_thresholds
# -------------------------
def search_thresholds(
    logits: np.ndarray,
    targets: np.ndarray,
    strategy: str = "per_class",
    steps: int = 200,
):
    """
    - logits: [N,C]
    - targets: [N,C] in {0,1}
    Returns:
      strategy="global"   -> (thresholds[C], best_macro_f1)
      strategy="per_class"-> (thresholds[C], macro_f1, micro_f1)
    """
    probs = sigmoid(logits)
    targets = targets.astype(np.int32)
    n, c = probs.shape

    if strategy == "global":
        # grid search global threshold; return best macro F1 (matches your old behavior)
        best_t = 0.5
        best_macro = -1.0

        # (optional) track micro too internally if you want later
        for i in range(steps + 1):
            t = i / steps
            preds = (probs >= t).astype(np.int32)
            macro, _micro = _f1_macro_micro_from_preds(preds, targets)
            if macro > best_macro:
                best_macro = macro
                best_t = t

        return np.array([best_t] * c, dtype=np.float32), float(best_macro)

    if strategy == "per_class":
        # exact per-class best threshold for binary F1 (no grid snapping)
        th = np.zeros((c,), dtype=np.float32)
        for j in range(c):
            tj, _ = _best_threshold_f1_exact_1d(probs[:, j], targets[:, j])
            th[j] = tj

        preds = (probs >= th[None, :]).astype(np.int32)
        macro, micro = _f1_macro_micro_from_preds(preds, targets)
        return th, float(macro), float(micro)

    raise ValueError(f"Unknown strategy: {strategy}")


def f1_from_thresholds(logits: np.ndarray, targets: np.ndarray, thresholds: np.ndarray):
    probs = sigmoid(logits)
    targets = targets.astype(np.int32)
    thresholds = thresholds.astype(np.float32).reshape(-1)
    preds = (probs >= thresholds[None, :]).astype(np.int32)
    macro, micro = _f1_macro_micro_from_preds(preds, targets)
    return float(macro), float(micro)


# -------------------------
# Stage-1 binary metrics (numpy-only)
# -------------------------
def binary_metrics_from_logits(
    logits: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    logits: [N] or [N,1]
    targets: [N] or [N,1] in {0,1}
    Returns: dict with bce, acc, f1, precision, recall, threshold
    """
    logits = logits.reshape(-1).astype(np.float32)
    y = targets.reshape(-1).astype(np.int32)

    probs = sigmoid(logits)
    pred = (probs >= float(threshold)).astype(np.int32)

    tp, fp, fn, tn = _binary_counts(pred, y)
    acc, f1, precision, recall = _binary_metrics_from_counts(tp, fp, fn, tn)

    return {
        "bce": bce_with_logits_np(logits, y),
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": float(threshold),
    }


def binary_search_threshold_for_f1(
    logits: np.ndarray,
    targets: np.ndarray,
    steps: int = 200,
) -> tuple[float, float]:
    """
    Keeps your old API. Uses EXACT search (better than grid).
    'steps' kept for compatibility but unused.
    """
    logits = logits.reshape(-1).astype(np.float32)
    y = targets.reshape(-1).astype(np.int32)
    probs = sigmoid(logits)
    best_t, best_f1 = _best_threshold_f1_exact_1d(probs, y)
    return float(best_t), float(best_f1)
