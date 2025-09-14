# evaluation/metrics.py
# ------------------------------------------------------------
# Metric robust cho hồi quy + phân loại (từ hồi quy).
# Giữ nguyên so với bản tốt gần nhất.
# ------------------------------------------------------------
from __future__ import annotations
import numpy as np

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False


def _to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(float).reshape(-1)


def smape_ratio(y_true, y_pred, eps: float = 1e-9) -> float:
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def robust_r2(y_true, y_pred, eps: float = 1e-12) -> float:
    y_true = _to_numpy(y_true); y_pred = _to_numpy(y_pred)
    y_bar = y_true.mean()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_bar) ** 2)
    if ss_tot < eps:
        return 0.0
    return float(1.0 - (ss_res / (ss_tot + eps)))


def trimmed_r2(y_true, y_pred, trim: float = 0.01) -> float:
    y_true = _to_numpy(y_true); y_pred = _to_numpy(y_pred)
    err = np.abs(y_true - y_pred)
    if y_true.size == 0:
        return 0.0
    thr = np.quantile(err, 1.0 - trim)
    keep = err <= thr
    if keep.sum() < 2:
        return robust_r2(y_true, y_pred)
    return robust_r2(y_true[keep], y_pred[keep])


def evaluate_regression(y_pred, y_true) -> dict:
    y_true = _to_numpy(y_true); y_pred = _to_numpy(y_pred)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    smape = smape_ratio(y_true, y_pred)
    r2 = robust_r2(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "SMAPE": smape, "R2": r2}


def classification_metrics_from_regression(
    y_true_real,
    y_score,
    pos_threshold: float | None = None,
    q: float = 0.90,
) -> dict:
    y_true_real = _to_numpy(y_true_real)
    y_score = _to_numpy(y_score)

    if pos_threshold is None:
        pos_threshold = float(np.quantile(y_true_real, q))
    y_bin = (y_true_real >= pos_threshold).astype(int)

    pos_rate = float(y_bin.mean()) if y_bin.size else 0.0
    if (y_bin.sum() == 0) or (y_bin.sum() == y_bin.size) or not _HAS_SK:
        return {"ROC_AUC": 0.5, "PR_AUC": pos_rate, "pos_rate": pos_rate, "threshold": pos_threshold}

    roc = float(roc_auc_score(y_bin, y_score))
    pr = float(average_precision_score(y_bin, y_score))
    return {"ROC_AUC": roc, "PR_AUC": pr, "pos_rate": pos_rate, "threshold": pos_threshold}
