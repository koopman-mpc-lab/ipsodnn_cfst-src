from __future__ import annotations

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def evaluate_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(target_names):
        true_col = y_true[:, index]
        pred_col = y_pred[:, index]
        metrics[name] = {
            "R2": r2_score(true_col, pred_col),
            "RMSE": rmse(true_col, pred_col),
            "MAE": mae(true_col, pred_col),
        }
    metrics["average"] = {
        "R2": float(np.mean([metrics[name]["R2"] for name in target_names])),
        "RMSE": float(np.mean([metrics[name]["RMSE"] for name in target_names])),
        "MAE": float(np.mean([metrics[name]["MAE"] for name in target_names])),
    }
    return metrics
