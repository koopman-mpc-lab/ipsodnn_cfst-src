from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _style_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(direction="in", top=True, right=True)


def plot_convergence(history: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(history, color="#2166ac", linewidth=1.0)
    ax.set(xlabel="Iteration", ylabel="Best fitness (validation MSE)", title="IPSO convergence")
    _style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, index, name in zip(axes, range(3), target_names):
        actual = y_true[:, index]
        predicted = y_pred[:, index]
        ax.scatter(actual, predicted, s=18, color="#2166ac", alpha=0.85, edgecolors="white", linewidth=0.3)
        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], color="0.2", linewidth=0.8)
        ax.set(xlabel=f"Experimental {name}", ylabel=f"Predicted {name}", title=name)
        _style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance: dict[str, float], output_path: Path) -> None:
    labels = list(importance.keys())
    values = np.array([importance[label] for label in labels])
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.barh(np.arange(len(labels)), values[order], color="#9ecae1", edgecolor="#2166ac")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([labels[i] for i in order])
    ax.set_xlabel("Permutation importance")
    _style_axes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
