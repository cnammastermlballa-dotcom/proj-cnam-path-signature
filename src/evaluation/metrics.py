"""
Evaluation metrics and visualisations — ECG Path Signature.

Available functions:
  - compute_test_metrics()       : accuracy, F1, ROC-AUC on the test set
  - print_cv_table()             : comparative table of CV results
  - plot_performance_vs_level()  : metric vs truncation level curve
  - plot_confusion_matrices()    : side-by-side confusion matrices
  - plot_roc_curves()            : overlaid ROC curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_test_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """
    Compute accuracy, F1 and ROC-AUC on the test set.

    Returns
    -------
    dict with keys: accuracy, f1, roc_auc
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred),
        "roc_auc":  roc_auc_score(y_true, y_pred_proba),
    }


# ---------------------------------------------------------------------------
# Comparative table
# ---------------------------------------------------------------------------

def print_cv_table(results: list[dict]) -> pd.DataFrame:
    """
    Display and return a comparative DataFrame of cross-validation results.

    Parameters
    ----------
    results : list[dict] — each dict must have the keys returned by
              cross_val_evaluate() from classifiers.py

    Returns
    -------
    pd.DataFrame with one row per model
    """
    rows = []
    for r in results:
        rows.append({
            "Model":      r["model_name"],
            "# Features": r["n_features"],
            "Accuracy":   f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
            "F1":         f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
            "ROC-AUC":    f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",
        })
    df = pd.DataFrame(rows)

    print("\n" + "=" * 85)
    print("CROSS-VALIDATION RESULTS — 5-fold stratified, ECG200 train set")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85 + "\n")
    return df


# ---------------------------------------------------------------------------
# Performance vs truncation level curve
# ---------------------------------------------------------------------------

def plot_performance_vs_level(
    sig_results: list[dict],
    baseline_result: dict,
    metric: str = "accuracy_mean",
    save_path: Path = None,
) -> None:
    """
    Plot the chosen metric as a function of truncation level.

    The baseline is shown as a horizontal red dashed line for comparison.
    The ±1 std band around the signature curve is shaded.

    Parameters
    ----------
    sig_results     : list[dict] — results for each level, each dict
                      must have the keys from cross_val_evaluate() + "level"
    baseline_result : dict — cross_val_evaluate() results for the baseline
    metric          : str — key to plot, e.g. "accuracy_mean", "f1_mean",
                      "roc_auc_mean"
    save_path       : Path or None — if provided, saves the figure
    """
    levels     = [r["level"] for r in sig_results]
    means      = [r[metric] for r in sig_results]
    std_key    = metric.replace("_mean", "_std")
    stds       = [r[std_key] for r in sig_results]
    baseline   = baseline_result[metric]
    metric_lbl = metric.replace("_mean", "").replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        levels, means, yerr=stds,
        marker="o", linewidth=2, capsize=5,
        label="Signature (time embedding)", color="steelblue",
    )
    ax.fill_between(
        levels,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.15, color="steelblue",
    )
    ax.axhline(
        baseline, color="tomato", linestyle="--", linewidth=1.8,
        label=f"Statistical baseline ({baseline:.4f})",
    )
    ax.set_xlabel("Truncation level N", fontsize=12)
    ax.set_ylabel(metric_lbl, fontsize=12)
    ax.set_title(
        f"{metric_lbl} vs Truncation Level — ECG200 (5-fold CV)",
        fontsize=13,
    )
    ax.set_xticks(levels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Confusion matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    results: list[dict],
    save_path: Path = None,
) -> None:
    """
    Display side-by-side confusion matrices (test set).

    Parameters
    ----------
    results   : list[dict] — each dict must have:
                  'model_name', 'y_true', 'y_pred'
    save_path : Path or None
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        cm  = confusion_matrix(r["y_true"], r["y_pred"])
        acc = accuracy_score(r["y_true"], r["y_pred"])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
        )
        ax.set_title(f"{r['model_name']}\nAccuracy={acc:.4f}", fontsize=11)

    plt.suptitle("Confusion matrices — ECG200 (test set)", fontsize=13, y=1.03)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    results: list[dict],
    save_path: Path = None,
) -> None:
    """
    Overlay ROC curves for multiple models (test set).

    Parameters
    ----------
    results   : list[dict] — each dict must have:
                  'model_name', 'y_true', 'y_pred_proba', 'roc_auc'
    save_path : Path or None
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10.colors

    for i, r in enumerate(results):
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_pred_proba"])
        ax.plot(
            fpr, tpr,
            label=f"{r['model_name']} (AUC={r['roc_auc']:.3f})",
            color=colors[i % len(colors)],
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("ROC Curves — ECG200 (test set)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()
