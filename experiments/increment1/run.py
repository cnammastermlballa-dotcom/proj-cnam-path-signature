"""
Increment 1 — Statistical baseline vs Path Signature (time embedding)
======================================================================

Full pipeline:
  1. Load ECG200 (100 train / 100 test, 96 timepoints)
  2. Extract global statistical features (baseline, 9 features)
  3. Extract path signature features with time embedding for N ∈ {1,2,3,4,5}
  4. Evaluate each model by cross-validation 5-fold on the train set
  5. Display comparative table + performance vs truncation level curves
  6. Train best models on the full train set, evaluate on the test set
  7. Save results to experiments/increment1/results/

Time embedding:
  γ(i) = (i/(T-1), xᵢ) ∈ ℝ²  with T=96 for ECG200
  → 2D path whose signature encodes temporal order and interactions
    between time and amplitude at each truncation level.
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH for src.* imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.loader import load_ecg200
from src.features.statistical import extract_statistical_features, FEATURE_NAMES
from src.features.signature import extract_signature_features, signature_dim, LEVELS
from src.models.classifiers import cross_val_evaluate, fit_and_predict
from src.evaluation.metrics import (
    compute_test_metrics,
    print_cv_table,
    plot_performance_vs_level,
    plot_confusion_matrices,
    plot_roc_curves,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# MAIN PIPELINE
# ===========================================================================

def main() -> None:

    # -----------------------------------------------------------------------
    # 1. Load ECG200
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("INCREMENT 1 — Baseline vs Signature (time embedding)")
    print("=" * 60)

    print("\n[1/7] Loading ECG200...")
    X_train, y_train, X_test, y_test = load_ecg200()
    print(f"  Train : {X_train.shape}  | Classes : {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Test  : {X_test.shape}   | Classes : {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    # -----------------------------------------------------------------------
    # 2. Statistical features (baseline)
    # -----------------------------------------------------------------------
    print("\n[2/7] Extracting statistical features (baseline)...")
    X_stat_train = extract_statistical_features(X_train)
    X_stat_test  = extract_statistical_features(X_test)
    print(f"  {X_stat_train.shape[1]} features : {FEATURE_NAMES}")

    # -----------------------------------------------------------------------
    # 3. Signature features — time embedding, N ∈ {1,2,3,4,5}
    # -----------------------------------------------------------------------
    print("\n[3/7] Extracting path signature features (time embedding)...")
    sig_train = {}
    sig_test  = {}
    for level in LEVELS:
        dim = signature_dim(level, d=2)
        print(
            f"  Level {level} : {dim:>2} features "
            f"(Σ_{{k=1}}^{{{level}}} 2^k)...",
            end=" ", flush=True,
        )
        sig_train[level] = extract_signature_features(X_train, level=level)
        sig_test[level]  = extract_signature_features(X_test,  level=level)
        print("✓")

    # -----------------------------------------------------------------------
    # 4. Cross-validation on the train set
    # -----------------------------------------------------------------------
    print("\n[4/7] Stratified 5-fold cross-validation (train set)...")

    # Baseline
    baseline_cv = cross_val_evaluate(
        X_stat_train, y_train,
        model_name="Baseline (statistical, 9 feat.)",
    )
    baseline_cv["level"] = None  # no level for the baseline

    # Signature: one result per truncation level
    sig_cv_list = []
    for level in LEVELS:
        dim  = signature_dim(level, d=2)
        name = f"Signature L={level} ({dim} feat.)"
        result = cross_val_evaluate(sig_train[level], y_train, model_name=name)
        result["level"] = level
        sig_cv_list.append(result)
        print(
            f"  L={level} : acc={result['accuracy_mean']:.4f}±{result['accuracy_std']:.4f}"
            f"  f1={result['f1_mean']:.4f}  auc={result['roc_auc_mean']:.4f}"
        )

    # -----------------------------------------------------------------------
    # 5. Comparative table + CSV export
    # -----------------------------------------------------------------------
    print("\n[5/7] Comparative table...")
    all_cv = [baseline_cv] + sig_cv_list
    df_cv  = print_cv_table(all_cv)
    csv_path = RESULTS_DIR / "cv_results.csv"
    df_cv.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"  CSV saved: {csv_path}")

    # -----------------------------------------------------------------------
    # 6. Performance vs truncation level curves
    # -----------------------------------------------------------------------
    print("\n[6/7] Generating performance vs truncation level curves...")
    for metric in ["accuracy_mean", "f1_mean", "roc_auc_mean"]:
        metric_short = metric.replace("_mean", "")
        plot_performance_vs_level(
            sig_results=sig_cv_list,
            baseline_result=baseline_cv,
            metric=metric,
            save_path=RESULTS_DIR / f"perf_vs_level_{metric_short}.png",
        )

    # -----------------------------------------------------------------------
    # 7. Evaluation on the test set
    # -----------------------------------------------------------------------
    print("\n[7/7] Final evaluation on the test set...")

    # Baseline
    y_pred_base, y_proba_base = fit_and_predict(X_stat_train, y_train, X_stat_test)
    base_test = compute_test_metrics(y_test, y_pred_base, y_proba_base)
    print(f"  Baseline       : {_fmt_metrics(base_test)}")

    # Best truncation level by CV accuracy
    best_result = max(sig_cv_list, key=lambda r: r["accuracy_mean"])
    best_level  = best_result["level"]
    print(f"  Best level (CV accuracy): L={best_level}")

    y_pred_sig, y_proba_sig = fit_and_predict(
        sig_train[best_level], y_train, sig_test[best_level],
    )
    sig_test_metrics = compute_test_metrics(y_test, y_pred_sig, y_proba_sig)
    print(f"  Signature L={best_level}    : {_fmt_metrics(sig_test_metrics)}")

    # Save test metrics
    df_test = pd.DataFrame([
        {"Model": "Baseline (statistical)", **base_test},
        {"Model": f"Signature L={best_level} (best CV)", **sig_test_metrics},
    ])
    test_csv = RESULTS_DIR / "test_metrics.csv"
    df_test.to_csv(test_csv, index=False, encoding="utf-8")
    print(f"\n  Test metrics saved: {test_csv}")

    # Confusion matrices (test set)
    plot_confusion_matrices(
        results=[
            {
                "model_name": "Baseline (stat.)",
                "y_true": y_test, "y_pred": y_pred_base,
            },
            {
                "model_name": f"Signature L={best_level}",
                "y_true": y_test, "y_pred": y_pred_sig,
            },
        ],
        save_path=RESULTS_DIR / "confusion_matrices.png",
    )

    # ROC curves (test set)
    plot_roc_curves(
        results=[
            {
                "model_name":   "Baseline (statistical)",
                "y_true":       y_test,
                "y_pred_proba": y_proba_base,
                "roc_auc":      base_test["roc_auc"],
            },
            {
                "model_name":   f"Signature L={best_level}",
                "y_true":       y_test,
                "y_pred_proba": y_proba_sig,
                "roc_auc":      sig_test_metrics["roc_auc"],
            },
        ],
        save_path=RESULTS_DIR / "roc_curves.png",
    )

    print(f"\n{'='*60}")
    print(f"Full results in: {RESULTS_DIR}")
    print(f"{'='*60}\n")


# ===========================================================================
# UTILITY
# ===========================================================================

def _fmt_metrics(m: dict) -> str:
    return (
        f"acc={m['accuracy']:.4f}  "
        f"f1={m['f1']:.4f}  "
        f"auc={m['roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()
