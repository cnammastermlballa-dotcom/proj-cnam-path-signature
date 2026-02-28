"""
Classifier: Logistic Regression inside a StandardScaler + LR pipeline.

=============================================================================
ARCHITECTURE CHOICES
=============================================================================

Both approaches (statistical baseline and path signature) use the same
classifier (Logistic Regression). This deliberate choice isolates the
contribution of the features, not the model — which is the thesis objective.

Pipeline: StandardScaler → LogisticRegression

StandardScaler:
  - Essential before any Logistic Regression (gradient descent is
    sensitive to scale differences between features)
  - Particularly important for signatures: terms of level k have
    magnitude ~ variation^k / k!, creating highly heterogeneous scales
    (order-5 terms ~ 1000× smaller than order-1 terms on ECG)

LogisticRegression:
  - solver='lbfgs'  : quasi-Newton, suited to small datasets (n=100)
                      and medium feature spaces (≤ 62)
  - max_iter=2000   : guaranteed convergence even for level-5 signatures
                      (62 features, few samples → difficult conditioning)
  - C=1.0 (default) : standard L2 regularisation
  - For increment 2+, consider tuning C with a grid search if overfitting

Cross-validation: StratifiedKFold(5)
  - Maintains class proportions in each fold
  - Essential with only n=100 training samples
  - random_state=42 for reproducibility
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate


# Cross-validation parameters
CV_FOLDS     = 5
RANDOM_STATE = 42

# Metrics evaluated at each fold
SCORING = {
    "accuracy": "accuracy",
    "f1":       "f1",
    "roc_auc":  "roc_auc",
}


def build_pipeline() -> Pipeline:
    """Build the StandardScaler → LogisticRegression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])


def cross_val_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Evaluate a model via stratified 5-fold cross-validation.

    Designed to be generic: accepts any feature matrix,
    whether statistical features or path signatures.

    Parameters
    ----------
    X          : np.ndarray, shape (n_samples, n_features)
    y          : np.ndarray, shape (n_samples,), binary labels {0, 1}
    model_name : str — model name for display and exports

    Returns
    -------
    dict with keys:
      model_name, n_features,
      accuracy_mean, accuracy_std,
      f1_mean, f1_std,
      roc_auc_mean, roc_auc_std,
      cv_raw : raw results from cross_validate (for further analysis)
    """
    pipeline = build_pipeline()
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    cv_raw = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=SCORING,
        return_train_score=False,
    )

    return {
        "model_name":    model_name,
        "n_features":    X.shape[1],
        "accuracy_mean": cv_raw["test_accuracy"].mean(),
        "accuracy_std":  cv_raw["test_accuracy"].std(),
        "f1_mean":       cv_raw["test_f1"].mean(),
        "f1_std":        cv_raw["test_f1"].std(),
        "roc_auc_mean":  cv_raw["test_roc_auc"].mean(),
        "roc_auc_std":   cv_raw["test_roc_auc"].std(),
        "cv_raw":        cv_raw,
    }


def fit_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train the pipeline on the full training set and predict on the test set.

    Returns
    -------
    y_pred       : np.ndarray — binary predictions {0, 1}
    y_pred_proba : np.ndarray — probabilities for the positive class (for ROC-AUC)
    """
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred       = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_proba
