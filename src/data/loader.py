"""
Loading the ECG200 dataset via aeon.

ECG200 (UCR Time Series Archive):
  - Binary classification: normal vs pathological ECG
  - Train: 100 samples | Test: 100 samples
  - Length of each series: 96 time points
  - Labels: "-1" (abnormal) and "1" (normal) → remapped to {0, 1}

Reference:
  Olszewski, R.T. (2001). Generalized Feature Extraction for Structural
  Pattern Recognition in Time-Series Data. CMU PhD Thesis.
"""

import numpy as np
from aeon.datasets import load_classification


def load_ecg200() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ECG200 and return train/test splits in matrix format.

    Returns
    -------
    X_train : np.ndarray, shape (100, 96)  — time series
    y_train : np.ndarray, shape (100,)     — labels {0, 1}
    X_test  : np.ndarray, shape (100, 96)
    y_test  : np.ndarray, shape (100,)
    """
    X_train_raw, y_train_raw = load_classification("ECG200", split="train")
    X_test_raw,  y_test_raw  = load_classification("ECG200", split="test")

    # aeon returns shape (n_samples, n_channels, n_timepoints)
    # ECG200 is univariate: n_channels = 1 → extract the single channel
    X_train = X_train_raw[:, 0, :]  # (100, 96)
    X_test  = X_test_raw[:, 0, :]   # (100, 96)

    # ECG200 labels: strings "-1" and "1" → remapped to integers 0 and 1
    y_train = _remap_labels(y_train_raw)
    y_test  = _remap_labels(y_test_raw)

    return X_train, y_train, X_test, y_test


def _remap_labels(y_raw: np.ndarray) -> np.ndarray:
    """
    Convert arbitrary labels (str) to integers {0, 1}.

    Lexicographic sort of unique labels:
      first unique label → 0, second → 1.
    For ECG200: "-1" → 0, "1" → 1.
    """
    y = y_raw.astype(str)
    unique_sorted = sorted(np.unique(y))
    label_map = {label: idx for idx, label in enumerate(unique_sorted)}
    return np.array([label_map[lbl] for lbl in y], dtype=int)
