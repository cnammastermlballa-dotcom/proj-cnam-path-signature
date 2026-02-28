"""
Global statistical features — time-agnostic baseline.

These 9 features capture the distribution of series values,
but deliberately ignore their temporal order.

This is precisely the limitation that path signatures address:
the signature encodes the geometry of the path and the temporal
sequence of variations, not merely the marginal distribution of values.

Extracted features (in order):
  1. mean               : arithmetic mean
  2. std                : standard deviation
  3. min                : minimum value
  4. max                : maximum value
  5. median             : median
  6. skewness           : distribution asymmetry (normalised 3rd central moment)
  7. kurtosis           : distribution flatness (4th moment)
  8. energy             : signal energy = ||x||²₂ = Σ xᵢ²
  9. dominant_frequency : index of the most energetic FFT bin (excluding DC)
"""

import numpy as np
from scipy import stats
from scipy.fft import rfft


FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "skewness",
    "kurtosis",
    "energy",
    "dominant_frequency",
]


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    """
    Extract global statistical features for each time series.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_timepoints)

    Returns
    -------
    features : np.ndarray, shape (n_samples, 9)
    """
    return np.vstack([_extract_one(x) for x in X])


def _extract_one(x: np.ndarray) -> np.ndarray:
    """Extract the 9 statistical features from a univariate series."""
    # Dominant frequency via FFT: index of the highest-amplitude bin
    # Bin 0 (DC component = mean) is excluded as it is redundant with mean
    magnitudes = np.abs(rfft(x))
    magnitudes[0] = 0.0
    dominant_freq = float(np.argmax(magnitudes))

    return np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.median(x),
        float(stats.skew(x)),
        float(stats.kurtosis(x)),
        float(np.sum(x ** 2)),   # energy = ||x||²₂
        dominant_freq,
    ])
