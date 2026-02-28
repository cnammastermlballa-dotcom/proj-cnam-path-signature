"""
Path Signature features with time embedding — Increment 1.

=============================================================================
THEORY: PATH SIGNATURE
=============================================================================

The signature of a path γ : [0,T] → ℝ^d is the sequence of tensors:

  S(γ) = (1, S¹(γ), S²(γ), ..., Sᴺ(γ), ...)

where the term of order k is the iterated integral:

  Sⁱ¹...ⁱᵏ(γ) = ∫_{0<t₁<...<tₖ<T} dγⁱ¹_{t₁} ⊗ ... ⊗ dγⁱᵏ_{tₖ}

Key properties:
  - Invariance under monotone increasing reparametrisation
  - Sensitivity to the order of events (iterated integrals)
  - Universality: the signature characterises the path (up to tree-like equivalence)
  - Terms of order k capture k-th order interactions between components

=============================================================================
TIME EMBEDDING — INCREMENT 1
=============================================================================

For a univariate series x = (x₀, ..., x_{T-1}), we construct the 2D path:

  γ : {0, ..., T-1} → ℝ²
  γ(i) = (tᵢ, xᵢ)   with   tᵢ = i / (T-1) ∈ [0, 1]

Why add time?
  → Without a time dimension, the signature of a 1D series is trivial:
    the order-1 term reduces to the total increment x_{T-1} - x₀.
    All information about intermediate oscillations is lost.

  → By adding t as a first component, we create a 2D path whose
    signature explicitly encodes:
      - S^t   = time increment (normalised to 1)
      - S^x   = total increment x_{T-1} - x₀
      - S^{tx} = ∫ t dx  → time-weighted integral
      - S^{xt} = ∫ x dt  → "time-weighted mean" of the series
      - S^{xx} = ∫ dx ∘ dx / 2 = (Δx)² / 2  → quadratic variation
      - (higher-order terms: more complex interactions)

Note: time embedding is the natural starting point (increment 1).
The lead-lag transformation (increment 2) is richer: it captures
local variation and is independent of the time parametrisation.

=============================================================================
TRUNCATION LEVEL
=============================================================================

We truncate the signature at level N, yielding a finite number of features:

  dim(N, d) = Σ_{k=1}^{N} d^k

For d=2 (time embedding):
  N=1 →  2 features   [Sᵗ, Sˣ]
  N=2 →  6 features   + [Sᵗᵗ, Sᵗˣ, Sˣᵗ, Sˣˣ]
  N=3 → 14 features   + 8 order-3 terms
  N=4 → 30 features   + 16 order-4 terms
  N=5 → 62 features   + 32 order-5 terms

Expressiveness / dimensionality trade-off:
  - N too small: under-captures complex dynamics
  - N too large: high dimension → overfitting on 100 samples
  → We test N ∈ {1, 2, 3, 4, 5} and select by cross-validation.

=============================================================================
NORMALISATION
=============================================================================

Terms of level k have magnitude ~ ||Δγ||^k / k!  (Taylor series).
For a normalised ECG series (amplitude ~ 1), higher orders are very
small (1/k!), creating highly heterogeneous scales across levels.
→ A StandardScaler is applied inside the classifier pipeline (see classifiers.py).
"""

import numpy as np
import esig


# Truncation levels tested in the experiment
LEVELS = [1, 2, 3, 4, 5]


def time_embed(x: np.ndarray) -> np.ndarray:
    """
    Build the 2D path by time embedding of a univariate series.

    Parameters
    ----------
    x : np.ndarray, shape (T,)

    Returns
    -------
    path : np.ndarray, shape (T, 2)
        Column 0: normalised time tᵢ = i/(T-1) ∈ [0, 1]
        Column 1: value xᵢ
    """
    T = len(x)
    t = np.linspace(0.0, 1.0, T)    # normalised time in [0, 1]
    return np.column_stack([t, x])   # shape (T, 2)


def extract_signature_features(
    X: np.ndarray,
    level: int = 4,
) -> np.ndarray:
    """
    Extract path signature features (time embedding) for each series.

    For each series:
      1. Build the 2D path γ(i) = (tᵢ, xᵢ)
      2. Compute the truncated signature at level `level` via esig.stream2sig()

    Parameters
    ----------
    X     : np.ndarray, shape (n_samples, n_timepoints)
    level : int — truncation level (1 to 5 recommended)

    Returns
    -------
    features : np.ndarray, shape (n_samples, dim)
        dim = Σ_{k=1}^{level} 2^k  (see signature_dim())
    """
    n_samples = X.shape[0]

    # Compute output dimension from the first sample
    # (more reliable than the theoretical formula for verification)
    first_sig = esig.stream2sig(time_embed(X[0]), level)
    n_features = len(first_sig)

    features = np.zeros((n_samples, n_features))
    for i, x in enumerate(X):
        path = time_embed(x)
        features[i] = esig.stream2sig(path, level)

    return features


def signature_dim(level: int, d: int = 2) -> int:
    """
    Theoretical dimension of the signature truncated at `level`.

    For d channels and truncation level N:
      dim = Σ_{k=1}^{N} d^k = d*(d^N - 1)/(d-1)  for d > 1

    Parameters
    ----------
    level : int — truncation level
    d     : int — path dimension (2 for time embedding or lead-lag)
    """
    return sum(d ** k for k in range(1, level + 1))
