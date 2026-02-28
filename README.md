# ECG Path Signature — Master's Thesis 2026

Classification of ECG time series using **Path Signatures**, benchmarked against classical statistical features.

**Dataset:** ECG200 (UCR Archive) — binary classification, 100 train / 100 test, 96 timepoints per series.

---

## Increments roadmap

| # | Approach | Classifier | Status |
|---|----------|------------|--------|
| **1** | Time embedding (t, xₜ) ∈ ℝ² | Logistic Regression | ✅ done |
| 2 | Lead-lag transformation | LogReg / Kernel SVM | ⏳ upcoming |
| 3 | Path Signature features | MLP vs LSTM | ⏳ upcoming |
| 4 | Signature Patching + Attention | Transformer-like | ⏳ upcoming |

Each completed increment is tagged in git (e.g. `v1.0`, `v2.0`, …).

---

## Requirements

- Python 3.10+
- Git
- A C compiler (`gcc` on Linux/macOS, Visual C++ Build Tools on Windows) — required by `esig`

---

## Installation

### 1 — Clone the repository

```bash
git clone https://github.com/cnammastermlballa-dotcom/proj-cnam-path-signature.git
cd proj-cnam-path-signature
```

### 2 — Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Verify the installation

```bash
python -c "import esig, aeon, sklearn; print('OK')"
```

---

## Running the experiments

### Increment 1 — Statistical baseline vs Path Signature (time embedding)

```bash
MPLBACKEND=Agg python experiments/increment1/run.py
```

> `MPLBACKEND=Agg` is required on headless / non-interactive environments to prevent matplotlib from blocking on a display connection.

**Output** — saved to `experiments/increment1/results/`:

| File | Content |
|------|---------|
| `cv_results.csv` | 5-fold cross-validation metrics for all models |
| `test_metrics.csv` | Final accuracy / F1 / AUC on the test set |
| `perf_vs_level_accuracy.png` | Accuracy vs truncation level N |
| `perf_vs_level_f1.png` | F1 vs truncation level N |
| `perf_vs_level_roc_auc.png` | ROC-AUC vs truncation level N |
| `confusion_matrices.png` | Confusion matrices (baseline vs best signature) |
| `roc_curves.png` | ROC curves (baseline vs best signature) |

**Increment 1 results (test set):**

| Model | Accuracy | F1 | ROC-AUC |
|-------|----------|----|---------|
| Statistical baseline (9 feat.) | 0.74 | 0.817 | 0.773 |
| **Path Signature L=5 (62 feat.)** | **0.83** | **0.866** | **0.870** |

---

## Project structure

```
proj-cnam-path-signature/
├── src/
│   ├── data/
│   │   └── loader.py            # ECG200 loading via aeon, label remapping
│   ├── features/
│   │   ├── statistical.py       # 9 time-agnostic statistical features (baseline)
│   │   └── signature.py         # Time embedding + truncated path signature (esig)
│   ├── models/
│   │   └── classifiers.py       # StandardScaler + LogisticRegression pipeline, CV
│   └── evaluation/
│       └── metrics.py           # Accuracy/F1/AUC, confusion matrices, ROC curves
├── experiments/
│   └── increment1/
│       ├── run.py               # Full pipeline for increment 1
│       └── results/             # Generated CSV and PNG (git-ignored)
├── environment.yml              # Conda environment definition
├── requirements.txt             # Pip-only alternative
└── README.md
```

---

## Reproducibility

| Parameter | Value |
|-----------|-------|
| Cross-validation | StratifiedKFold(5), random_state=42 |
| Normalisation | StandardScaler (fitted on train fold only) |
| Classifier | LogisticRegression(solver=lbfgs, max_iter=2000, C=1.0) |
| Truncation levels tested | [1, 2, 3, 4, 5] |

---

## References

- Lyons, T. (1998). *Differential equations driven by rough signals.* Revista Matemática Iberoamericana.
- Chevyrev, I. & Kormilitzin, A. (2016). *A primer on the signature method in machine learning.* arXiv:1603.03788.
- Dau, H.A. et al. (2019). *The UCR Time Series Archive.* IEEE/CAA Journal of Automatica Sinica.
- [`esig` library](https://esig.readthedocs.io) — efficient computation of path signatures.
- [`aeon` library](https://www.aeon-toolkit.org) — time series machine learning toolkit.
