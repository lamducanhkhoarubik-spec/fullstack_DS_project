# UK Online Retail — Customer Segmentation

Behavioural analysis and customer segmentation for a UK-based online retail store, built on a feature engineering pipeline and K-Means clustering.

---

## Overview

This project answers one question: **what behavioural segments exist among the store's customers, and how should each be treated?**

Instead of relying on classic RFM (Recency, Frequency, Monetary) alone, the pipeline extends to **~20 behavioural features across 5–7 families** (purchase rhythm, spending shape, basket behaviour, volume & bulk, product concentration, customer lifecycle, seasonality), then carefully filters and normalizes them before K-Means, so the resulting clusters reflect real behaviour rather than artifacts of scale or outliers.

---

## Project structure

```
.
├── data/
│   ├── raw/                    # Raw source data (online_retail.csv)
│   └── processed/               # Cleaned data / computed features
├── notebooks/
│   ├── eda_and_cleaning.ipynb       # Exploration + data cleaning
│   └── feature_engineering.ipynb    # Feature pipeline build & QA
├── src/
│   ├── clustering_library/
│   │   ├── cleaner.py                       # Raw data cleaning
│   │   ├── features.py                      # FeatureEngineer — feature pipeline
│   │   ├── clustering_with_rfm_features.py  # K-Means training & evaluation
│   │   └── visualizer.py                    # Cluster analysis & plotting
│   ├── notebook_io.py            # Shared read/write utilities for notebooks
│   └── visual_style.py           # Shared plotting theme/style
├── tests/                        # Unit tests
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the raw dataset at `data/raw/online_retail.csv` (encoding `ISO-8859-1`, with columns `InvoiceNo`, `StockCode`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`).

---

## Pipeline

| Step | File | Output |
|---|---|---|
| 1. Cleaning | `notebooks/eda_and_cleaning.ipynb` (uses `cleaner.py`) | `data/processed/cleaned_uk_data.csv` |
| 2. Feature engineering | `notebooks/feature_engineering.ipynb` (uses `features.py`) | `data/processed/customer_features*.csv` + `models/*.pkl` |
| 3. Clustering | `clustering_with_rfm_features.py` | Cluster labels + evaluation (silhouette, inertia...) |
| 4. Visualization | `visualizer.py` | Cluster profile plots & analysis |

```python
from clustering_library.features import FeatureEngineer

fe = FeatureEngineer(
    data_path="data/processed/cleaned_uk_data.csv",
    raw_data_path="data/raw/online_retail.csv",
)
fe.load_data()
fe.create_customer_features()   # 20 candidates → correlation filter in model space
fe.transform_features()         # QuantileTransformer(normal)
fe.scale_features()             # StandardScaler + ±3σ clip + family weighting
fe.save_features()              # Export CSVs + persist pipeline (scaler, quantile transformer)
```

---

## Methodology (summary)

**5 behavioural feature families:**

- **Purchase Rhythm** — order frequency & regularity (`Recency`, `MonthlyOrderRate`, `InterPurchaseCV`...)
- **Spending Shape** — shape of spend over time (`AOV`, `SpendGini`, `SpendAcceleration`...)
- **Basket Behaviour** — product selection habits within baskets
- **Volume & Bulk** — bulk-buying / burst behaviour
- **Customer Lifecycle** — return rate, seasonal concentration

**Normalization pipeline (order matters):**

```
Raw candidates (20 features)
  → QuantileTransformer(normal)      # map every feature to the same normal-quantile scale
  → Correlation filter (|r| > 0.85)  # drop redundant features IN model space
  → StandardScaler + ±3σ clip        # unify scale, cap remaining outliers
  → Family weighting                 # 1/√(features remaining per family)
  → K-Means
```

A set of highly interpretable features (`Recency`, `MonthlyOrderRate`, `ReturnRate`, `QuarterConcentration`, `SKU_HHI`) is force-kept (`FORCE_KEEP`) so the final clusters remain explainable in business terms, even where they show high statistical correlation with other features.

---

## Requirements

See `requirements.txt` / `pyproject.toml`. Core stack: `pandas`, `numpy`, `scikit-learn`, `joblib`.

---

## CI/CD

This repo runs automated checks on every push / pull request:

- **Tests** — `pytest` over `tests/`
- **Lint / format** — style and static checks on `src/`
- **(optional) Notebook execution check** — sanity-run of `notebooks/*.ipynb` to catch broken pipeline changes early

> Update the badge and workflow file paths below once the CI provider/workflow file names are finalized.

```
[![CI](https://github.com/<org>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<org>/<repo>/actions/workflows/ci.yml)
```

Workflow file lives at `.github/workflows/ci.yml`. Typical stages:

```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

---

## Notes

- Source dataset: online retail transactions for a company based in the UK, filtered to `Country == "United Kingdom"`.
- `data/raw/` and `data/processed/` are not committed with real data to Git (see `.gitignore`) — only `.gitkeep` files are kept to preserve folder structure.
