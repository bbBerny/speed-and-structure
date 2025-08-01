# Seismic Analysis Pipeline

## Overview

This repository implements a workflow for seismic velocity model inversion using synthetic shot-gather data. The pipeline includes data inspection, feature engineering, clustering, geophysical analysis, and baseline modeling before moving on to end-to-end CNN inversion.

## Repository Structure

```
├── data/
│   ├── raw/
│   │   ├── train_data/   # Synthetic shot-gather + vp_model arrays
│   │   └── test_data/    # Unlabeled data for final evaluation
│   └── processed/
│       ├── features.csv           # Final hand-crafted feature matrix
│       ├── clusters2d.csv         # UMAP+DBSCAN cluster labels
│       └── final_features.csv     # Features + UMAP coords for modeling

├── notebooks/
│   ├── 01_data_overview.ipynb     # Data shapes, sanity checks, original distributions
│   ├── 02_feature_engineering.ipynb # Compute: time-domain, spectral, spatial descriptors
│   ├── 03_clustering_pca.ipynb    # 2D PCA/UMAP, clustering methods, outlier detection
│   ├── 04_semblance_analysis.ipynb # Semblance & first-break analysis
│   ├── 05_baseline_modeling.ipynb  # Feature-only and CNN+MLP baselines

├── src/
│   ├── loaders.py        # Data-loading utilities
│   ├── features.py       # Feature-computing functions
│   ├── geophysical.py    # Similar as features
│   └── plotting.py       # Matplotlib and Seaborn functions

├── models/
│   └── cnn_prototype.keras

├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## Prerequisites

- Python 3.9
- Virtual environment (recommended)
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Usage

1. **Explore data:**
   ```bash
   jupyter lab notebooks/01_data_overview.ipynb
   ```
2. **Engineer features:**
   ```bash
   jupyter lab notebooks/02_feature_engineering.ipynb
   ```
3. **Clustering & outliers:**
   ```bash
   jupyter lab notebooks/03_clustering_pca.ipynb
   ```
4. **Geophysical analysis:**
   ```bash
   jupyter lab notebooks/04_semblance_analysis.ipynb
   ```
5. **Baseline modeling:**
   ```bash
   jupyter lab notebooks/05_baseline_modeling.ipynb
   ```

## Currently

- Integrating CNN inversion model into end-to-end training.
- Optimizing hyperparameters via automated search.
