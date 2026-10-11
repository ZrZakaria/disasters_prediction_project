# 🌍 Natural Disasters Prediction & Analysis Project

> A complete ML pipeline for analyzing and predicting natural disaster trends — from raw data cleaning to interactive model comparison  
> *S7 Database & Python Project — ENSA, Academic Year 2025–2026*

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-purple)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange)](https://xgboost.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org/)

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Pipeline Steps](#pipeline-steps)
- [Technology Stack](#technology-stack)
- [How to Run](#how-to-run)
- [Limitations](#limitations)

---

## Overview

This project builds a complete machine learning pipeline for natural disaster analysis across three predictive tasks:

1. **Classification** — Predict the type of disaster (earthquake, flood, storm, etc.) from location and magnitude features.
2. **Clustering** — Identify geographic regions with similar disaster profiles using unsupervised learning.
3. **Regression** — Estimate human impact (total deaths, affected people, financial damages) from contextual features.

The pipeline is organized into four Jupyter notebooks, progressing from data preparation through interactive model comparison.

---

## Dataset

| Attribute | Value |
|-----------|-------|
| **Source** | NOAA / EM-DAT natural disaster database |
| **Original rows** | 16,126 disaster events |
| **Original columns** | 45 |
| **Cleaned rows** | 15,818 (after dropping 308 rows with critical missing values) |
| **Cleaned columns** | 23 (after removing 22 administrative/identifier columns) |
| **Time span** | 1900–2023 |
| **Geographic coverage** | Global (all continents) |
| **Disaster types** | Earthquake, Flood, Storm, Drought, Wildfire, Volcanic Activity, etc. |

### Features Used

| Feature | Type | Description |
|----------|------|-------------|
| `Year` | int | Event year |
| `Month` | int | Event month |
| `Duration_Days` | int | Duration of the event |
| `Magnitude` | float | Magnitude scale value |
| `Latitude` | float | Event latitude |
| `Longitude` | float | Event longitude |
| `Total_Deaths` | int | Total fatalities (regression target) |
| `Total_Affected` | int | Total people affected (regression target) |
| `Total_Damages_USD` | float | Financial damages in USD (regression target) |
| `Disaster_Type` | str | Primary disaster category (classification target) |
| `Region` | str | Geographic region |
| `Country` | str | Country affected |

---

## Model Performance

### Classification — Disaster Type Prediction

| Model | Accuracy | Precision (weighted) | Recall (weighted) | F1 (weighted) |
|-------|----------|---------------------|-------------------|---------------|
| KNN (k=5) | 54.14% | — | — | — |
| SVM (RBF) | 46.93% | — | — | — |
| Decision Tree | 58.47% | — | — | — |
| **Random Forest** ✅ | **61.03%** | — | — | — |
| MLP (Neural Network) | 56.29% | — | — | — |

> **Best model:** Random Forest with 61.03% accuracy. The multi-class classification task is inherently challenging due to the imbalanced distribution of disaster types and the high dimensionality of geographic features.

### Clustering — Geographic Region Profiling

| Model | Silhouette Score | Davies-Bouldin Index |
|-------|-----------------|---------------------|
| **K-Means (k=4)** ✅ | **0.299** | **1.114** |
| DBSCAN | 0.272 | 0.941 |
| Spectral Clustering | -0.414 | 1.898 |

> **Best model:** K-Means with k=4 clusters. A silhouette score of 0.30 indicates moderately well-separated clusters — acceptable for geographic disaster profiling where boundaries are inherently fuzzy. DBSCAN found 5 clusters with 892 noise points.

### Regression — Impact Estimation

All regression targets were log-transformed (`log(y+1)`) to handle extreme right-skewed distributions. MAE is reported in log-space; interpretation: `exp(MAE) ≈ multiplicative factor of error`.

#### Total Deaths

| Model | MAE (log) | RMSE (log) | R² |
|-------|-----------|------------|-----|
| Random Forest | 1.209 | 1.570 | 0.398 |
| **XGBoost** ✅ | **1.179** | **1.547** | **0.416** |
| Linear Regression | 1.428 | 1.806 | 0.205 |

> XGBoost achieves R² = 0.416, meaning it explains ~42% of variance in disaster fatalities. In log-space, MAE of 1.18 means predictions are within a factor of ~3.3× of actual values.

#### Total People Affected

| Model | MAE (log) | RMSE (log) | R² |
|-------|-----------|------------|-----|
| **Random Forest** ✅ | **2.931** | **3.789** | **0.355** |
| XGBoost | 2.891 | 3.803 | 0.351 |
| Linear Regression | 3.672 | 4.390 | 0.135 |

#### Total Financial Damages

| Model | MAE (log) | RMSE (log) | R² |
|-------|-----------|------------|-----|
| **Random Forest** ✅ | **3.055** | **4.085** | **0.401** |
| XGBoost | 2.977 | 4.088 | 0.400 |
| Linear Regression | 4.364 | 5.017 | 0.096 |

> **Key insight:** For all three regression targets, tree-based models (Random Forest, XGBoost) dramatically outperform linear regression, confirming the non-linear relationships between disaster features and human/financial impact.

---

## Project Structure

```
disasters_prediction_project/
├── README.md                         # This file
├── .gitattributes
│
├── STEP1_PRETREATMENT/
│   ├── Pretraitement.ipynb           # Data cleaning, feature engineering
│   ├── DISASTERS.csv                 # Raw dataset (16,126 rows)
│   ├── DISASTERS_CLEANED.csv         # Cleaned dataset (15,818 rows, 23 cols)
│   └── extract_mappings.py           # Geographic code mappings
│
├── STEP2_VISUALISATION/
│   ├── visualisation_1.ipynb         # Temporal trends, frequency analysis
│   └── visualisation_2.ipynb         # Geospatial mapping, magnitude distribution
│
├── STEP3_MODELS/
│   ├── classification_models.ipynb   # KNN, SVM, DT, RF, MLP → RF best (61.0%)
│   ├── clustering_models.ipynb       # KMeans, DBSCAN, Spectral → KMeans best
│   └── regression_models.ipynb       # LR, RF, XGBoost → XGBoost/RF best
│
├── STEP4_COMPARISON/
│   └── compare_all_models.ipynb      # Interactive ipywidgets dashboard
│
└── rapport/
    ├── project_report.pdf            # Full LaTeX report
    ├── project_report.tex            # LaTeX source
    ├── *.png                         # 15 visualization figures
    └── step*.docx                    # Step-by-step documentation
```

---

## Pipeline Steps

### Step 1 — Pre-treatment
Removes 22 administrative columns, handles missing values, creates temporal features (`Year`, `Month`, `Duration_Days`), and exports the cleaned dataset.

### Step 2 — Visualization
EDA notebooks produce temporal trend charts, geospatial heatmaps, magnitude distributions, correlation heatmaps, and cluster visualizations.

### Step 3 — Modeling
Three independent notebooks implement and evaluate models for each task:
- **Classification:** 5 models compared via accuracy on held-out test set
- **Clustering:** 3 algorithms compared via silhouette score and Davies-Bouldin index
- **Regression:** 3 models compared via R², MAE, RMSE on log-transformed targets

### Step 4 — Interactive Comparison
An `ipywidgets`-based dashboard allows real-time hyperparameter tuning and model switching across all three tasks, with live-updating performance charts.

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.10+ |
| **Data handling** | Pandas, NumPy |
| **Machine learning** | Scikit-Learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Interaction** | ipywidgets |
| **Notebook env** | Jupyter Notebook / JupyterLab |

---

## How to Run

```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn xgboost ipywidgets jupyter

# Run the pipeline sequentially
jupyter notebook STEP1_PRETREATMENT/Pretraitement.ipynb
jupyter notebook STEP2_VISUALISATION/visualisation_1.ipynb
jupyter notebook STEP2_VISUALISATION/visualisation_2.ipynb
jupyter notebook STEP3_MODELS/classification_models.ipynb
jupyter notebook STEP3_MODELS/clustering_models.ipynb
jupyter notebook STEP3_MODELS/regression_models.ipynb
jupyter notebook STEP4_COMPARISON/compare_all_models.ipynb
```

---

## Limitations

| Limitation | Impact | Proposed improvement |
|------------|--------|---------------------|
| No temporal train/test split | Potential data leakage (future data leaks into training) | Split by year: train on pre-2020, test on 2020+ |
| Classification accuracy ~61% | Limited practical utility for rare disaster types | Use class-weighted loss, SMOTE, or focal loss |
| Regression R² < 0.5 | Models explain less than half the variance | Add external features: GDP, infrastructure quality, population density |
| No deep learning models | Missing CNN/RNN baselines for spatial-temporal patterns | Experiment with LSTM for temporal sequences |
| Static notebooks | No automated evaluation pipeline | Add a `evaluation/` script with CI integration |
| No model persistence | Models not saved for production use | Export best models with `joblib` or `pickle` |

---

*Developed as part of the S7 Python & Data Science course at ENSA.*
