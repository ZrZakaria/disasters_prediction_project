# 🌍 Natural Disasters Prediction & Analysis Project

This project provides a complete machine learning pipeline for analyzing and predicting natural disaster trends. It spans from raw data cleaning to interactive model comparison using advanced visualization and hyperparameter tuning.

## 🚀 Project Overview

The main objective is to understand disaster patterns (spatial, temporal, and magnitude) and build predictive models for:
1. **Classification**: Predicting the type of disaster based on location and magnitude.
2. **Clustering**: Identifying geographic regions with similar disaster profiles.
3. **Regression**: Estimating the human impact (total deaths) of events.

---

## 📁 Project Structure

The repository is organized into four logical steps, each containing specialized notebooks:

### [STEP 1: Pre-treatment](./STEP1_PRETREATMENT/)
Focuses on data cleaning and feature engineering.
- `Pretraitement.ipynb`: Handles missing values, encodes categorical features, and exports the `DISASTERS_CLEANED.csv` dataset.

### [STEP 2: Visualisation](./STEP2_VISUALISATION/)
Exploratory Data Analysis (EDA) to find correlations.
- `visualisation_1.ipynb`: Temporal trends and disaster frequency analysis.
- `visualisation_2.ipynb`: Geospatial mapping and magnitude distribution.

### [STEP 3: Models](./STEP3_MODELS/)
Individual implementation of machine learning tasks.
- `classification_models.ipynb`: KNN, SVM, Random Forest, MLP, etc.
- `clustering_models.ipynb`: KMeans, DBSCAN, Spectral Clustering.
- `regression_models.ipynb`: Linear Regression, Random Forest, XGBoost.

### [STEP 4: Comparison](./STEP4_COMPARISON/)
The crown jewel of the project — an interactive dashboard.
- `compare_all_models.ipynb`: A unified notebook with **ipywidgets** allowing users to:
  - Toggle between different models for each task.
  - Dynamically adjust **Hyperparameters** (e.g., learning rate, depth, neighbors).
  - Train and observe results in real-time with automatically updated charts.

---

## 🛠️ Technical Stack

- **Language**: Python 3.x
- **Data Handling**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost
- **Visualization**: Matplotlib, Seaborn
- **Interaction**: Ipywidgets

---

## 💻 How to Run

1. **Install Dependencies**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost ipywidgets
   ```
2. **Start the Pipeline**:
   - Begin with `STEP1_PRETREATMENT/Pretraitement.ipynb` to generate the clean dataset.
   - Explore individual models in `STEP3_MODELS/`.
   - Run the final dashboard in `STEP4_COMPARISON/compare_all_models.ipynb` for the full interactive experience.

---

## ✅ Key Achievements

- **Dataset Cleaned**: Transformed raw disaster data into a machine-learning-ready format.
- **Robust Modeling**: Implemented 10+ different algorithms across 3 ML domains.
- **Interactive UI**: Developed a custom tuning interface within Jupyter for real-time model evaluation.
