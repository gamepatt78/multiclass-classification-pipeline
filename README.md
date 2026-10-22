# 🌸 Multi-Class Benchmark Classification Pipeline
### Iris & Wine Datasets | Decision Tree · KNN · SVM · Random Forest · Logistic Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

A complete **multi-class classification benchmark pipeline** trained and evaluated on the **Iris** dataset. Five industry-standard classifiers are compared using accuracy, precision, recall, F1-score, confusion matrices, and decision boundary visualizations.

The pipeline demonstrates:
- Structured EDA with pairplots and correlation heatmaps
- Standardized preprocessing with `StandardScaler`
- Training & evaluation of 5 classifiers in a unified pipeline
- Head-to-head benchmark metrics table
- Confusion matrix heatmaps for every model
- 2D decision boundary plots using PCA-reduced features

---

## 📂 Project Structure

```
Task1_MultiClass_Classification_Pipeline/
│
├── iris.csv                              # Iris dataset (local)
├── iris_classification_pipeline.ipynb   # Main Jupyter Notebook
├── pipeline.py                          # Reusable pipeline module
├── build_notebook.py                    # Notebook builder script
├── requirements.txt                     # Python dependencies
├── figures/                             # Saved plot outputs
│   ├── 01_iris_pairplot.png
│   ├── 02_eda_distributions_and_correlation.png
│   ├── 03_decision_tree_architecture.png
│   ├── 04_benchmark_metrics_comparison.png
│   ├── 05_confusion_matrix_heatmaps.png
│   └── 06_decision_boundary_comparison.png
└── README.md
```

---

## 📊 Dataset

**Iris Dataset** — Fisher's classic multivariate dataset (1936).

| Feature | Description |
|---|---|
| `sepal_length` | Sepal length in cm |
| `sepal_width` | Sepal width in cm |
| `petal_length` | Petal length in cm |
| `petal_width` | Petal width in cm |
| `species` | ⭐ **Target** — Iris-setosa / versicolor / virginica |

- **Rows:** 150 samples
- **Classes:** 3 (perfectly balanced — 50 each)
- **Missing values:** None

---

## 🤖 Models Benchmarked

| # | Model | Key Hyperparameters |
|---|-------|---------------------|
| 1 | **Logistic Regression** | `max_iter=1000`, `multi_class=auto` |
| 2 | **Decision Tree** | `max_depth` tuned via CV |
| 3 | **K-Nearest Neighbors** | `n_neighbors` tuned via CV |
| 4 | **Support Vector Machine** | `kernel=rbf`, `C` tuned via CV |
| 5 | **Random Forest** | `n_estimators=100`, `max_depth` tuned |

---

## 🔧 Preprocessing Pipeline

```
Features (sepal/petal length & width)
    └── StandardScaler → Classifier

Train / Test Split: 80% / 20%  |  random_state=42
Cross-Validation: 5-fold Stratified CV
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~97% | ~97% | ~97% | ~97% |
| Decision Tree | ~95% | ~95% | ~95% | ~95% |
| KNN | ~97% | ~97% | ~97% | ~97% |
| **SVM** | **~98%** | **~98%** | **~98%** | **~98%** |
| Random Forest | ~97% | ~97% | ~97% | ~97% |

> Exact values vary per run due to train/test split randomness.

---

## 📉 Visualizations

| Plot | Description |
|------|-------------|
| Pairplot | Feature relationships colored by species |
| Distributions | KDE plots per feature per class |
| Correlation Heatmap | Pearson correlations between features |
| Decision Tree | Tree architecture visualization |
| Benchmark Bar Chart | Accuracy/F1 grouped bar comparison |
| Confusion Matrices | 5-panel heatmaps (one per model) |
| Decision Boundaries | 2D PCA projection of classifier boundaries |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Notebook
```bash
jupyter notebook iris_classification_pipeline.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| `Python 3.10+` | Core language |
| `Scikit-Learn` | All classifiers, pipelines, metrics |
| `Pandas / NumPy` | Data manipulation |
| `Matplotlib / Seaborn` | Visualizations |
| `Jupyter Notebook` | Interactive development |

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">
  Made with ❤️ as part of an ML classification benchmark study
</div>
