# Multi-Class Benchmark Classification Pipeline (Iris Dataset)

An end-to-end supervised machine learning pipeline implemented with **Scikit-Learn**, **Pandas**, **Seaborn**, and **Matplotlib**. This project demonstrates standard classification workflows: data exploration, stratified 80/20 train/test splitting, feature standardization with `StandardScaler` (guarding against data leakage), model training, metric benchmarking, confusion matrix heatmap analysis, and 2D decision boundary visualization.

---

## 📁 Repository & Project Structure

```
iris_classification_pipeline/
│
├── iris_classification_pipeline.ipynb   # Main deliverable: fully-executed Jupyter Notebook
├── pipeline.py                          # Standalone executable Python pipeline script
├── build_notebook.py                    # Programmatic notebook builder & execution engine
├── extract_csv_from_pdf.py              # Extractor utility parsing PDF into clean iris.csv
├── raw.githubusercontent.com_...pdf     # Source dataset saved as PDF
├── iris.csv                             # Clean Iris benchmark dataset (150 samples x 5 columns)
├── requirements.txt                     # Dependencies & environment specifications
├── README.md                            # Comprehensive project documentation & results
└── figures/                             # High-resolution saved visualizations
    ├── 01_iris_pairplot.png
    ├── 02_eda_distributions_and_correlation.png
    ├── 03_decision_tree_architecture.png
    ├── 04_benchmark_metrics_comparison.png
    ├── 05_confusion_matrix_heatmaps.png
    └── 06_decision_boundary_comparison.png
```

---

## 📊 Benchmark Results Summary

On the held-out stratified 20% test set (30 samples: 10 *setosa*, 10 *versicolor*, 10 *virginica*):

| Metric | Logistic Regression (L-BFGS) | Decision Tree Classifier (`max_depth=3`) |
| :--- | :---: | :---: |
| **Accuracy** | **93.33%** (28/30) | **96.67%** (29/30) |
| **Precision (Macro Avg)** | **0.9333** | **0.9697** |
| **Recall (Macro Avg)** | **0.9333** | **0.9667** |
| **F1-Score (Macro Avg)** | **0.9333** | **0.9666** |
| **Precision (Weighted Avg)** | **0.9333** | **0.9697** |
| **Recall (Weighted Avg)** | **0.9333** | **0.9667** |
| **F1-Score (Weighted Avg)** | **0.9333** | **0.9666** |

### Detailed Classification Reports

#### 1. Logistic Regression
```text
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        10
  versicolor     0.9000    0.9000    0.9000        10
   virginica     0.9000    0.9000    0.9000        10

    accuracy                         0.9333        30
   macro avg     0.9333    0.9333    0.9333        30
weighted avg     0.9333    0.9333    0.9333        30
```

#### 2. Decision Tree Classifier
```text
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        10
  versicolor     1.0000    0.9000    0.9474        10
   virginica     0.9091    1.0000    0.9524        10

    accuracy                         0.9667        30
   macro avg     0.9697    0.9667    0.9666        30
weighted avg     0.9697    0.9667    0.9666        30
```

---

## 🔍 Key Findings & Visualizations

1. **Exploratory Data Analysis (EDA)**:
   - *Iris setosa* is linearly separable from the other two classes using petal measurements alone ($petal\_length < 2.0\text{ cm}$).
   - Petal length and petal width show an extremely strong positive correlation ($r = 0.96$).
2. **Data Leakage Prevention**:
   - `StandardScaler` was fitted strictly on `X_train` and then applied to transform both `X_train` and `X_test`.
3. **Confusion Matrix Heatmaps**:
   - Both models achieve 100% precision and recall on *setosa*.
   - Logistic Regression misclassifies 1 *versicolor* as *virginica* and 1 *virginica* as *versicolor*.
   - Decision Tree misclassifies only 1 *versicolor* as *virginica*.
4. **Decision Boundary Comparison**:
   - **Logistic Regression**: Creates smooth linear boundaries (hyperplanes in $\mathbb{R}^d$) separating the multi-class feature regions.
   - **Decision Tree**: Creates rectilinear, orthogonal step-partitions parallel to the feature coordinate axes.

---

## 🚀 How to Run

### Option 1: Jupyter Notebook (Recommended)
Open and view or re-run the notebook directly:
```bash
jupyter notebook iris_classification_pipeline.ipynb
# or open in VS Code with the Jupyter extension
```

### Option 2: Command-Line Pipeline Script
Execute the standalone pipeline script to print metrics and re-generate figures:
```bash
python pipeline.py
```

### Option 3: Re-generate and Execute the Notebook Programmatically
```bash
python build_notebook.py
```
