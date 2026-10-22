"""
Multi-Class Benchmark Classification Pipeline (Iris Dataset)
=============================================================
This script implements an end-to-end supervised classification pipeline
benchmarking Logistic Regression vs. Decision Tree on the Iris dataset.

Steps:
  1. Load and inspect the benchmark Iris dataset using Pandas.
  2. Exploratory Data Analysis (EDA) and visualization using Seaborn / Matplotlib.
  3. Stratified 80/20 Train-Test splitting.
  4. Feature Normalization using StandardScaler (preventing data leakage).
  5. Train Logistic Regression and Decision Tree Classifier models.
  6. Benchmark evaluation metrics (Accuracy, Precision, Recall, F1-Score) and Classification Reports.
  7. Plot Confusion Matrix Heatmaps.
  8. Visualize Decision Boundaries in 2D feature space.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------
# 0. Configuration & Paths
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "iris.csv")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set visual style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.dpi"] = 150

RANDOM_STATE = 42

def main():
    print("=" * 75)
    print("MULTI-CLASS BENCHMARK CLASSIFICATION PIPELINE: IRIS DATASET")
    print("=" * 75)

    # ---------------------------------------------------------
    # 1. Load Dataset
    # ---------------------------------------------------------
    pdf_source = os.path.join(BASE_DIR, "raw.githubusercontent.com_mwaskom_seaborn-data_master_iris.csv.pdf")
    if not os.path.exists(DATA_PATH):
        if os.path.exists(pdf_source):
            print(f"Dataset CSV not found. Extracting from local PDF: {pdf_source}...")
            try:
                import pymupdf
                doc = pymupdf.open(pdf_source)
                full_text = "\n".join([page.get_text() for page in doc])
                raw_lines = [line.strip() for line in full_text.splitlines() if line.strip()]
                csv_rows = [line for line in raw_lines if len(line.split(",")) == 5]
                with open(DATA_PATH, "w", encoding="utf-8") as f:
                    f.write("\n".join(csv_rows) + "\n")
                print(f"Extracted {len(csv_rows)-1} samples into {DATA_PATH}.")
            except Exception as e:
                print(f"Could not extract from PDF: {e}")
        if not os.path.exists(DATA_PATH):
            print(f"Dataset not found locally. Downloading to {DATA_PATH}...")
            url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
            import urllib.request
            urllib.request.urlretrieve(url, DATA_PATH)
            print("Download complete.")

    df = pd.read_csv(DATA_PATH)
    print(f"\n[+] Dataset loaded successfully.")
    print(f"    Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset Summary Information:")
    df.info()

    print("\nDescriptive Statistics:")
    print(df.describe().T)

    print("\nTarget Class Distribution:")
    class_counts = df["species"].value_counts()
    print(class_counts)
    print(f"Missing Values Check:\n{df.isnull().sum()}")

    # ---------------------------------------------------------
    # 2. Exploratory Data Analysis (EDA) & Pairplot
    # ---------------------------------------------------------
    print("\n[+] Generating Exploratory Data Analysis Visualizations...")

    # Pairplot
    pairplot_path = os.path.join(FIGURES_DIR, "01_iris_pairplot.png")
    palette = {"setosa": "#3498db", "versicolor": "#2ecc71", "virginica": "#e74c3c"}
    g = sns.pairplot(
        df,
        hue="species",
        palette=palette,
        markers=["o", "s", "D"],
        plot_kws={"alpha": 0.8, "s": 45},
        diag_kind="kde",
        corner=False
    )
    g.fig.suptitle("Iris Dataset Exploratory Pairplot (Feature Distributions by Species)", y=1.02, fontsize=14, fontweight="bold")
    g.savefig(pairplot_path, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {pairplot_path}")

    # Feature Distributions & Correlation Matrix Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    
    # 1. Violin Plot of Petal Length
    sns.violinplot(ax=axes[0, 0], data=df, x="species", y="petal_length", hue="species", palette=palette, legend=False, inner="quartile")
    axes[0, 0].set_title("Petal Length Distribution across Species", fontweight="bold")
    axes[0, 0].set_ylabel("Petal Length (cm)")
    
    # 2. Violin Plot of Petal Width
    sns.violinplot(ax=axes[0, 1], data=df, x="species", y="petal_width", hue="species", palette=palette, legend=False, inner="quartile")
    axes[0, 1].set_title("Petal Width Distribution across Species", fontweight="bold")
    axes[0, 1].set_ylabel("Petal Width (cm)")

    # 3. Correlation Heatmap
    corr = df[feature_names].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1, 0], vmin=-1, vmax=1)
    axes[1, 0].set_title("Feature Pearson Correlation Matrix", fontweight="bold")

    # 4. Boxplot comparison for all features
    df_melted = pd.melt(df, id_vars=["species"], value_vars=feature_names, var_name="Feature", value_name="Measurement (cm)")
    sns.boxplot(ax=axes[1, 1], data=df_melted, x="Feature", y="Measurement (cm)", hue="species", palette=palette)
    axes[1, 1].set_title("Feature Measurements Comparison by Species", fontweight="bold")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].legend(title="Species", loc="upper right")

    plt.tight_layout()
    eda_summary_path = os.path.join(FIGURES_DIR, "02_eda_distributions_and_correlation.png")
    fig.savefig(eda_summary_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {eda_summary_path}")

    # ---------------------------------------------------------
    # 3. Train/Test Split (80/20) & Normalization
    # ---------------------------------------------------------
    print("\n[+] Preparing Train/Test Split and Feature Scaling...")
    X = df[feature_names]
    y = df["species"]

    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"    Total samples: {len(X)}")
    print(f"    Training samples (80%): {len(X_train)} ({y_train.value_counts().to_dict()})")
    print(f"    Testing samples (20%):  {len(X_test)} ({y_test.value_counts().to_dict()})")

    # Feature Scaling using StandardScaler
    scaler = StandardScaler()
    # Fit strictly on X_train to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("    StandardScaler applied: fit on train, transformed train & test.")
    print("    Sample unscaled train means:", np.round(X_train.mean().values, 3))
    print("    Scaled train means (approx 0):", np.round(X_train_scaled.mean(axis=0), 3))
    print("    Scaled train std dev (approx 1):", np.round(X_train_scaled.std(axis=0), 3))

    # ---------------------------------------------------------
    # 4. Train Models: Logistic Regression & Decision Tree
    # ---------------------------------------------------------
    print("\n[+] Training Classification Models...")
    
    # Model 1: Logistic Regression
    lr_model = LogisticRegression(
        max_iter=200,
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    lr_model.fit(X_train_scaled, y_train)
    print("    [1/2] Logistic Regression trained.")

    # Model 2: Decision Tree Classifier
    dt_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=3,
        random_state=RANDOM_STATE
    )
    dt_model.fit(X_train_scaled, y_train)
    print("    [2/2] Decision Tree Classifier trained.")

    # Visualize Decision Tree structure
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=dt_model.classes_,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax
    )
    ax.set_title("Trained Decision Tree Architecture (max_depth=3)", fontsize=13, fontweight="bold")
    dt_tree_path = os.path.join(FIGURES_DIR, "03_decision_tree_architecture.png")
    fig.savefig(dt_tree_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {dt_tree_path}")

    # ---------------------------------------------------------
    # 5. Model Evaluation & Benchmarking
    # ---------------------------------------------------------
    print("\n[+] Evaluating Models on Test Set...")

    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_dt = dt_model.predict(X_test_scaled)

    # Classification Reports
    print("\n" + "=" * 50)
    print("LOGISTIC REGRESSION CLASSIFICATION REPORT:")
    print("=" * 50)
    cr_lr = classification_report(y_test, y_pred_lr, digits=4)
    print(cr_lr)

    print("\n" + "=" * 50)
    print("DECISION TREE CLASSIFICATION REPORT:")
    print("=" * 50)
    cr_dt = classification_report(y_test, y_pred_dt, digits=4)
    print(cr_dt)

    # Benchmark metrics compilation
    models_dict = {
        "Logistic Regression": (lr_model, y_pred_lr),
        "Decision Tree": (dt_model, y_pred_dt)
    }

    metrics_list = []
    for name, (_, preds) in models_dict.items():
        metrics_list.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision (Macro)": precision_score(y_test, preds, average="macro"),
            "Recall (Macro)": recall_score(y_test, preds, average="macro"),
            "F1-Score (Macro)": f1_score(y_test, preds, average="macro"),
            "Precision (Weighted)": precision_score(y_test, preds, average="weighted"),
            "Recall (Weighted)": recall_score(y_test, preds, average="weighted"),
            "F1-Score (Weighted)": f1_score(y_test, preds, average="weighted")
        })

    benchmark_df = pd.DataFrame(metrics_list)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON SUMMARY TABLE:")
    print("=" * 80)
    print(benchmark_df.to_string(index=False))

    # Benchmark comparison plot
    metrics_to_plot = ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"]
    plot_df = benchmark_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5))
    bar = sns.barplot(data=plot_df, x="Metric", y="Score", hue="Model", palette=["#2980b9", "#27ae60"], ax=ax)
    ax.set_ylim(0.80, 1.02)
    ax.set_title("Benchmark Metric Comparison: Logistic Regression vs. Decision Tree", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score (0.0 to 1.0)")
    for p in bar.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom", fontsize=9, xytext=(0, 3), textcoords="offset points")

    benchmark_path = os.path.join(FIGURES_DIR, "04_benchmark_metrics_comparison.png")
    fig.savefig(benchmark_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {benchmark_path}")

    # ---------------------------------------------------------
    # 6. Plot Confusion Matrix Heatmaps
    # ---------------------------------------------------------
    print("\n[+] Plotting Confusion Matrix Heatmaps...")
    labels = np.unique(y)
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=labels)
    cm_dt = confusion_matrix(y_test, y_pred_dt, labels=labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Logistic Regression Heatmap
    sns.heatmap(
        cm_lr,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=axes[0],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[0].set_title(f"Logistic Regression Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_lr)*100:.1f}%", fontweight="bold")
    axes[0].set_xlabel("Predicted Class", fontweight="bold")
    axes[0].set_ylabel("Actual Class", fontweight="bold")

    # Decision Tree Heatmap
    sns.heatmap(
        cm_dt,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=axes[1],
        annot_kws={"size": 13, "weight": "bold"}
    )
    axes[1].set_title(f"Decision Tree Confusion Matrix\nAccuracy: {accuracy_score(y_test, y_pred_dt)*100:.1f}%", fontweight="bold")
    axes[1].set_xlabel("Predicted Class", fontweight="bold")
    axes[1].set_ylabel("Actual Class", fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "05_confusion_matrix_heatmaps.png")
    fig.savefig(cm_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {cm_path}")

    # ---------------------------------------------------------
    # 7. Decision Boundary Visualization (2D Feature Space)
    # ---------------------------------------------------------
    print("\n[+] Visualizing Decision Boundaries (Petal Length vs. Petal Width)...")
    # Features for 2D visualization
    feature_pair = ["petal_length", "petal_width"]
    X_2d = df[feature_pair].values
    
    # Map species labels to integers for contour rendering
    species_map = {name: idx for idx, name in enumerate(labels)}
    y_int = np.array([species_map[item] for item in y])

    X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
        X_2d, y_int, test_size=0.20, random_state=RANDOM_STATE, stratify=y_int
    )

    # Scale 2D features
    scaler_2d = StandardScaler()
    X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
    X_test_2d_scaled = scaler_2d.transform(X_test_2d)

    # Train 2D models
    lr_2d = LogisticRegression(max_iter=200, random_state=RANDOM_STATE).fit(X_train_2d_scaled, y_train_2d)
    dt_2d = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(X_train_2d_scaled, y_train_2d)

    # Create meshgrid
    x_min, x_max = X_train_2d_scaled[:, 0].min() - 0.8, X_train_2d_scaled[:, 0].max() + 0.8
    y_min, y_max = X_train_2d_scaled[:, 1].min() - 0.8, X_train_2d_scaled[:, 1].max() + 0.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    custom_cmap = plt.matplotlib.colors.ListedColormap(["#a1c9f4", "#b4e4c2", "#ffb3ba"])
    scatter_colors = ["#1f77b4", "#2ca02c", "#d62728"]
    markers = ["o", "s", "^"]

    for idx, (title, model) in enumerate([("Logistic Regression (Linear Boundaries)", lr_2d),
                                          ("Decision Tree (Orthogonal Axis Partitions)", dt_2d)]):
        ax = axes[idx]
        Z = model.predict(grid_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        ax.contourf(xx, yy, Z, alpha=0.45, cmap=custom_cmap)
        ax.contour(xx, yy, Z, colors="black", linewidths=0.8, linestyles="dashed")

        # Plot training data points
        for class_idx, class_name in enumerate(labels):
            mask = (y_train_2d == class_idx)
            ax.scatter(
                X_train_2d_scaled[mask, 0],
                X_train_2d_scaled[mask, 1],
                color=scatter_colors[class_idx],
                marker=markers[class_idx],
                s=40,
                alpha=0.6,
                edgecolors="k",
                label=f"Train: {class_name}"
            )

        # Plot testing data points
        for class_idx, class_name in enumerate(labels):
            mask = (y_test_2d == class_idx)
            ax.scatter(
                X_test_2d_scaled[mask, 0],
                X_test_2d_scaled[mask, 1],
                color=scatter_colors[class_idx],
                marker=markers[class_idx],
                s=90,
                alpha=0.95,
                edgecolors="black",
                linewidth=1.5,
                label=f"Test: {class_name}"
            )

        acc = accuracy_score(y_test_2d, model.predict(X_test_2d_scaled))
        ax.set_title(f"{title}\nTest Set Accuracy: {acc*100:.1f}%", fontsize=12, fontweight="bold")
        ax.set_xlabel("Petal Length (Standardized)", fontweight="bold")
        ax.set_ylabel("Petal Width (Standardized)", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    db_path = os.path.join(FIGURES_DIR, "06_decision_boundary_comparison.png")
    fig.savefig(db_path, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {db_path}")

    print("\n" + "=" * 75)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"All figures have been saved to: {FIGURES_DIR}")
    print("=" * 75)

if __name__ == "__main__":
    main()
