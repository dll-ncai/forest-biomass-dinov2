# Pakistani Dataset Analysis: Methodology Report

**Folder**: `/home/assad/biomass/pakistani_data_analysis`  
**Dataset**: Pakistani (Balakot) aerial imagery biomass — full DINO features (384), 25 samples.

This report describes the methodology applied to the Pakistani dataset, aligned with the German data workflow: feature–biomass correlation, feature ranking, UMAP visualization, and best-model selection over top-ranked features (20–250 in steps of 10).

---

## 1. Overview

1. **Feature–biomass correlation**: Pearson and Spearman correlation of each feature with biomass; features ranked by absolute correlation.
2. **UMAP visualization**: 2D and 3D UMAP of the feature space, colored by biomass and by train/test split.
3. **Best models by top features**: For each number of top-ranked features (20, 30, …, 250), we train **Linear Regression (Ridge)**, **XGBoost**, and **NNLS** and evaluate with **Leave-One-Out (LOO)** validation. Best configuration per method is chosen by lowest % RMSE.

All outputs are written in this folder.

---

## 2. Data and Inputs

- **Input features**: Full DINO features from `dino_features_with_labels_and_split.csv` (384 features: `feature_0` … `feature_383`), plus `label` and `split`.
- **Samples**: 25.
- **Label**: Biomass (raw units; no conversion applied in this analysis).
- **Feature ranking**: Obtained from the correlation analysis in this folder: `feature_correlations.csv` (sorted by absolute Pearson correlation with biomass).

---

## 3. Feature–Biomass Correlation and Ranking

### 3.1 Procedure

- For each feature, compute **Pearson** and **Spearman** correlation with the biomass label.
- Rank features by **absolute Pearson correlation** (descending).
- Outputs:
  - `feature_correlations.csv`: All features with correlations and p-values.
  - `correlation_summary.json`: Summary stats and top-10 list.
  - `top_features_correlation_analysis.png`, `top_10_features_detailed.png`, `correlation_distribution.png`: Visualizations.

### 3.2 Use of Ranking

The same ranking is used for the “top-N features” experiments: for each N in {20, 30, …, 250}, we take the **top N features** from this ranked list (up to the number of available features).

---

## 4. UMAP Visualization

- **Algorithm**: UMAP (n_neighbors=15, min_dist=0.1, random_state=42).
- **Input**: Full 384-dimensional feature vectors (or the same feature set used in the rest of the analysis).
- **Outputs**:
  - 2D: `umap_visualization.png`, `umap_biomass.png`, `umap_train_test.png`.
  - 3D: `umap_3d_biomass.png`, `umap_3d_train_test.png`, `umap_3d_multiple_views.png`.
- Points are coloured by biomass and by train/test (or train/val) split when available.

---

## 5. Best Models by Top-Ranked Features (20–250, Step 10)

### 5.1 Design

- **Feature counts**: 20, 30, 40, …, 250 (steps of 10). If the dataset has fewer than 250 features, the list is capped accordingly.
- **Per feature count**: Use only the **top N** features from the correlation-based ranking.
- **Validation**: **Leave-One-Out (LOO)** — each sample is left out once; model is trained on the remaining 24 and used to predict the held-out sample. Metrics are computed over all 25 folds.
- **Models**:
  - **Linear Regression (Ridge)** with scaling and fixed alpha (e.g. 10.0).
  - **XGBoost** with fixed hyperparameters (aligned with comparison_results where applicable).
  - **NNLS** (dictionary learning with non-negative least squares): k-nearest neighbors, cosine similarity, simplex constraint; fixed k (e.g. 3).

Hyperparameters are fixed (no inner tuning per feature count) so that the only variable is the number of top features.

### 5.2 Metrics

- **RMSE**, **% RMSE**, **MAE**, **R²** (all from LOO predictions).

### 5.3 Outputs

- **results_top_features.json** / **results_top_features_full.json**: LOO metrics (and optionally predictions) per feature count and per method.
- **results_top_features.csv**: Same results in tabular form (long format).
- **best_configurations.csv** / **best_configurations.json**: For each method, the **number of top features** that gave the **lowest % RMSE** (and corresponding R²).
- **Visualizations**:
  - **performance_vs_n_features.png**: Line plots of % RMSE and R² vs number of top features for each method.
  - **heatmap_percent_rmse.png**: Heatmap of % RMSE (n_features × method).
  - **heatmap_r2.png**: Heatmap of R² (n_features × method).
  - **best_configurations_table.png**: Table of best configuration per method.

---

## 6. Summary of Results

Best configuration per method (lowest LOO % RMSE over top-N features, N = 20–250 in steps of 10):

| Method              | Best n_features | % RMSE (LOO) | R² (LOO) |
|---------------------|-----------------|--------------|----------|
| Linear Regression   | 120             | 25.11        | 0.919    |
| XGBoost             | 20              | 59.49        | 0.548    |
| NNLS                | 250             | **20.16**    | **0.948**|

- **NNLS** achieves the best LOO performance (20.16% RMSE, R² = 0.948) with 250 top-ranked features.
- **Linear Regression** is best with 120 features (25.11% RMSE, R² = 0.919).
- **XGBoost** performs best with only 20 features (59.49% RMSE) and degrades with more features, suggesting overfitting or that fewer features suit this small sample.

Full results are in **best_configurations.csv**, **best_configurations.json**, and **results_top_features.csv**. The plots **performance_vs_n_features.png** and the heatmaps show how LOO % RMSE and R² vary with the number of top-ranked features.

---

## 7. Relation to German Workflow

- **German**: Step 2 uses top features (20–250, step 10), multiple cluster counts, train/test split, and cluster-specific models; Step 3 builds comparison visualizations and best-config tables.
- **Pakistani**: Same feature-count grid and same three methods (Linear, XGBoost, NNLS), but **no clustering** and **LOO** instead of a single train/test split, to suit the small sample size (25). All outputs are in this single folder.

---

## 8. Train/Test Comparison Figure (comparison_multiple_feature_counts.png)

A **train/test comparison** (like the German `comparison_multiple_feature_counts.png`) is produced by `run_train_test_comparison.py`:

- **Split**: Uses the dataset’s `split` column (train vs test/val) or an 80–20 random split.
- **Feature counts**: 20 to 300 in steps of 10.
- **Hyperparameter tuning**: Done **on the train set only** (e.g. 3-fold CV on train). Best config is then refit on full train and evaluated on **train** and **test**.
- **Figure**: One row of subplots (NNLS, LINEAR, XGB). Each subplot shows:
  - **Test %RMSE** (solid line) vs number of features.
  - **Train %RMSE** (dashed line) vs number of features.
  - Shaded **train–test gap**.
  - **Best test** point marked (star) and annotated.
- **Summary table**: For each method, best number of features (by test %RMSE), test %RMSE, **train %RMSE at that feature count**, test R², train R², and a **Note** when the train–test gap indicates overfitting.

**Why XGBoost can look “bad” or unstable**: With small data, XGBoost often fits the training set very well (train %RMSE near 0) while test %RMSE stays higher. The figure makes this **overfitting** visible: a large gap between train and test lines. Tuning on the train set (e.g. stronger regularization, shallower trees, fewer features) helps; the script tunes max_depth, reg_lambda, min_child_weight, etc., on the train set only, and comparison is always on the **test** set.

**To reproduce**:  
`python3 run_train_test_comparison.py --output_dir .`  
Outputs: `comparison_multiple_feature_counts.csv`, `comparison_multiple_feature_counts.png`.

---

## 9. Files in This Folder

| File | Description |
|------|-------------|
| **Feature correlation** | |
| feature_correlations.csv | Per-feature correlations and ranking |
| correlation_summary.json | Summary and top-10 features |
| top_features_correlation_analysis.png | Bar and scatter plots for top features |
| top_10_features_detailed.png | Scatter plots for top 10 vs biomass |
| correlation_distribution.png | Histograms of correlations |
| **UMAP** | |
| umap_visualization.png | 2D UMAP (4-panel) |
| umap_biomass.png, umap_train_test.png | 2D UMAP by biomass and split |
| umap_3d_*.png | 3D UMAP plots |
| **Best models (top features)** | |
| results_top_features.csv | LOO metrics per n_features and method |
| results_top_features.json | Same, short form |
| best_configurations.csv / .json | Best n_features per method |
| performance_vs_n_features.png | % RMSE and R² vs n_features |
| heatmap_percent_rmse.png, heatmap_r2.png | Heatmaps |
| best_configurations_table.png | Best-config table figure |
| **Train/test comparison** | |
| comparison_multiple_feature_counts.csv | Per (n_features, method, split) metrics |
| comparison_multiple_feature_counts.png | Train vs test %RMSE curves + summary table |
| run_train_test_comparison.py | Train/test run + figure (tune on train, compare on test) |
| **Scripts & report** | |
| feature_biomass_correlation_analysis.py | Correlation and ranking |
| visualize_with_umap.py | UMAP 2D/3D |
| find_best_models_top_features.py | Best models 20–250 features |
| run_analysis.sh | Run correlation + UMAP |
| README.md | Usage and paths |
| METHODOLOGY_REPORT.md | This document |

---

## 9. How to Reproduce

1. **Correlation and UMAP** (default: full-feature CSV in repo root):
   ```bash
   cd /home/assad/biomass/pakistani_data_analysis
   ./run_analysis.sh
   ```
2. **Best models by top features** (requires `feature_correlations.csv` from step 1):
   ```bash
   python3 find_best_models_top_features.py
   ```
   Optional: `--input_csv`, `--correlations_csv`, `--output_dir` to override paths.

---

**Report version**: 1.0  
**Dataset**: Pakistani (Balakot), 25 samples, 384 DINO features  
**Validation**: Leave-One-Out for model comparison and best-model selection
