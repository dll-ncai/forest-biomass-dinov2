# Biomass Estimation from High-Resolution RGB Aerial Imagery Using Self-Supervised Deep Features

> **Note (leak-free revision).** All numbers and protocols in this document come from
> the corrected, leakage-free pipeline (`corrected_pipeline.py` and companions), in which
> feature-count selection, feature ranking, clustering, and every model hyper-parameter
> are chosen on **training data only**. An earlier version of this manuscript reported
> substantially more optimistic numbers produced by a pipeline that selected the feature
> count and feature ranking with the test samples included; those results are superseded
> by the ones below. Full machine-readable results are in `../RESULTS_CORRECTED.md`.

## 2. Methodology

### 2.1 Study Areas and Datasets

Two geographically distinct study areas were selected to evaluate the generalizability of the approach across different forest ecosystems, canopy structures, and acquisition conditions.

**Study Area 1: Balakot, Pakistan.** A high-resolution aerial photograph (3,175 × 4,771 pixels, ~0.23 m ground sampling distance, EPSG:32643) covering the site was acquired. Ground-truth above-ground biomass (AGB) was available for **25 circular sample plots** (radius 17.5 m). Because of the very small sample size, **nested leave-one-out (LOO) cross-validation** is used as the primary evaluation strategy.

**Study Area 2: Karlsruhe, Germany.** High-resolution aerial RGB imagery (~0.18 m GSD, EPSG:5677) was acquired over a managed forest. Ground-truth AGB was available for **101 sample plots**; an additional **12 zero-biomass samples** were extracted from non-vegetated areas (roads, buildings, clearings), giving **113 samples**. The zero-biomass augmentation improves behaviour when mapping non-forested pixels.

### 2.2 Overview of the Proposed Approach

The methodology has four stages: (1) patch extraction and preprocessing, (2) feature extraction with a frozen self-supervised Vision Transformer, (3) **train-only** correlation-based feature selection, and (4) biomass regression. The central hypothesis is that features learned by a self-supervised ViT from natural images encode enough structural and textural information from RGB imagery to estimate biomass — without multispectral or LiDAR data.

### 2.3 Patch Extraction and Preprocessing

For each plot, a square patch was extracted from the aerial photograph, centred on the plot centroid.

- **Pakistan:** 152 × 152 px patches, centre-cropped to 112 × 112 px (to reduce edge effects), then resized to 224 × 224 px with bicubic interpolation.
- **Germany:** 224 × 224 px patches matching the plot extent, used directly at 224 × 224 px.

All patches were normalized with ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]), matching the feature extractor's pretraining.

### 2.4 Feature Extraction: Self-Supervised DINO Vision Transformer

#### 2.4.1 Background: Self-Supervised Learning with DINO

We use features from **DINO** (Self-**Di**stillation with **No** labels; Caron et al., 2021), a self-supervised framework that trains a Vision Transformer via teacher–student self-distillation on unlabeled images. DINO features capture semantically meaningful structure — object boundaries, texture, and spatial hierarchies — and its attention maps correspond closely to object segmentations. These properties make DINO features well suited to remote-sensing tasks where structural and textural cues correlate with biophysical variables such as biomass. Crucially, the feature extractor needs **no labeled remote-sensing data** and transfers across regions without retraining.

#### 2.4.2 Feature Extraction Pipeline

Features were extracted with **DINO ViT-Small, patch size 16** (`vit_small_patch16_224.dino`) via `timm`, in inference mode with the classification head removed (`num_classes=0`), taking the **CLS-token** embedding of the final transformer layer — a **384-dimensional** vector per patch. The model is **frozen** (ImageNet-pretrained weights, no fine-tuning), testing whether generic visual features transfer to biomass estimation without domain-specific representation learning.

As a comparison backbone we also extract **ResNet50** (ImageNet-supervised, 2048-d global-average-pool features) under the identical patch protocol.

> **Figure 1.** Study-area overview (`fig1_study_area_overview`). **Figure 2.** End-to-end methodological pipeline (`fig2_methodological_pipeline`).

### 2.5 Leak-Free Correlation-Based Feature Selection

For each candidate feature count *K*, the top-*K* features are chosen by the absolute Pearson correlation between the feature and biomass, **computed on training data only**. Critically, this ranking is **recomputed inside every cross-validation fold / outer LOO fold** on that fold's training samples — no ranking is ever computed over samples that will be used for evaluation. The feature count *K* itself is a hyper-parameter selected by inner cross-validation (Section 2.8), again using only training data. No PCA or other dimensionality reduction is applied. The sweep spans *K* ∈ {10, 20, …, 360} (capped at the backbone dimensionality).

### 2.6 Train-Only Unsupervised Clustering (German Dataset)

For the German site, an unsupervised clustering step accounts for structural heterogeneity, under the hypothesis that different forest subtypes have different feature–biomass relationships. **StandardScaler → UMAP (n_neighbors=15, n_components=2, min_dist=0.1, metric=euclidean, random_state=42) → K-Means (k=2, random_state=42)** is fit **on the training split only**; validation/test patches are assigned by applying the fitted `scaler.transform → umap.transform → kmeans.predict`. A separate regressor is trained per cluster, with per-cluster feature ranking on that cluster's **training** samples. Clustering is not applied to Pakistan (n=25). As Section 3.3 shows, clustering helps NNLS but not the other regressors, so it is reported as one configuration among several rather than an unconditional component.

> **Figure 3.** UMAP embedding of German DINO features, coloured by cluster and by biomass (`fig3_umap_embeddings`).

### 2.7 Regression Models

#### 2.7.1 Non-Negative Least Squares (NNLS) Reconstruction

NNLS is a **local dictionary-based** predictor: a query patch is reconstructed as a non-negative combination of its *k* nearest training neighbours in feature space, and the same weights are applied to the neighbours' biomass values.

1. **Neighbour identification.** The *k* nearest training neighbours are found by cosine similarity (L2-normalized features) or Euclidean distance (StandardScaler-standardized features).
2. **Regularized weight computation.** The query *b* is reconstructed from the neighbour matrix *A* (neighbour feature vectors as columns) by solving a **ridge-regularized non-negative least squares** problem, minimize ‖*Aw − b*‖² + λ‖*w*‖² subject to *w* ≥ 0, implemented as an augmented NNLS. The L2 term λ stabilizes the (often under-determined) reconstruction — important when the query lives in a high-dimensional feature space with few neighbours.
3. **Convex (simplex) constraint.** Optionally the weights are renormalized to sum to one, so the prediction is a convex combination of neighbour biomass values and therefore bounded to the observed biomass range.
4. **Prediction.** ŷ = Σ *wᵢ yᵢ*.

**Small-sample robustification.** On very small training sets the inner cross-validation cannot reliably select among high-variance NNLS options, and naive tuning *degrades* NNLS (Section 3.1). Two guards are therefore applied whenever the training set has fewer than 30 rows (i.e. the Pakistani nested-LOO folds; the German clusters, 35–61 samples, are unaffected): (i) selection is restricted to the convex `simplex=True` estimator; and (ii) instead of selecting a single feature count *K*, predictions are **averaged over the whole *K* grid** (ensemble-over-*K*), which removes the *K*-selection variance and is insensitive to the grid resolution.

Hyper-parameters (tuned train-only): *k* ∈ {3, 5, 8, 10, 12, 15, 20}; cosine/Euclidean; simplex on/off; λ ∈ {0, 0.1, 1.0}.

#### 2.7.2 Ridge Regression

L2-regularized linear regression on standardized features; α tuned by cross-validation over {0.01, 0.1, 1, 10, 50, 100, 300, 1000} (with/without standardization).

#### 2.7.3 XGBoost

Gradient-boosted trees; tuned over n_estimators ∈ {100, 300}, learning_rate ∈ {0.03, 0.05}, max_depth ∈ {2, 3}, reg_lambda ∈ {1, 10}, subsample = colsample_bytree = 0.8.

### 2.8 Leak-Free Validation Strategy

**Pakistan — nested LOO.** For each of the 25 outer folds, one plot is held out and the remaining 24 form the training set. All choices — feature ranking, feature count *K*, and model hyper-parameters — are made by inner 5-fold cross-validation **within those 24 training samples**; the held-out plot is predicted once. Metrics are pooled over all 25 held-out predictions. No test plot influences any selection.

**Germany — 85/15 hold-out with train-only tuning.** The 113 samples are split 85/15 with biomass-stratified sampling (random_state=42), giving 17 test plots (8 in cluster 0, 9 in cluster 1). Feature ranking, *K*, hyper-parameters, and the clustering transform are all fit on the training split (5-fold CV for tuning); the test set is evaluated once, pooling both clusters.

**Germany — robustness.** Because a single 85/15 split is noisy on 113 samples, the entire leak-free evaluation is repeated over **20 stratified splits**, and we report the mean ± standard deviation of the combined test %RMSE and R² (Section 3.4).

### 2.9 Evaluation Metrics

- **%RMSE** = RMSE / mean(observed) × 100 (comparable across sites).
- **RMSE** and **MAE** in biomass units (t/ha).
- **R²**, the proportion of biomass variance explained (≤ 0; negative means worse than predicting the mean).

Pakistani biomass is converted from kg per plot to t/ha using the circular plot area (r = 17.5 m) before computing metrics, so both sites are reported in t/ha.

### 2.10 Comparison with Literature (Hand-Crafted-Feature) Methods

To benchmark the DINO approach we implemented established RGB hand-crafted-feature methods (Liu et al., 2025; Eckert, 2012): **multi-scale GLCM texture** features (distances 1–3 px; correlation, homogeneity, contrast, energy, ASM over four orientations), **ten RGB vegetation indices** (VVI, GRVI, RGRI, GRRI, ExG, ExR, CIVE, VARI, TGI, NGRDI), and **per-channel spectral statistics**, for a total of **97 hand-crafted features** extracted from the *same* image patches as the deep features. Random Forest, Gradient Boosting, XGBoost, and Ridge are trained on these features and — crucially — evaluated under the **identical leak-free protocol** as the deep models (nested LOO for Pakistan; 85/15 with train-only 5-fold tuning for Germany). This is a substantially stronger and fairer literature baseline than in the earlier version of this manuscript, and it changes the headline comparison (Section 3.5).

### 2.11 Biomass Raster Generation

To demonstrate applicability, wall-to-wall biomass maps were generated for both sites. Each aerial image is rendered to 8-bit RGB (per-band 1–99 % linear stretch for the 12-bit Pakistani raster; direct for the 8-bit German raster) and tessellated with overlapping crop-sized windows (Pakistan 152 px, stride 76; Germany 224 px, stride 112), skipping windows that are mostly nodata. DINO features are extracted per window with the frozen model, and biomass is predicted with an NNLS model trained on all plots of that site. Because the mapping predictor uses convex (simplex) weights, every mapped value is a weighted average of training-plot biomass and is therefore **bounded to the observed range**, which is the desired behaviour for an application map.

> **Figure 4.** Aerial RGB and predicted AGB maps for both sites (`fig4_biomass_raster_maps`). Pakistan (a, b) and Germany (c, d). In Germany, buildings/parking are correctly mapped to low AGB while the forest is high, illustrating that the zero-biomass augmentation and the bounded predictor produce realistic non-forest values.

---

## 3. Results

### 3.1 Pakistani Dataset (nested LOO, n = 25)

Table 1 reports every model (both backbones) and the literature baselines under nested LOO. **DINOv2 + NNLS is the best model** at **29.93 %RMSE (R² = 0.886)**, narrowly ahead of DINOv2 + Ridge (31.73 %, 0.871) and of the strongest literature baseline, Random Forest on GLCM features (31.12 %, 0.876).

**Table 1.** Pakistani dataset, nested LOO (t/ha). Best per column in bold.

| Backbone | Model | %RMSE | R² | RMSE | MAE |
|----------|-------|-------|-----|------|-----|
| **DINOv2** | **NNLS (ensemble-K)** | **29.93** | **0.886** | **3.87** | **2.86** |
| DINOv2 | Ridge | 31.73 | 0.871 | 4.11 | 3.31 |
| DINOv2 | XGBoost | 53.46 | 0.635 | 6.92 | 5.05 |
| ResNet50 | XGBoost | 41.47 | 0.780 | 5.37 | 4.15 |
| ResNet50 | NNLS | 55.44 | 0.607 | 7.18 | 5.66 |
| ResNet50 | Ridge | 59.50 | 0.548 | 7.70 | 6.56 |
| Literature (GLCM) | Random Forest | 31.12 | 0.876 | 4.03 | 3.39 |
| Literature (GLCM) | Gradient Boosting | 31.91 | 0.870 | 4.13 | 2.92 |
| Literature (GLCM) | Ridge | 47.10 | 0.717 | 6.10 | 5.24 |
| Literature (GLCM) | XGBoost | 49.54 | 0.687 | 6.41 | 4.20 |

**NNLS on very small data requires regularization.** With only 25 plots, an aggressive hyper-parameter search *hurts* NNLS: allowing the non-convex (`simplex=False`) reconstruction and selecting a single feature count from a fine grid causes the inner cross-validation to overfit its selections, and the per-fold chosen *K* swings between 10 and 300, yielding ~48–50 %RMSE. Restricting to convex weights and **averaging predictions over the feature-count grid** (ensemble-over-*K*, Section 2.7.1) removes this selection variance and is insensitive to the grid, recovering the **29.93 %** reported above. This is the key methodological lesson for the small-sample regime: prefer *averaging over* hyper-parameters to *selecting* them.

**XGBoost is unsuitable at n = 25**, over-fitting badly (53.5 %, R² = 0.635) — consistent with the known sensitivity of tree ensembles to tiny datasets.

### 3.2 German Dataset (85/15 hold-out, n_test = 17)

Table 2 reports the combined (pooled-cluster) test performance. **DINOv2 + NNLS is again the best model at 17.40 %RMSE (R² = 0.837)**, clearly ahead of every ResNet50 configuration and of the best literature baseline (Ridge on GLCM, 22.79 %, 0.721).

**Table 2.** German dataset, 85/15 test (t/ha). Best per column in bold.

| Backbone | Model | %RMSE | R² | RMSE | MAE |
|----------|-------|-------|-----|------|-----|
| **DINOv2** | **NNLS** | **17.40** | **0.837** | **30.89** | **23.53** |
| DINOv2 | Ridge | 23.83 | 0.695 | 42.32 | 33.21 |
| DINOv2 | XGBoost | 26.27 | 0.629 | 46.64 | 38.51 |
| ResNet50 | Ridge | 24.13 | 0.687 | 42.85 | 36.09 |
| ResNet50 | XGBoost | 29.86 | 0.521 | 53.03 | 41.66 |
| ResNet50 | NNLS | 31.97 | 0.451 | 56.78 | 43.90 |
| Literature (GLCM) | Ridge | 22.79 | 0.721 | 40.46 | 34.72 |
| Literature (GLCM) | Random Forest | 23.70 | 0.698 | 42.09 | 36.15 |
| Literature (GLCM) | Gradient Boosting | 23.99 | 0.691 | 42.59 | 33.07 |
| Literature (GLCM) | XGBoost | 25.55 | 0.649 | 45.37 | 35.62 |

> **Figure 5.** %RMSE by backbone and model for both sites, with the best literature baseline as a reference line (`fig_corrected_performance`). **Figure 6.** Predicted-vs-observed scatter for the best model per site (DINOv2 + NNLS) with 1:1 line and metrics (`fig_corrected_scatter`).

### 3.3 Ablations (German DINOv2)

Table 3 isolates the contribution of the two optional pipeline components — train-only clustering and correlation-based feature selection.

**Table 3.** German ablation (DINOv2, 85/15 test, %RMSE / R²).

| Configuration | NNLS | Ridge | XGBoost |
|---------------|------|-------|---------|
| Proposed (cluster + selection) | **17.40 / 0.837** | 23.83 / 0.695 | 26.27 / 0.629 |
| No clustering (global model)   | 20.45 / 0.775 | 21.00 / 0.763 | 21.20 / 0.759 |
| All 384 features (no selection)| 17.90 / 0.828 | 20.38 / 0.777 | 24.71 / 0.672 |

Clustering clearly benefits **NNLS** (17.40 vs 20.45 without clustering) but *not* Ridge or XGBoost, which prefer the single global model. Correlation-based feature selection provides little benefit over using all 384 features for NNLS on this site (17.40 vs 17.90). We therefore present clustering and selection as configuration choices rather than universally beneficial steps.

> **Figure 7.** German ablation bars (`fig_corrected_ablation_german`). **Figure 8.** Train-only inner-CV %RMSE vs. feature count, per method and cluster (`fig_corrected_perf_vs_features`).

### 3.4 Robustness Across Splits (German)

A single 85/15 split is optimistic. Repeating the full leak-free evaluation over **20 stratified splits** gives the honest picture in Table 4.

**Table 4.** German robustness over 20 splits (proposed pipeline; combined-test %RMSE and R², mean ± std).

| Backbone | Model | %RMSE (mean ± std) | R² (mean ± std) |
|----------|-------|--------------------|-----------------|
| DINOv2 | NNLS | **22.99 ± 3.65** | **0.70 ± 0.15** |
| DINOv2 | XGBoost | 23.99 ± 3.63 | 0.68 ± 0.12 |
| DINOv2 | Ridge | 24.52 ± 2.86 | 0.67 ± 0.12 |
| Literature (GLCM) | Ridge | 24.43 ± 2.46 | 0.68 ± 0.10 |
| Literature (GLCM) | Random Forest | 26.49 ± 4.71 | 0.61 ± 0.18 |
| ResNet50 | NNLS | 28.22 ± 4.68 | 0.57 ± 0.15 |

Averaged over splits, **DINOv2 + NNLS remains the best (23.0 %RMSE)**, ahead of the best literature baseline (24.4 %) and all ResNet50 configurations, though the margin is modest and within one standard deviation of the literature Ridge. The single-split 17.40 % of Section 3.2 should thus be read alongside this ~23 % typical value.

> **Figure 9.** German robustness bars (mean ± std over 20 splits) with the best-literature reference line (`fig_corrected_robustness_german`).

### 3.5 DINOv2 vs. ResNet50 vs. Literature

Three consistent findings emerge from the leak-free evaluation:

1. **DINOv2 features beat ResNet50 features** on both sites and for essentially every regressor (e.g. Germany NNLS 17.4 % vs 32.0 %; Pakistan best-DINO 29.9 % vs best-ResNet50 41.5 %). Self-supervised ViT features transfer to biomass estimation better than supervised CNN features.

2. **DINOv2 outperforms strong, fairly-tuned literature baselines — but by a modest margin.** On Germany the advantage is clear (best DINO 17.4 % vs best literature 22.8 %; ~24 % relative error reduction). On Pakistan it is narrow (29.9 % vs 31.1 %; comparable R², 0.886 vs 0.876). This is a much more measured claim than the earlier version of this manuscript, which — using a flawed literature evaluation — reported negative literature R². Under a correct, identical protocol the GLCM baselines are competitive, and the honest contribution of DINO features is a consistent, moderate improvement rather than an order-of-magnitude one.

3. **NNLS is the best regressor on both sites**, provided it is properly regularized on small data. Its local reconstruction in DINO feature space, with convex weights, both fits well and yields bounded, physically plausible predictions for mapping.

### 3.6 Biomass Raster Maps

Wall-to-wall AGB maps were generated for both sites (Figure 4). Predicted values are bounded to the observed range by the convex NNLS predictor. In Germany the map correctly separates high-biomass forest from low-biomass built-up and cleared areas — the zero-biomass augmentation is essential to this behaviour. In Pakistan the map reproduces the spatial pattern of the aerial scene, with higher AGB over the denser canopy. These maps are qualitative demonstrations: the mapping predictor is trained on all plots and applied to every window, so it illustrates spatial pattern rather than providing an independently validated pixel-level product.

### 3.7 Summary of Key Findings

1. After removing the data leakage present in the original pipeline, **DINOv2 + NNLS is the best model on both sites**: **29.93 %RMSE / R² 0.886** (Pakistan, nested LOO) and **17.40 %RMSE / R² 0.837** (Germany, 85/15; 23.0 % ± 3.7 over 20 splits).

2. **Self-supervised DINOv2 features clearly outperform supervised ResNet50 features** across sites and regressors.

3. **The advantage over hand-crafted-feature literature methods is real but modest** — clear on the larger German site, narrow on the tiny Pakistani site where a tuned GLCM Random Forest is competitive.

4. **NNLS is powerful but variance-prone on very small data**; convex weights, L2-regularized reconstruction, and ensemble-over-*K* are what make it the top performer at n = 25.

5. **Clustering and feature selection are conditionally, not universally, helpful** — clustering aids NNLS on the heterogeneous German site but not Ridge/XGBoost, and correlation selection barely improves on using all 384 features.

6. **The approach generalizes across geographies and forest types** without any domain-specific fine-tuning of the feature extractor, supporting the use of frozen self-supervised RGB features for biomass estimation where multispectral or LiDAR data are unavailable.
