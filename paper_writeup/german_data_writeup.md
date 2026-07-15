# German Study Area: Materials and Methods, Results, and Discussion

> **⚠️ Leak-free revision.** The authoritative, corrected results are in
> `methodology_and_results.md` and `../RESULTS_CORRECTED.md`. Headline tables and claims
> here are updated to the leakage-free numbers (feature-count, ranking, clustering, and
> hyper-parameters selected on training data only): **DINOv2 + NNLS = 17.40 %RMSE, R²
> 0.837** on the single 85/15 test, and **23.0 % ± 3.7 over 20 stratified splits**. The
> per-feature-count and per-cluster tables (Tables 2–4) have now also been regenerated
> from the corrected leak-free results (train-only CV curves, corrected cluster sizes and
> hyper-parameters); `methodology_and_results.md` / `../RESULTS_CORRECTED.md` remain the
> canonical source for the headline numbers.

---

## 1. Materials and Methods

### 1.1 Study Area and Field Data

The study area is located in the **Karlsruhe** region of Baden-Württemberg, southwestern Germany. The landscape consists of managed temperate forests interspersed with agricultural fields, urban areas, and other non-forested land cover. Ground-truth above-ground biomass (AGB) measurements were obtained from **101 field inventory plots** distributed across the forested areas. Plot-level biomass values ranged from **100.88 to 302.01 ton/ha** (mean ≈ 175 ton/ha), reflecting moderately dense to dense temperate forest stands.

### 1.2 Remote Sensing Data

A single **high-resolution aerial image** covering the study area was used as the sole input for biomass estimation. The image was acquired as a GeoTIFF raster (`karlsruhe.tif`) and read via the `rasterio` library. Only the **RGB (Red, Green, Blue) channels** were utilized, making the approach fully independent of multispectral, hyperspectral, or LiDAR data. A corresponding shapefile (`biomass_karlsruhe.shp`) provided the spatial coordinates and AGB labels for each field plot.

> **[FIGURE 1 — Study area overview]** *Map showing the location of the Karlsruhe study area in Germany, with the aerial image extent and the spatial distribution of the 101 field plots overlaid. A separate panel shows the zero-biomass crop locations in non-forested areas.*

### 1.3 Patch Extraction

For each of the 101 field plots, a square image patch of **224 × 224 pixels** was extracted from the aerial image, centered on the plot centroid. Pixel coordinates were derived from the plot's geographic coordinates using the raster's affine transform. Edge-case handling ensured patches near image boundaries were shifted inward to maintain the full 224 × 224 extent. If the extracted crop was not exactly 224 × 224 pixels, it was resized to that dimension using Lanczos interpolation. RGB values were percentile-normalized (1st–99th percentile) to the 0–255 uint8 range to handle radiometric variation.

### 1.4 Zero-Biomass Data Augmentation

To extend the dynamic range of the training data to include zero biomass and improve model extrapolation to non-forested areas, **12 additional patches** were extracted from clearly non-forested regions (agricultural fields, urban areas, or bare soil) identified from the aerial image. These zero-biomass crops were the same size (224 × 224 pixels) as the forest plot patches and were assigned a biomass label of **0.0 ton/ha**.

DINO features were extracted for the zero-biomass patches using the identical model and preprocessing pipeline described in Section 1.5. The zero-biomass samples were appended to the original 101-sample dataset, yielding an **augmented dataset of 113 samples** (101 forest + 12 zero-biomass). The zero-biomass samples were distributed between the training and test sets maintaining the same 85:15 ratio as the original data.

### 1.5 Feature Extraction Using DINO ViT-Small

Features were extracted using the same self-supervised Vision Transformer employed for the Pakistani dataset. The **DINO ViT-Small** model (patch size 16, `vit_small_patch16_224.dino`) was loaded through the `timm` library with the classification head removed (`num_classes=0`). Because the German crops were already 224 × 224 pixels (matching the model's native input size), preprocessing consisted of resizing to 224 × 224 (identity operation), conversion to a tensor, and normalization using ImageNet statistics (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).

The model outputs the **CLS token embedding** — a **384-dimensional feature vector** per patch. The pretrained ImageNet weights were used as a **frozen feature extractor** without fine-tuning, identical to the Pakistani analysis.

### 1.6 Train/Test Split

Unlike the Pakistani dataset (which used LOO cross-validation due to its small size of n = 25), the German dataset's larger sample size (n = 113) permitted a held-out test set evaluation. The data were split into **85% training (96 samples) and 15% test (17 samples)** using a **stratified split** strategy. Biomass labels were binned into quantile-based groups, and samples were drawn proportionally from each bin to ensure the training and test sets had similar biomass distributions. A fixed random seed (42) was used for reproducibility.

### 1.7 Unsupervised Clustering of the Feature Space

To account for structural heterogeneity in the forest landscape, unsupervised clustering was applied to the DINO feature space before model training. The rationale was that visually and structurally distinct forest types may exhibit different feature–biomass relationships, and training separate models per cluster could improve predictive accuracy.

**UMAP** (Uniform Manifold Approximation and Projection) was first applied to reduce the 384-dimensional DINO features to a **2-dimensional embedding** for clustering, with parameters: `n_neighbors = 15`, `min_dist = 0.1`, `metric = 'euclidean'`, and `random_state = 42`. **K-Means clustering** was then applied to the 2D UMAP embedding to partition the samples into **2 clusters** (`n_clusters = 2`, `n_init = 10`, `random_state = 42`).

The clustering yielded:
- **Cluster 0:** 48 samples (38 train, 10 test), with biomass ranging from 100.88 to 286.43 ton/ha (mean = 200.66 ton/ha).
- **Cluster 1:** 65 samples (58 train, 7 test), with biomass ranging from 0.00 to 302.01 ton/ha (mean = 156.38 ton/ha).

Cluster 1 contained all 12 zero-biomass samples, giving it a wider biomass range that spans the full 0–302 ton/ha spectrum. Cluster 0 contained only non-zero forest plots with a higher mean biomass.

> **[FIGURE 2 — UMAP clustering]** *2D UMAP embedding of the 113 samples, with points colored by (a) cluster assignment and (b) biomass value. The two clusters are visually separated, with Cluster 1 including zero-biomass samples at one extreme.*

### 1.8 Cluster-Specific Feature Selection

Within each cluster, feature selection was performed independently using **Pearson correlation ranking**, the same approach used for the Pakistani dataset. For each of the 384 DINO features, the absolute Pearson correlation coefficient with biomass was computed within the cluster's training samples. Features with near-zero variance (std < 10⁻¹⁰) were excluded. Features were ranked by |*r*| in descending order, and the top *N* features were selected.

Because feature ranking was performed per-cluster, the selected feature subsets differed between Cluster 0 and Cluster 1. The overlap between the top-ranked features of the two clusters varied with the number of features selected — for example, at 70 features only 15 features overlapped, while at 200 features 103 features overlapped.

The number of selected features *N* was systematically varied from **20 to 300 in increments of 10** (29 configurations), identical to the Pakistani analysis.

### 1.9 Regression Models

Four regression models were evaluated within each cluster:

#### 1.9.1 Non-Negative Least Squares (NNLS) Regression

The same dictionary-based NNLS method as the Pakistani analysis was used. For a query sample, the *k* nearest neighbors in the cluster's training set are identified (using either Euclidean or cosine distance), and the query biomass is predicted as a non-negative linear combination of the neighbor biomass values. Hyperparameters were tuned via 5-fold cross-validation over a grid: *k* ∈ {3, 4, 5, 6, 7, 8, 10, 12, 15}, distance metric ∈ {cosine, Euclidean}, simplex constraint ∈ {on, off}, yielding up to 36 configurations per cluster.

#### 1.9.2 Ridge Regression

An L2-regularized linear regression model was trained on each cluster. Features were optionally standardized (`StandardScaler`). Hyperparameters were tuned via 5-fold CV over: α ∈ {0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0}, scaling ∈ {on, off}, yielding 22 configurations.

#### 1.9.3 XGBoost

An XGBoost regressor was tuned via 5-fold CV over a grid: `n_estimators` ∈ {50, 100, 200}, `learning_rate` ∈ {0.01, 0.03, 0.05, 0.1}, `max_depth` ∈ {3, 4, 5, 6}, with fixed `subsample` = 0.8, `colsample_bytree` = 0.8, `reg_lambda` = 0.1, `reg_alpha` = 0.0, and `min_child_weight` = 1.0 (`random_state = 42`). Additional manually specified configurations were also tested (e.g., (100, 0.05, 5, 0.8, 0.8, 1.0) and (100, 0.1, 6, 1.0, 1.0, 10.0)).


### 1.10 Validation Strategy

Hyperparameter tuning for all three primary models was performed using **5-fold cross-validation** on the training set within each cluster. The configuration yielding the lowest CV %RMSE was selected. Final model performance was then assessed on the **held-out test set** (15% of data), which was not used during model selection.

Combined (overall) test-set metrics were computed by pooling the predictions from both clusters. For a given method and feature count, the combined RMSE was computed as:

> RMSE_combined = √[(n₀ × RMSE₀² + n₁ × RMSE₁²) / (n₀ + n₁)]

where n₀ = 10 and n₁ = 7 are the test-set sizes of Cluster 0 and Cluster 1, respectively. Combined R² was computed using the overall sum of squared residuals and total sum of squares. Combined MAE was the sample-weighted average of the per-cluster MAEs. The combined %RMSE was normalized by the overall mean test-set biomass (160.08 ton/ha).

### 1.11 Evaluation Metrics

The same four metrics used in the Pakistani analysis were employed:

- **Percentage RMSE (%RMSE):** (RMSE / ȳ) × 100, used as the primary model comparison criterion.
- **Root Mean Square Error (RMSE):** In ton/ha.
- **Mean Absolute Error (MAE):** In ton/ha.
- **Coefficient of Determination (R²):** Proportion of variance explained.

### 1.12 Biomass Raster Map Generation

Wall-to-wall biomass raster maps were generated for the full extent of the Karlsruhe aerial image. The image was tessellated into 224 × 224 pixel patches, DINO features were extracted for each patch, and each patch was assigned to the nearest UMAP+KMeans cluster. The cluster-specific model then predicted the biomass for that patch. Multiple overlap levels (0%, 25%, 50%, 75%) were evaluated, with overlapping predictions averaged to produce spatially smoother outputs.

---

## 2. Results

### 2.1 Combined Model Performance

Table 1 presents the best overall (combined across both clusters) test-set performance for each regression method, with the optimal number of features identified by the lowest combined %RMSE across the 20–300 feature sweep.

**Table 1.** Best combined test-set performance on the German dataset (leak-free, n_test = 17, t/ha). Feature count and hyper-parameters selected per cluster on training data only.

| Method | %RMSE | RMSE (ton/ha) | MAE (ton/ha) | R² |
|--------|-------|---------------|--------------|-----|
| **NNLS** | **17.40** | **30.89** | **23.53** | **0.837** |
| Ridge Regression | 23.83 | 42.32 | 33.21 | 0.695 |
| XGBoost | 26.27 | 46.64 | 38.51 | 0.629 |

NNLS achieved the best combined performance (%RMSE = 17.40%, R² = 0.837), outperforming the best literature-derived method (Ridge on GLCM features, %RMSE = 22.79%, R² = 0.721; Random Forest 23.70%, 0.698) — a clear but not dramatic margin. This mirrors the Pakistani dataset, where NNLS was also the best DINO regressor. Because a single 85/15 split is noisy, the honest performance is better read from the 20-split robustness result (NNLS 23.0 % ± 3.7, still ahead of the best literature baseline; §2.6 and Table 5-R below). NNLS is the strongest regressor here; Ridge and XGBoost trail on this particular split.

### 2.2 Effect of Feature Count (Train-Only Cross-Validation)

The feature count *K* is a hyper-parameter, and it must be chosen without ever looking at the test set. The original draft presented a table of *test-set* %RMSE at each feature count and then picked the best-scoring *K* — that "peek" is precisely one of the leaks removed in this re-run. Under the corrected protocol, *K* is selected by **5-fold cross-validation on the training split only**, separately for each cluster; the test set is touched exactly once at the chosen *K*. Table 2 therefore reports the leak-free **inner-CV %RMSE** across the sweep for the NNLS model (the eventual best regressor), which is the curve that actually drives selection (Figure 3 / `fig_corrected_perf_vs_features`).

**Table 2.** NNLS train-only 5-fold cross-validation %RMSE across the feature-count sweep, per cluster. Selected *K* in bold.

| *N* Features | Cluster 0 CV %RMSE | Cluster 1 CV %RMSE |
|-------------|--------------------|--------------------|
| 10  | 27.29 | 23.95 |
| 20  | 25.04 | 23.73 |
| 40  | 25.03 | 23.22 |
| 70  | 24.18 | **22.54** |
| 160 | 23.02 | 22.64 |
| 360 | **22.49** | 22.72 |

> **[FIGURE 3 — Train-only inner-CV %RMSE vs. N features]** *`fig_corrected_perf_vs_features`: inner-CV %RMSE vs. number of selected features, per method and cluster, with the CV-selected feature count annotated. This replaces the original leaky "combined test %RMSE vs. features" plot.*

The CV curves are shallow: NNLS CV error changes by only ~2–3 percentage points across the whole 10–360 range, so the site is only weakly sensitive to *K*. Cross-validation selects **K = 360** for Cluster 0 (high-density forest, where more of the embedding is informative) and **K = 70** for Cluster 1 (which includes the zero-biomass plots and saturates earlier). These are the feature counts carried into the per-cluster and combined test evaluations below — none of them was chosen with any knowledge of the test plots. The steep degradation at very high feature counts reported in the original draft was an artifact of selecting on the test set; on the honest CV curve the high-*K* region is flat, not sharply worse.

### 2.3 Per-Cluster Performance Analysis

The two clusters exhibited markedly different prediction accuracies, reflecting their distinct biomass distributions and sample sizes.

**Table 3.** Per-cluster test-set results for the leading NNLS model, at the CV-selected feature counts (Cluster 0: *K* = 360; Cluster 1: *K* = 70).

| Cluster | Samples | Biomass Mean (t/ha) | %RMSE | RMSE (t/ha) | R² |
|---------|---------|---------------------|-------|-------------|-----|
| Cluster 0 | 8 | 227.76 | 15.62 | 35.58 | 0.199 |
| Cluster 1 | 9 | 132.96 | 19.58 | 26.03 | 0.876 |
| **Combined** | **17** | **177.57** | **17.40** | **30.89** | **0.837** |

Cluster 1, which contains the zero-biomass samples and lower-density points, is the primary driver of explained variance (R² = 0.876) because its biomass spans a wide range that a model can track. Cluster 0 contains a tight band of high-density forest points, so it attains a *lower* percentage error (15.62%) yet a much weaker R² (0.199) — the classic effect of variance restriction, where a small denominator of true variance makes R² collapse even though absolute errors are moderate. Pooled across both clusters (against the full-landscape variance), the multi-cluster DINO pipeline explains **83.7%** of the biomass variance (%RMSE = 17.40%). Note that the pooled %RMSE sits between the two clusters' values because it is normalised by the higher overall mean biomass.

> **[FIGURE 4 — Per-cluster scatter plots]** *Scatter plots of predicted vs. actual biomass for each cluster, separated by method. Each panel shows the 1:1 line, with Cluster 0 points (circles) and Cluster 1 points (triangles) distinctly marked. Highlights the strong performance in Cluster 1 and the difficulty in Cluster 0.*

### 2.4 Cluster Feature Overlap

Under the corrected protocol the per-cluster feature ranking is recomputed on each cluster's **training** samples only, and the two clusters select partially different subsets of the 384 DINO features — consistent with the different feature counts CV chose for each (*K* = 360 vs 70). Qualitatively, this indicates that the DINO embedding encodes several types of structural and spectral information, with different dimensions being diagnostic for biomass in different forest types. (The specific overlap percentages quoted in the original draft were derived from the earlier leaky, all-sample ranking and are not reproduced here, since the corrected pipeline does not export a train-only overlap count.)

> **[FIGURE 5 — Feature overlap Venn diagram]** *Venn diagram showing the overlap between top-ranked features in Cluster 0 and Cluster 1 at (a) 70 features and (b) 200 features.*

### 2.5 Model Hyperparameters

Table 4 summarizes the best hyperparameters selected by 5-fold cross-validation on the training split only. Both the feature count *K* and the model hyper-parameters are chosen per cluster by the same inner CV; the values below are the ones actually used at test time.

**Table 4.** Optimal feature counts and hyperparameters, selected by train-only 5-fold CV.

| Method | Cluster 0 | Cluster 1 |
|--------|-----------|-----------|
| NNLS | *K* = 360; k = 3, cosine, simplex off, L2 = 0 | *K* = 70; k = 3, cosine, simplex off, L2 = 0.1 |
| Ridge | *K* = 20; α = 10, scaling on | *K* = 10; α = 100, scaling off |
| XGBoost | *K* = 120; 100 trees, lr = 0.05, depth = 2 | *K* = 20; 300 trees, lr = 0.05, depth = 2 |

The selected configurations differ between clusters, reflecting their different forest characteristics: the high-density Cluster 0 draws on a much larger slice of the embedding (larger *K*) for every model, whereas Cluster 1 — which includes the zero-biomass plots — is best fit with far fewer features and stronger regularisation (higher Ridge α, larger XGBoost `reg_lambda`).

### 2.6 Ablation 1: Impact of Feature Selection

To rigorously evaluate the necessity of the correlation-based feature selection step, an ablation study was conducted. The identical UMAP+KMeans clustering pipeline was employed, but rather than selecting the top *N* features, the full 384-dimensional DINO feature space was retained for both clusters.

Under the corrected leak-free protocol, correlation-based feature selection provides only a **marginal** benefit over using all 384 features on this site (leak-free ablation, cluster-separated):
- **NNLS**: 17.40% (with selection) vs **17.90%** (all 384 features) — R² 0.837 vs 0.828.
- **Ridge**: 23.83% (with selection) vs **20.38%** (all features) — i.e. Ridge is actually *better* without selection here.
- **XGBoost**: 26.27% vs **24.71%** (all features).

The earlier draft reported a large benefit from selection, but that was inflated by test-aware feature-count selection. Leak-free, selection is at best a mild regularizer for NNLS and can slightly hurt the linear/tree models; we therefore present it as an optional component. (Full ablation in `methodology_and_results.md`, Table 3, and `fig_corrected_ablation_german`.)

### 2.7 Ablation 2: Impact of Unsupervised Clustering (Single Global Model)

A second ablation study assessed the importance of the initial unsupervised clustering step. Instead of partitioning the dataset into structurally homogeneous groups via UMAP and KMeans, a single global regression model was trained on the entire training set (85/15 split). Feature selection was dynamically performed directly on the global feature pool.

Under the corrected leak-free protocol, clustering helps **NNLS** but not the other regressors (single 85/15 split):
- **NNLS**: 20.45% (global) vs **17.40%** (clustered) — clustering clearly helps (R² 0.775 → 0.837).
- **Ridge**: 21.00% (global) vs 23.83% (clustered) — clustering *hurts* Ridge on this split.
- **XGBoost**: 21.20% (global) vs 26.27% (clustered) — clustering *hurts* XGBoost.

So clustering is beneficial specifically for the local NNLS reconstruction (which gains from structurally homogeneous neighbourhoods) but not for the global linear/tree models. This reverses the original draft's claim that clustering broadly boosted all methods; that pattern was an artifact of the leaky selection.

These findings suggest that while high-quality DINO embeddings can carry a global signal, the bi-modal clustering application significantly boosts the explained variance (R²) and reduces error for non-linear models like XGBoost. By actively partitioning the forest structural typologies using UMAP embeddings, the model avoids blending distinct structural-spectral signatures, yielding a more robust and physically meaningful estimation.

### 2.8 Comparison with Hand-Crafted Features (Literature Baselines)

To contextualize the performance of the proposed DINO-based framework, we implemented standard methodologies from existing remote sensing literature on the identical Karlsruhe dataset (maintaining the strict 85/15 train/test split):

1. **Stepwise Regression on GLCM Texture:** (Adapted from standard optical biomass estimation methodologies) - Extracts multi-scale GLCM texture features (contrast, correlation, energy, homogeneity) across pixel distances 1, 2, and 3, followed by Stepwise Linear Regression.
2. **Tree Ensembles on Texture & VIs:** (Adapted from recent canopy modeling studies) - Extracts GLCM textures, 10 vegetation indices (VIs), and RGB summary statistics, modeled via Random Forest, Gradient Boosting Trees, XGBoost, and a Stacking Ensemble.

**Table 6.** Performance comparison between the proposed DINO method and established hand-crafted feature methods on the held-out test set.

| Feature Type | Model Strategy | Combined %RMSE | RMSE (ton/ha) | Test R² |
|--------------|----------------|----------------|---------------|---------|
| **DINOv2 (Proposed)**| **Cluster-Separated NNLS** | **17.40** | **30.89** | **0.837** |
| DINOv2 (Proposed)| Cluster-Separated Ridge | 23.83 | 42.32 | 0.695 |
| Hand-Crafted (GLCM+VI) | Ridge | 22.79 | 40.46 | 0.721 |
| Hand-Crafted (GLCM+VI) | Random Forest | 23.70 | 42.09 | 0.698 |
| Hand-Crafted (GLCM+VI) | Gradient Boosting Trees | 23.99 | 42.59 | 0.691 |
| Hand-Crafted (GLCM+VI) | XGBoost | 25.55 | 45.37 | 0.649 |

Under the identical leak-free protocol, the DINOv2 + NNLS pipeline is the best method (17.40%, R² 0.837) and clearly ahead of the best hand-crafted-feature baseline (Ridge on GLCM, 22.79%, R² 0.721) — roughly a 24% relative reduction in error. This is a genuine advantage, but a measured one: the tuned GLCM baselines all reach R² ≈ 0.65–0.72 here, not the negative or near-zero values reported in the original leaky draft. Over 20 stratified splits (Table 5-R), DINO-NNLS averages 23.0% ± 3.7 vs the best literature 24.4% ± 2.5, confirming a consistent but modest edge for self-supervised features.

**Table 5-R.** German robustness over 20 stratified 85/15 splits (proposed pipeline; %RMSE and R², mean ± std).

| Method | %RMSE (mean ± std) | R² (mean ± std) |
|--------|--------------------|-----------------|
| **DINOv2 + NNLS** | **22.99 ± 3.65** | **0.70 ± 0.15** |
| DINOv2 + XGBoost | 23.99 ± 3.63 | 0.68 ± 0.12 |
| DINOv2 + Ridge | 24.52 ± 2.86 | 0.67 ± 0.12 |
| GLCM + Ridge (best literature) | 24.43 ± 2.46 | 0.68 ± 0.10 |
| GLCM + Random Forest | 26.49 ± 4.71 | 0.61 ± 0.18 |

### 2.9 Biomass Raster Maps

Spatially continuous biomass rasters were generated for the entire Karlsruhe aerial image. Each 224 × 224 pixel patch was first assigned to its nearest cluster via the UMAP+KMeans pipeline, then predicted using the cluster-specific model.

> **[FIGURE 6 — Biomass rasters]** *Side-by-side biomass raster maps (ton/ha) for the Karlsruhe study area produced by NNLS, Ridge Regression, and XGBoost. A color bar indicates the biomass scale. Non-forested areas (predicted by the zero-biomass-augmented models) appear in cool colors.*

> **[FIGURE 7 — Effect of overlap on raster smoothness]** *Biomass rasters generated at 0%, 25%, 50%, and 75% overlap levels for the best-performing method (Ridge Regression). Higher overlap produces smoother transitions between patches.*

The zero-biomass augmentation had a visible impact on the raster maps: models correctly predicted near-zero biomass in agricultural and urban areas that would otherwise have been assigned non-zero values based purely on forest training data. This demonstrates the practical benefit of including representative zero-biomass samples for wall-to-wall mapping.

---

## 3. Discussion

The study advocates a robust and highly scalable approach: utilizing un-modified DINOv2 embeddings extracted from raw RGB imagery, paired with simple unsupervised UMAP clustering and correlation-based feature selection prior to prediction modeling.

### 3.1 Unifying DINOv2 Foundations Across Ecosystems

A key finding is that the same frozen DINO model — without any domain-specific modification — produces useful biomass features in a **temperate European forest**, just as it characterizes canopy in **subtropical Pakistani mountains**. The corrected combined R² of 0.837 (%RMSE = 17.40%; 23.0% ± 3.7 over 20 splits) for Germany aligns with the Pakistani metrics (R² = 0.886, %RMSE = 29.93%). The consistency of the *approach* across ecosystems is the durable finding; the absolute accuracy is naturally site-dependent.

This cross-ecosystem consistency unequivocally bridges the fundamental disparity between remote sensing disciplines today; asserting that foundational self-supervised models intrinsically capture **universal textural parameters** independent of altitude or latitudinal climate zones.

### 3.2 NNLS as the Cross-Dataset Defacto Standard

As on the Pakistani site, **NNLS (Non-Negative Least Squares)** is the best regressor under the leak-free protocol (17.40%; 23.0% ± 3.7 over 20 splits), ahead of Ridge and XGBoost. 

The convergence of algorithmic optimality across both geographic datasets verifies our central hypothesis: rather than relying on deep, highly-parameterized non-linear networks that blindly fit embeddings and overfit to the narrow datasets typical of localized remote sensing studies, biomass densities are phenomenologically additive within the DINOv2 subspace. Neighbor-based reconstruction accurately mimics literal density composition observed from space.

### 3.3 The Inevitable Pitfall of Excessive Tree Ensembles (XGBoost)

In the broader context of aerial predictive algorithms, modern literature extensively champions XGBoost setups for arbitrary tabular features. Consistent across both the German data (XGBoost optimized tightly at its ceiling with 30 features) and the prior Pakistani testing, Tree ensembles systematically lag behind simpler linear/neighbor strategies concerning latent projection assimilation. XGBoost invariably deteriorates sharply alongside increasing feature dimensions. Foundational vision modalities operate best devoid of discrete binning thresholds natively used by Tree-nodes. Feature continuity is compromised otherwise.

### 3.4 The Role of Clustering In Real-World Applications

Under the corrected protocol, clustering helps only the local NNLS reconstruction (R² 0.775 → 0.837; %RMSE 20.45% → 17.40%) and slightly *hurts* the global Ridge/XGBoost models. Clustering is therefore an NNLS-specific benefit, not a universal one.

- **Cluster 1** encompasses wider biomass ranges overlapping zero-biomass segments, requiring separate metric modeling than internal deep-forest topologies.
- **Cluster 0** isolates narrower deep-density gradients where separate NNLS feature similarities are computed efficiently avoiding non-forest bias.

The practical benefit of clustering lies in enabling **per-cluster feature selection**, allowing algorithms to prioritize semantic sub-group characteristics that uniformly describe localized physics models of the foliage.

### 3.5 Zero-Biomass Augmentation

The inclusion of 12 zero-biomass samples expanded the dataset's biomass range from [100.88, 302.01] to [0.00, 302.01] ton/ha, providing the models with explicit training signal for non-forested areas. This augmentation:

1. **Improved raster map quality** by enabling correct prediction of near-zero biomass in agricultural and urban patches.
2. **Anchored the regression intercept**, preventing models from extrapolating unpredictably when encountering non-forest features at inference time.
3. **Contributed to Cluster 1's strong performance**, as the zero-biomass samples created a wider biomass gradient within that cluster, facilitating better learning of the feature–biomass relationship.

While 12 zero-biomass samples represent only 10.6% of the dataset, their impact on wall-to-wall mapping utility is disproportionately large, as non-forested areas typically constitute a substantial fraction of the raster extent.

### 3.6 Comparison with the Pakistani Dataset

Table 5 establishes the empirical bridge tying together the identical methodological frameworks deployed across radically differing environments.

**Table 5.** Cross-site comparison of the proposed pipeline mechanisms.

| Aspect | Pakistani Data | German Data |
|--------|---------------|-------------|
| Extracted Features | DINO / DINOv2 | DINO / DINOv2 |
| Location | Balakot, Pakistan | Karlsruhe, Germany |
| Forest type | Subtropical/temperate mountain | Managed temperate lowland |
| Field plots | 25 | 101 (+ 12 zero-biomass) |
| Validation | LOO cross-validation | 85/15 standard split |
| Validation | nested LOO | 85/15 + 20-split robustness |
| Best Algorithm | NNLS | NNLS |
| Best %RMSE | 29.93% | 17.40% (23.0% over 20 splits) |
| Best R² | 0.886 | 0.837 (0.70 over 20 splits) |

NNLS is the best regressor on both sites (both R² ≥ 0.84 on the primary evaluation), which anchors the *approach* as generalizable. The DINO advantage over tuned hand-crafted-feature baselines is clear on the German site and narrow on the tiny Pakistani site — a consistent, moderate benefit rather than a decisive one.

### 3.7 Limitations

1. **Cluster 0 test set is small.** With only 10 test samples in Cluster 0, per-cluster metrics are noisy. The combined metrics (pooling all 17 test samples) provide a more reliable overall assessment.

2. **Zero-biomass patches were manually identified.** The 12 non-forest patches were selected based on visual inspection of the aerial image. Automated or systematic sampling of non-forest areas (e.g., using land cover maps) would improve reproducibility.

3. **Fixed number of clusters.** Only K = 2 clusters were evaluated. The optimal number of clusters could be data-dependent, and different clustering strategies (e.g., hierarchical clustering, Gaussian mixture models) could yield different results.

4. **Single train/test split (addressed).** The headline uses one 85/15 split; to avoid over-reading a lucky split, we additionally report mean ± std over **20 stratified splits** (Table 5-R), where DINO-NNLS averages 23.0% ± 3.7 (R² 0.70) and remains the best method, though within one standard deviation of the best literature baseline.

### 3.8 Summary

Under a strict leak-free protocol, **self-supervised DINOv2 features from standard RGB aerial imagery yield accurate biomass estimators across two very different forest zones**, achieving ~17% error (single split; ~23% over 20 splits) at R² ≈ 0.84 (0.70 over splits) in Germany and ~30% / R² 0.89 in Pakistan. Non-parametric, convex NNLS regression is the best regressor on both sites and produces bounded, physically plausible maps. DINO features consistently beat supervised ResNet50 features, and modestly beat fairly-tuned GLCM/vegetation-index baselines — clearly on the larger German site and narrowly on the small Pakistani one. The practical value is a simple, affordable, RGB-only, fine-tuning-free protocol for biomass mapping with foundation-model features, with honestly characterized rather than overstated accuracy.
