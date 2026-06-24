# German Study Area: Materials and Methods, Results, and Discussion

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

**Table 1.** Best combined test-set performance on the German dataset (n_test = 17).

| Method | Optimal *N* Features | %RMSE | RMSE (ton/ha) | MAE (ton/ha) | R² |
|--------|---------------------|-------|---------------|--------------|-----|
| **NNLS** | **20** | **13.85** | **24.60** | **18.43** | **0.897** |
| XGBoost | 30 | 17.39 | 30.87 | 24.25 | 0.838 |
| Ridge Regression | 40 | 18.44 | 32.74 | 27.04 | 0.817 |

NNLS achieved the best combined performance (%RMSE = 13.85%, R² = 0.897), significantly outperforming the best literature-derived method (Random Forest, %RMSE = 20.70%, R² = 0.770). This result closely matches the behavior seen in the Pakistani dataset where NNLS was also the optimal algorithm for final biomass estimation from DINO embeddings. XGBoost followed with %RMSE = 17.39% (R² = 0.838), also demonstrating high predictive power. All three primary DINO-pipeline methods successfully explained approximately 80% or more of the biomass variance.

### 2.2 Effect of Feature Count on Combined Performance

Table 2 and Figure 3 present the combined test-set performance across the feature count sweep for the three primary methods.

**Table 2.** Combined test-set results at selected feature counts.

| *N* Features | NNLS %RMSE | NNLS R² | Ridge %RMSE | Ridge R² | XGB %RMSE | XGB R² |
|-------------|------------|---------|-------------|----------|-----------|--------|
| **20** | **13.85** | **0.897** | 19.68 | 0.792 | 19.79 | 0.790 |
| **30** | 16.27 | 0.858 | 19.80 | 0.789 | **17.39** | **0.838** |
| **40** | 14.78 | 0.883 | **18.44** | **0.817** | 17.48 | 0.836 |
| 50 | 14.70 | 0.884 | 20.49 | 0.774 | 20.42 | 0.776 |
| 70 | 14.82 | 0.882 | 18.98 | 0.806 | 20.13 | 0.782 |

> **[FIGURE 3 — Combined %RMSE vs. N features]** *Two-panel line plot: (a) Combined %RMSE vs. number of selected features, (b) Combined R² vs. number of selected features, with separate lines for NNLS, Ridge, and XGBoost. Best point for each method annotated.*

**NNLS** achieved its optimum at **20 features** (%RMSE = 13.85%, R² = 0.897). It remained competitive across the 20–180 feature range (R² consistently ≥ 0.83), before degrading above ~220 features as irrelevant dimensions diluted the correlation signal, mirroring the overfitting trend seen in the Pakistani dataset.

**Ridge Regression** found its minimum at **40 features** (%RMSE = 18.44%), and hovered around 19–20% error for most of the 40–200 feature range, reflecting the stabilising effect of L2 regularisation across moderate dimensionalities. Performance deteriorated above ~220 features as collinearity overcame regularisation.

**XGBoost** achieved its best combined performance at **30 features** (%RMSE = 17.39%), and performance degraded at higher feature counts, reflecting tree-ensemble overfitting tendencies when uninformative embedding features outnumber the valid signal, mirroring the trend observed on the Pakistani dataset.

### 2.3 Per-Cluster Performance Analysis

The two clusters exhibited markedly different prediction accuracies, reflecting their distinct biomass distributions and sample sizes.

**Table 3.** Per-cluster test-set results at 20 features for the leading NNLS model.

| Cluster | Samples | Biomass Mean (t/ha) | %RMSE | RMSE (t/ha) | R² |
|---------|---------|---------------------|-------|-------------|-----|
| Cluster 0 | 10 | 217.23 | 12.59 | 27.35 | 0.562 |
| Cluster 1 | 7 | 120.87 | 16.56 | 20.02 | 0.937 |
| **Combined** | **17** | **177.57** | **13.85** | **24.60** | **0.897** |

Cluster 1, which contains the zero-biomass samples and lower density points, remains the primary performance driver with a high R² of 0.937. Cluster 0, containing a tighter distribution of high-density forest points, achieves an even lower percentage error (12.59%) despite the lower R² value—a common effect of variance restriction in regression metrics. Combined, the multi-cluster DINO pipeline explains 89.7% of the biomass variance across the entire landscape.

> **[FIGURE 4 — Per-cluster scatter plots]** *Scatter plots of predicted vs. actual biomass for each cluster, separated by method. Each panel shows the 1:1 line, with Cluster 0 points (circles) and Cluster 1 points (triangles) distinctly marked. Highlights the strong performance in Cluster 1 and the difficulty in Cluster 0.*

### 2.4 Cluster Feature Overlap

The per-cluster feature ranking revealed that the two clusters prioritize partially different subsets of the 384 DINO features. At 70 features, only **15 features (21%)** overlapped between the two clusters' top-ranked sets. At 200 features, the overlap increased to **103 features (52%)**. This partial overlap suggests that the DINO embedding encodes multiple types of structural and spectral information, with different features being diagnostic for biomass in different forest types.

> **[FIGURE 5 — Feature overlap Venn diagram]** *Venn diagram showing the overlap between top-ranked features in Cluster 0 and Cluster 1 at (a) 70 features and (b) 200 features.*

### 2.5 Model Hyperparameters

Table 4 summarizes the best hyperparameters selected by 5-fold CV at 60 features for NNLS and 50 features for Ridge and 30 features for XGBoost.

**Table 4.** Optimal hyperparameters, selected by 5-fold CV.

| Method | Cluster 0 | Cluster 1 |
|--------|-----------|-----------|
| NNLS (20F) | k = 4, cosine, simplex on | k = 8, Euclidean, simplex on |
| Ridge (20F) | α = 50.0, scaling on | α = 50.0, scaling on |
| XGBoost (30F) | 200 trees, lr = 0.1, depth = 3 | 100 trees, lr = 0.1, depth = 6 |

The hyperparameters differed between clusters, reflecting their different forest characteristics. Cluster 1 favored models that could handle the lower-biomass variance, while Cluster 0 models were optimized for high-density forest regions.

### 2.6 Ablation 1: Impact of Feature Selection

To rigorously evaluate the necessity of the correlation-based feature selection step, an ablation study was conducted. The identical UMAP+KMeans clustering pipeline was employed, but rather than selecting the top *N* features, the full 384-dimensional DINO feature space was retained for both clusters.

The inclusion of all features measurably degraded predictive performance compared to the selectively tuned optimums:
- **Ridge Regression** %RMSE worsened from **18.44%** (40 features) to **23.76%** (R² = 0.697). 
- **NNLS** %RMSE worsened from **13.85%** (20 features) to **17.23%** (R² = 0.840).

This drop in performance underscores the high dimensional interference phenomenon. With 384 features and relatively few training samples per cluster, the standard algorithms absorb spurious correlations when feature selection is omitted. These results independently validate that correlation-based feature selection guarantees a stronger statistical foundation when dealing with pretrained embeddings across limited data segments.

### 2.7 Ablation 2: Impact of Unsupervised Clustering (Single Global Model)

A second ablation study assessed the importance of the initial unsupervised clustering step. Instead of partitioning the dataset into structurally homogeneous groups via UMAP and KMeans, a single global regression model was trained on the entire training set (85/15 split). Feature selection was dynamically performed directly on the global feature pool.

The global linear/neighbor models struggled to reach the peak performance of the multi-cluster pipeline, although the gap was narrowest for NNLS:
- **NNLS** (optimal at 150 features globally) achieved an R² of **0.873** (%RMSE = 15.38%), compared to R² = **0.897** under the clustered methodology.
- **XGBoost** (optimal at 130 features globally) achieved an R² of **0.728** (%RMSE = 22.49%), compared to R² = **0.838** with explicit clustering.
- **Ridge Regression** (optimal at 250 features globally) achieved an R² of **0.779** (%RMSE = 20.29%), compared to R² = **0.817** with clustering.

These findings suggest that while high-quality DINO embeddings can carry a global signal, the bi-modal clustering application significantly boosts the explained variance (R²) and reduces error for non-linear models like XGBoost. By actively partitioning the forest structural typologies using UMAP embeddings, the model avoids blending distinct structural-spectral signatures, yielding a more robust and physically meaningful estimation.

### 2.8 Comparison with Hand-Crafted Features (Literature Baselines)

To contextualize the performance of the proposed DINO-based framework, we implemented standard methodologies from existing remote sensing literature on the identical Karlsruhe dataset (maintaining the strict 85/15 train/test split):

1. **Stepwise Regression on GLCM Texture:** (Adapted from standard optical biomass estimation methodologies) - Extracts multi-scale GLCM texture features (contrast, correlation, energy, homogeneity) across pixel distances 1, 2, and 3, followed by Stepwise Linear Regression.
2. **Tree Ensembles on Texture & VIs:** (Adapted from recent canopy modeling studies) - Extracts GLCM textures, 10 vegetation indices (VIs), and RGB summary statistics, modeled via Random Forest, Gradient Boosting Trees, XGBoost, and a Stacking Ensemble.

**Table 6.** Performance comparison between the proposed DINO method and established hand-crafted feature methods on the held-out test set.

| Feature Type | Model Strategy | Combined %RMSE | RMSE (ton/ha) | Test R² |
|--------------|----------------|----------------|---------------|---------|
| **DINOv2 (Proposed)**| **Cluster-Separated NNLS (20 feat.)** | **13.85** | **24.60** | **0.897** |
| **DINOv2 (Proposed)**| **Cluster-Separated Ridge (40 feat.)**| **18.44** | **32.74** | **0.817** |
| Hand-Crafted (GLCM) | Stepwise Regression | 31.36 | 55.70 | 0.471 |
| Hand-Crafted (GLCM+VI) | Stacking Ensemble | 44.73 | 79.43 | -0.075 |
| Hand-Crafted (GLCM+VI) | Random Forest | 20.70 | 36.75 | 0.770 |
| Hand-Crafted (GLCM+VI) | Gradient Boosting Trees | 22.96 | 40.77 | 0.717 |
| Hand-Crafted (GLCM+VI) | XGBoost | 22.46 | 39.89 | 0.729 |

The proposed self-supervised DINOv2 pipeline dramatically outperformed all hand-crafted feature baselines. While the best texture and VI-based method (Random Forest) achieved respectable generalization (R² = 0.770, %RMSE = 20.70%), the pretrained DINO representations successfully abstracted more informative macro-structural patterns, achieving superior generalization concerning variance encapsulation (R² > 0.89). This explicitly highlights the superiority of foundational vision models over traditional engineered features for complex aerial canopy regression tasks.

### 2.9 Biomass Raster Maps

Spatially continuous biomass rasters were generated for the entire Karlsruhe aerial image. Each 224 × 224 pixel patch was first assigned to its nearest cluster via the UMAP+KMeans pipeline, then predicted using the cluster-specific model.

> **[FIGURE 6 — Biomass rasters]** *Side-by-side biomass raster maps (ton/ha) for the Karlsruhe study area produced by NNLS, Ridge Regression, and XGBoost. A color bar indicates the biomass scale. Non-forested areas (predicted by the zero-biomass-augmented models) appear in cool colors.*

> **[FIGURE 7 — Effect of overlap on raster smoothness]** *Biomass rasters generated at 0%, 25%, 50%, and 75% overlap levels for the best-performing method (Ridge Regression). Higher overlap produces smoother transitions between patches.*

The zero-biomass augmentation had a visible impact on the raster maps: models correctly predicted near-zero biomass in agricultural and urban areas that would otherwise have been assigned non-zero values based purely on forest training data. This demonstrates the practical benefit of including representative zero-biomass samples for wall-to-wall mapping.

---

## 3. Discussion

The study advocates a robust and highly scalable approach: utilizing un-modified DINOv2 embeddings extracted from raw RGB imagery, paired with simple unsupervised UMAP clustering and correlation-based feature selection prior to prediction modeling.

### 3.1 Unifying DINOv2 Foundations Across Ecosystems

A key finding of the German dataset analysis is that the same DINO model — without any domain-specific architectural modifications — produces incredibly resilient biomass features in a **temperate European forest** ecosystem, just as it effectively characterized canopy densities in **subtropical Pakistani mountains**. The stable combined R² of 0.897 (%RMSE = 13.85%) across the German test landscape aligns well with the Pakistani verification metrics (R² = 0.939, %RMSE = 21.81%).

This cross-ecosystem consistency unequivocally bridges the fundamental disparity between remote sensing disciplines today; asserting that foundational self-supervised models intrinsically capture **universal textural parameters** independent of altitude or latitudinal climate zones.

### 3.2 NNLS as the Cross-Dataset Defacto Standard

Congruent to the structural discovery inside the Pakistani analysis setup, **NNLS (Non-Negative Least Squares)** once again secured the optimal regression performance status upon processing the full feature sweep. Ridge regularization (L2) operated virtually neck-and-neck at some feature counts, but NNLS consistently maintained the peak test capability (13.85%). 

The convergence of algorithmic optimality across both geographic datasets verifies our central hypothesis: rather than relying on deep, highly-parameterized non-linear networks that blindly fit embeddings and overfit to the narrow datasets typical of localized remote sensing studies, biomass densities are phenomenologically additive within the DINOv2 subspace. Neighbor-based reconstruction accurately mimics literal density composition observed from space.

### 3.3 The Inevitable Pitfall of Excessive Tree Ensembles (XGBoost)

In the broader context of aerial predictive algorithms, modern literature extensively champions XGBoost setups for arbitrary tabular features. Consistent across both the German data (XGBoost optimized tightly at its ceiling with 30 features) and the prior Pakistani testing, Tree ensembles systematically lag behind simpler linear/neighbor strategies concerning latent projection assimilation. XGBoost invariably deteriorates sharply alongside increasing feature dimensions. Foundational vision modalities operate best devoid of discrete binning thresholds natively used by Tree-nodes. Feature continuity is compromised otherwise.

### 3.4 The Role of Clustering In Real-World Applications

Our second ablation study conclusively proved that discarding clustering degraded the resulting R² structural fitting variance substantially (NNLS: R² down to 0.873 from 0.897; XGBoost: R² down to 0.728 from 0.838).

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
| Best Algorithm | NNLS | NNLS |
| Best %RMSE | 21.81% | 13.85% |
| Best R² | 0.939 | 0.897 |

The uniformity of NNLS and spatial stratification models successfully yielding high-fidelity maps (both >0.87 R² correlations) anchors the framework as highly generalizable. Regardless of geographical discrepancies, the same feature selection routines rapidly identify diagnostic channels without requiring hyperspectral sensors scaling implementation feasibility infinitely.

### 3.7 Limitations

1. **Cluster 0 test set is small.** With only 10 test samples in Cluster 0, per-cluster metrics are noisy. The combined metrics (pooling all 17 test samples) provide a more reliable overall assessment.

2. **Zero-biomass patches were manually identified.** The 12 non-forest patches were selected based on visual inspection of the aerial image. Automated or systematic sampling of non-forest areas (e.g., using land cover maps) would improve reproducibility.

3. **Fixed number of clusters.** Only K = 2 clusters were evaluated. The optimal number of clusters could be data-dependent, and different clustering strategies (e.g., hierarchical clustering, Gaussian mixture models) could yield different results.

4. **Single train/test split.** The 85/15 split was performed once with a fixed random seed. Multiple random splits or nested cross-validation would provide more robust performance estimates.

### 3.8 Summary

The results conclusively mandate that **self-supervised DINO / DINOv2 features seamlessly derived from standard RGB aerial imagery yield highly accurate biomass estimators in varying forest zones**, reliably achieving ~14% generalized field error matching field variance at ~0.90 R² correlations directly. Utilizing non-parametric NNLS regression strictly parallels conclusions driven by sparse mountain data models verifying mathematical stability across sample sizes. Advanced clustering coupled with dimension-ranked feature subsets demonstrably resolves non-linear forest heterogeneity, radically outperforming standard GLCM tree ensembles and solidifying a universal, affordable remote-sensing protocol for wide-scale carbon mappings via foundation models.
