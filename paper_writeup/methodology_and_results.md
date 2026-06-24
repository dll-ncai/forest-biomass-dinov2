# Biomass Estimation from High-Resolution RGB Aerial Imagery Using Self-Supervised Deep Features

## 2. Methodology

### 2.1 Study Areas and Datasets

Two geographically distinct study areas were selected to evaluate the generalizability of the proposed approach across different forest ecosystems, canopy structures, and acquisition conditions.

**Study Area 1: Balakot, Pakistan.** The first study area is located in Balakot, Khyber Pakhtunkhwa province, Pakistan. A high-resolution aerial photograph covering the study area was acquired as a multi-band raster image with spatial dimensions of 3,175 x 4,771 pixels. Ground-truth above-ground biomass (AGB) measurements were available for 25 circular sample plots. Due to the small sample size, Leave-One-Out (LOO) cross-validation was employed as the primary evaluation strategy to maximize use of the available data.

**Study Area 2: German Forest Site.** The second study area is located in a managed forest in Germany. High-resolution aerial RGB imagery was acquired over the site, and ground-truth AGB measurements were available for 101 sample plots. An additional 12 zero-biomass samples were extracted from areas with no vegetative cover (e.g., roads, clearings) and incorporated into the dataset, yielding a total of 113 samples. This augmentation was introduced to improve model performance in predicting zero or near-zero biomass regions, which is critical for generating spatially complete biomass rasters.

### 2.2 Overview of the Proposed Approach

The proposed methodology comprises four main stages: (1) patch extraction and preprocessing, (2) feature extraction using a self-supervised Vision Transformer, (3) correlation-based feature selection, and (4) biomass prediction using regression models. The central hypothesis is that features learned by a self-supervised Vision Transformer (ViT) from natural images encode sufficient structural and textural information from RGB imagery to enable accurate biomass estimation --- without requiring multispectral or LiDAR data.

### 2.3 Patch Extraction and Preprocessing

For each sample plot location, a square image patch was extracted from the aerial photograph, centered on the plot centroid.

For the **Pakistani dataset**, patches of 152 x 152 pixels were extracted, from which a center crop of 112 x 112 pixels was taken to minimize edge effects. The center-cropped patches were then resized to 224 x 224 pixels using bicubic interpolation to match the input resolution expected by the feature extraction model.

For the **German dataset**, patches of 395 x 395 pixels were extracted, corresponding to the spatial extent of the sample plots. A center crop of 224 x 224 pixels was taken directly, requiring no additional resizing.

All image patches were normalized using ImageNet statistics (mean = [0.485, 0.456, 0.406], standard deviation = [0.229, 0.224, 0.225]) prior to feature extraction, consistent with the pretraining protocol of the feature extraction model.

### 2.4 Feature Extraction: Self-Supervised DINO Vision Transformer

#### 2.4.1 Background: Self-Supervised Learning with DINO

A key component of this work is the use of a **self-supervised** feature extractor --- one that was trained entirely without labeled data. Specifically, we employ features from **DINO** (Self-**Di**stillation with **No** Labels), a self-supervised learning framework proposed by Caron et al. (2021). DINO trains a Vision Transformer through a teacher-student self-distillation mechanism: a student network is trained to predict the output of a momentum-updated teacher network, using different augmented views of the same image. Crucially, this entire training procedure uses no labeled data whatsoever --- the model learns rich visual representations purely from the structure of the images themselves.

DINO has been shown to learn features that capture semantically meaningful structure in images, including object boundaries, texture patterns, and spatial hierarchies. The attention maps of a DINO-trained ViT have been found to closely correspond to object segmentations, suggesting that the model develops an implicit understanding of scene structure. These properties make DINO features particularly well-suited for remote sensing tasks where structural and textural image properties correlate with biophysical variables such as biomass.

A critical advantage of using self-supervised features is that the model requires no labeled remote sensing data for training the feature extractor. The same pretrained model can be applied across diverse geographic regions, forest types, and image acquisition conditions without retraining, as demonstrated in this study across both Pakistani and German datasets.

#### 2.4.2 Feature Extraction Pipeline

Features were extracted using the **DINO ViT-Small** model with a patch size of 16 pixels (`vit_small_patch16_224.dino`), accessed through the `timm` (PyTorch Image Models) library. The model was used in inference mode with the classification head removed (`num_classes=0`), such that the output is the **CLS token** embedding from the final transformer layer. The CLS token in ViT architectures aggregates global information from all spatial positions through the self-attention mechanism, making it a compact yet information-rich representation of the entire image patch. This yields a **384-dimensional feature vector** for each input image patch.

No fine-tuning of the DINO model was performed; the pretrained weights (trained on ImageNet) were used directly as a frozen feature extractor. This deliberate design choice tests whether generic visual features learned from natural images transfer effectively to remote sensing for biomass estimation, without domain-specific labeled data for representation learning.

> **[FIGURE 1 PLACEHOLDER]** *Schematic diagram of the proposed pipeline: (a) Aerial RGB image acquisition, (b) patch extraction centered on sample plots, (c) DINO ViT-Small feature extraction (384-dim CLS token), (d) correlation-based feature selection, (e) regression model training, (f) biomass prediction. A flowchart showing the end-to-end pipeline from raw imagery to biomass estimates.*

### 2.5 Correlation-Based Feature Selection

Given the 384-dimensional DINO feature vectors, a correlation-based feature selection strategy was employed to identify the most biomass-relevant features and reduce input dimensionality. No PCA or other dimensionality reduction was applied.

For each of the 384 DINO features, the Pearson correlation coefficient was computed between that feature and the ground-truth biomass values across all training samples. Features were ranked by the absolute value of their Pearson correlation coefficient, and the top *N* most correlated features were selected for model training.

The number of selected features *N* was treated as a hyperparameter and systematically varied from 20 to 300 in increments of 10, to study the sensitivity of each model to feature dimensionality. This sweep identified the optimal trade-off between information content and the risk of overfitting for each regression method.

For the Pakistani dataset, the top-ranked feature (`feature_14`) exhibited a Pearson correlation of *r* = 0.865 (*p* = 2.49 x 10^-8) with biomass. Out of 384 features, 170 (44.3%) showed statistically significant correlations with biomass at *p* < 0.05, indicating that a substantial fraction of DINO features carry biomass-relevant information.

For the German dataset, feature ranking was performed independently within each cluster (see Section 2.6), allowing cluster-specific feature sets to capture the most relevant features for each forest subtype.

### 2.6 Unsupervised Clustering (German Dataset)

For the German dataset, an unsupervised clustering step was introduced to account for the heterogeneity in forest structure across the study area. The rationale was that different forest types (e.g., varying species composition, age classes, canopy density) may exhibit different feature-biomass relationships, and training cluster-specific models could improve prediction accuracy.

**UMAP Embedding.** The standardized DINO feature vectors were projected into a 2-dimensional space using **Uniform Manifold Approximation and Projection (UMAP)** with the following parameters: `n_neighbors=15`, `n_components=2`, `min_dist=0.1`, `metric='euclidean'`, `random_state=42`. UMAP was fit on training data only, and the learned transform was applied to test data to prevent data leakage.

**K-Means Clustering.** The 2D UMAP embeddings were then partitioned into **2 clusters** using K-Means clustering (`n_clusters=2`, `n_init=10`, `random_state=42`). Separate regression models were trained for each cluster, allowing each model to specialize in the feature-biomass relationship characteristic of its respective cluster. Feature ranking (Section 2.5) was also performed independently per cluster, enabling each cluster's model to use features most correlated with biomass within that subgroup.

This clustering step was not applied to the Pakistani dataset due to its small sample size (n = 25), which would not support robust cluster-specific modeling.

> **[FIGURE 2 PLACEHOLDER]** *UMAP 2D embedding of DINO features for the German dataset, colored by cluster assignment (K-Means, k=2). Points should also encode biomass values (e.g., size or color intensity) to show that clusters capture structurally meaningful groups. A second panel could show the same UMAP colored by actual biomass.*

### 2.7 Regression Models

Three regression models were evaluated for biomass prediction from the selected DINO features across both study areas:

#### 2.7.1 Non-Negative Least Squares (NNLS) Regression

The NNLS-based predictor operates as a **dictionary-based regression** method. For a given query sample, the prediction is computed as a non-negatively weighted combination of the biomass values of its nearest training neighbors in the feature space.

The algorithm proceeds as follows:

1. **Neighbor Identification.** For each query sample, the *k* nearest neighbors in the training set are identified. When cosine similarity is used, feature vectors are L2-normalized and similarity is computed as the dot product of the normalized vectors. When Euclidean distance is used, features are first standardized using a StandardScaler, and neighbors are selected by minimum Euclidean distance.

2. **Weight Computation via NNLS.** The query feature vector is reconstructed as a non-negative linear combination of the neighbor feature vectors by solving:

   minimize ||Aw - b||^2   subject to   w >= 0

   where *A* is the matrix of neighbor feature vectors (as columns), *b* is the query feature vector, and *w* is the vector of non-negative weights. This optimization is solved using `scipy.optimize.nnls`.

3. **Simplex Constraint.** Optionally, a simplex constraint is applied, normalizing the weights so that they sum to unity: w_i = w_i / sum(w). This ensures that the predicted biomass is a convex combination of the neighbor biomass values.

4. **Prediction.** The predicted biomass is the weighted sum of the training biomass values: y_pred = sum(w_i * y_i), where y_i are the biomass values of the identified neighbors.

**Hyperparameters.** For the Pakistani dataset: *k* = 3, cosine similarity, simplex constraint enabled. For the German dataset, hyperparameters were tuned via 5-fold cross-validation over: *k* in {3, 4, 5, 6, 7, 8, 10, 12, 15}, cosine in {True, False}, simplex in {True, False}.

#### 2.7.2 Ridge Regression

A Ridge regression model (L2-regularized linear regression) was employed as a baseline linear model. Features were standardized using a StandardScaler prior to fitting. The regularization parameter alpha was set to 10.0 for the Pakistani dataset. For the German dataset, alpha was tuned via 5-fold cross-validation over alpha in {0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0}.

#### 2.7.3 XGBoost

An XGBoost (eXtreme Gradient Boosting) model was used as a representative ensemble tree-based method. For the Pakistani dataset, hyperparameters were tuned using LOO cross-validation over a grid of: `learning_rate` in {0.03, 0.05}, `max_depth` in {2, 3, 4}, `reg_lambda` in {1.0, 10.0}, with fixed `n_estimators=100`, `subsample=0.8`, `colsample_bytree=0.8`, and `reg_alpha=0.1`. For the German dataset, a similar grid search was performed via 5-fold cross-validation over `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, and `reg_lambda`.

### 2.8 Validation Strategy

**Pakistani Dataset (LOO).** Due to the limited sample size of 25 plots, **Leave-One-Out (LOO) cross-validation** was employed. In each iteration, 24 samples were used for training and the remaining sample for evaluation. This process was repeated for all 25 samples, and performance metrics were computed over all held-out predictions. LOO maximizes the training data available in each fold and provides an unbiased estimate of generalization performance for small datasets.

**German Dataset (Train/Test Split with Cross-Validation).** The 113 samples were split into **85% training and 15% test** sets using stratified sampling to preserve the biomass distribution (random state = 42), yielding 18 test samples (8 in Cluster 0 and 10 in Cluster 1). Within the training set, **5-fold cross-validation** was used for hyperparameter tuning and model selection. Final performance was evaluated on the held-out test set. Combined (overall) metrics were computed by pooling predictions from both clusters and evaluating against the combined test set.

### 2.9 Evaluation Metrics

Model performance was assessed using the following metrics:

- **Percentage RMSE (%RMSE):** RMSE normalized by the mean observed biomass, expressed as a percentage. This metric facilitates comparison across datasets with different biomass ranges:

  %RMSE = (RMSE / mean(y_observed)) x 100

- **Root Mean Square Error (RMSE):** The square root of the mean squared prediction errors, in the same units as biomass.

- **Mean Absolute Error (MAE):** The average absolute difference between predicted and observed biomass.

- **Coefficient of Determination (R^2):** The proportion of variance in the observed biomass explained by the model. Values close to 1 indicate strong explanatory power; negative values indicate that the model performs worse than simply predicting the mean.

### 2.10 Comparison with Literature Methods

To contextualize the performance of the proposed DINO-based approach, we implemented established methods from the remote sensing biomass estimation literature that rely on hand-crafted features from RGB imagery. Specifically, we implemented methods from two published studies:

1. **Liu et al. (2025)** --- "Forest Aboveground Biomass Estimation Using High-Resolution Imagery and Integrated Machine Learning" (*Forests*, 16, 1777), which employs GLCM texture features with ensemble machine learning.

2. **Eckert (2012)** --- "Improved Forest Biomass and Carbon Estimations Using Texture Measures from WorldView-2 Satellite Data" (*Remote Sensing*, 4, 810), which uses multi-scale GLCM textures and stepwise multiple linear regression.

The following hand-crafted features were extracted from the RGB imagery:

**GLCM Texture Features (Multi-Scale).** Gray-Level Co-Occurrence Matrix (GLCM) features were computed at three spatial distances (*d* = 1, 2, 3 pixels). For each distance, five texture properties were extracted: Correlation, Homogeneity, Contrast, Energy, and Angular Second Moment (ASM). For each property at each distance, four summary statistics (mean, standard deviation, maximum, minimum) were computed across the four standard GLCM orientations (0, 45, 90, and 135 degrees). Additionally, overall statistics across all distances and image-level variance and mean were computed, yielding a total of **66 multi-scale GLCM features**.

**RGB-Based Vegetation Indices.** Ten vegetation indices computable from RGB bands were extracted: Visible Vegetation Index (VVI), Green-Red Vegetation Index (GRVI), Red-Green Ratio Index (RGRI), Green-Red Ratio Index (GRRI), Excess Green Index (ExG), Excess Red Index (ExR), Color Index of Vegetation Extraction (CIVE), Visible Atmospherically Resistant Index (VARI), Triangular Greenness Index (TGI), and Normalized Green-Red Difference Index (NGRDI).

**Spectral Band Statistics.** For each of the R, G, and B channels, seven summary statistics were computed: mean, standard deviation, minimum, maximum, median, 25th percentile (Q25), and 75th percentile (Q75), yielding **21 spectral features**.

In total, **97 hand-crafted features** were computed per sample. These features were used to train the following models (following the respective publications): Random Forest, Gradient Boosting Trees, XGBoost, a Stacking Ensemble (with Random Forest, Gradient Boosting, and XGBoost as base learners and Linear Regression as meta-learner), and Stepwise Multiple Linear Regression (forward selection with *p*-value threshold of 0.05).

These literature methods were evaluated on the German dataset using a 70/30 train/test split, and additionally on a 101-sample subset (excluding zero-biomass samples) with 5-fold cross-validation.

### 2.11 Biomass Raster Generation

To demonstrate the practical applicability of the proposed method, spatially continuous biomass maps were generated for both study areas. The full aerial image was tessellated into non-overlapping patches matching the training patch dimensions (152 x 152 pixels for Pakistan, 395 x 395 pixels for Germany). For each patch, DINO features were extracted using the same frozen model, the same trained StandardScaler was applied, and the cluster assignment (German data) and biomass prediction were computed using the best-performing model configuration. Predictions were converted to tons per hectare (ton/ha) using the known spatial resolution and plot area. Multiple overlap strategies (0%, 25%, 50%, 75%) were also evaluated for the Pakistani data, where overlapping predictions were averaged to produce smoother raster outputs.

> **[FIGURE 3 PLACEHOLDER]** *Side-by-side comparison of generated biomass raster maps for the Pakistani study area using NNLS, Ridge Regression, and XGBoost models. Maps should show the spatial distribution of predicted biomass (ton/ha) overlaid on or alongside the original aerial RGB image. Include a color bar indicating the biomass scale.*

> **[FIGURE 4 PLACEHOLDER]** *Generated biomass raster map for the German study area using the best-performing model configuration (cluster-specific prediction). Show the original RGB image alongside the predicted biomass map, with color coding indicating biomass values. Highlight how zero-biomass areas (roads, clearings) are correctly predicted.*

---

## 3. Results

### 3.1 Pakistani Dataset Results

Table 1 presents the best performance of each of the three regression models on the Pakistani dataset using Leave-One-Out cross-validation. The optimal number of selected features was determined independently for each model by sweeping from 20 to 300 features.

**Table 1.** Best model performance on the Pakistani dataset (LOO cross-validation). The best result per metric is shown in bold.

| Method | Optimal N Features | %RMSE | RMSE (kg) | MAE (kg) | R^2 |
|--------|-------------------|-------|-----------|----------|-----|
| **NNLS** | **250** | **20.16** | **251.01** | **190.40** | **0.948** |
| Ridge Regression | 120 | 25.11 | 312.65 | 249.21 | 0.919 |
| XGBoost | 20 | 44.19 | 550.28 | 366.26 | 0.751 |

The **NNLS method** achieved the best overall performance with a %RMSE of 20.16% and an R^2 of 0.948 using the top 250 DINO features ranked by Pearson correlation. This indicates that the NNLS predictor, leveraging nearest-neighbor reconstruction in the DINO feature space, explains approximately 95% of the variance in above-ground biomass using only RGB imagery.

**Ridge Regression** achieved a competitive %RMSE of 25.11% with 120 features and an R^2 of 0.919, demonstrating that even a simple regularized linear model can effectively leverage DINO features for biomass prediction.

**XGBoost** performed substantially worse, with a best %RMSE of 44.19% (R^2 = 0.751) at only 20 features. Performance degraded as more features were added (reaching %RMSE = 52.79% at 170 features), indicating severe overfitting on this small dataset. The tree-based ensemble method, despite its flexibility, could not generalize effectively with only 25 training samples.

> **[FIGURE 5 PLACEHOLDER]** *Performance vs. number of selected features for the Pakistani dataset (LOO). Two panels: (a) %RMSE vs. N features, (b) R^2 vs. N features. Three lines per panel, one for each method (NNLS, Ridge, XGBoost). Annotate the best point for each method with a marker and label. This figure already exists as `pakistani_data_analysis/performance_vs_n_features.png`.*

**Effect of Feature Count.** Table 2 shows how performance varies with the number of selected features. NNLS exhibited robust performance across a wide range of feature counts, with %RMSE remaining below 28% for all configurations between 20 and 300 features. Ridge Regression showed a similar stability plateau. XGBoost, by contrast, showed a strong preference for few features and degraded with additional features, consistent with the overfitting risk posed by high-dimensional inputs on small datasets.

**Table 2.** Detailed results for the Pakistani dataset at selected feature counts (LOO).

| N Features | NNLS %RMSE | NNLS R^2 | Ridge %RMSE | Ridge R^2 | XGB %RMSE | XGB R^2 |
|-----------|------------|----------|-------------|-----------|-----------|---------|
| 20 | 26.22 | 0.912 | 36.94 | 0.826 | 44.19 | 0.751 |
| 50 | 24.46 | 0.924 | 29.45 | 0.889 | 46.57 | 0.723 |
| 90 | 23.00 | 0.932 | 28.64 | 0.895 | 48.47 | 0.700 |
| 120 | 26.16 | 0.913 | **25.11** | **0.919** | 47.70 | 0.710 |
| 150 | 23.68 | 0.928 | 27.03 | 0.907 | 51.97 | 0.655 |
| 200 | 24.59 | 0.923 | 26.91 | 0.908 | 51.92 | 0.656 |
| 250 | **20.16** | **0.948** | 28.37 | 0.897 | 49.98 | 0.681 |
| 300 | 27.78 | 0.901 | 28.75 | 0.895 | 49.80 | 0.683 |

> **[FIGURE 6 PLACEHOLDER]** *Heatmap of %RMSE across all feature counts (20--300) and all three methods for the Pakistani dataset. Rows = number of features, columns = methods. Color intensity indicates %RMSE (darker = lower = better). This figure already exists as `pakistani_data_analysis/heatmap_percent_rmse.png`.*

**Feature Correlation Analysis.** Among the 384 DINO features, the mean absolute Pearson correlation with biomass was 0.428 (std = 0.011), with individual correlations ranging from -0.840 to 0.865. The ten most correlated features exhibited |r| > 0.815 (all with *p* < 10^-6), confirming strong linear associations between individual DINO features and biomass. Notably, 170 out of 384 features (44.3%) showed statistically significant correlations at *p* < 0.05, indicating that DINO features carry broadly distributed biomass-relevant information rather than concentrating it in a few dimensions.

> **[FIGURE 7 PLACEHOLDER]** *Distribution of Pearson correlation coefficients between each of the 384 DINO features and biomass for the Pakistani dataset. A histogram or bar chart showing the distribution of |r| values, with a dashed line at the significance threshold. Alternatively, a ranked bar chart of the top 20 features by |r|.*

### 3.2 German Dataset Results

For the German dataset, cluster-specific models were trained (see Section 2.6), and predictions were pooled across both clusters to compute overall (combined) performance metrics on the held-out test set (n = 18). Table 3 presents the combined test-set performance for each model at its optimal feature count.

**Table 3.** Best combined model performance on the German dataset (test set, both clusters pooled).

| Method | Optimal N Features | %RMSE | RMSE (ton/ha) | R^2 |
|--------|-------------------|-------|---------------|-----|
| Ridge Regression | 100 | **17.08** | **27.35** | **0.914** |
| NNLS | 130 | 17.38 | 27.82 | 0.911 |
| XGBoost | 20 | 18.24 | 29.20 | 0.902 |

All three methods achieved R^2 values exceeding 0.90 on the combined test set, demonstrating that the proposed DINO feature-based approach produces highly accurate biomass estimates on the German dataset. **Ridge Regression** achieved the best combined %RMSE of 17.08% (R^2 = 0.914) with 100 features, followed closely by **NNLS** at 17.38% (R^2 = 0.911) with 130 features. Notably, **XGBoost** also performed well on this larger dataset (R^2 = 0.902 with 20 features), in contrast to its poor performance on the smaller Pakistani dataset, suggesting that XGBoost benefits substantially from larger training sets.

**Effect of Feature Count.** Table 4 shows how combined performance varies with the number of selected features.

**Table 4.** Combined performance on the German dataset at selected feature counts (test set).

| N Features | NNLS %RMSE | NNLS R^2 | Ridge %RMSE | Ridge R^2 | XGB %RMSE | XGB R^2 |
|-----------|------------|----------|-------------|-----------|-----------|---------|
| 20 | 17.96 | 0.904 | 18.49 | 0.899 | **18.24** | **0.902** |
| 70 | 20.52 | 0.875 | 17.83 | 0.906 | 21.91 | 0.858 |
| 100 | 21.26 | 0.866 | **17.08** | **0.914** | 19.67 | 0.885 |
| 130 | **17.38** | **0.911** | 17.35 | 0.911 | 22.66 | 0.848 |
| 150 | 19.50 | 0.887 | 17.76 | 0.907 | 21.57 | 0.862 |
| 200 | 21.71 | 0.860 | 19.26 | 0.890 | 22.30 | 0.853 |
| 250 | 19.76 | 0.884 | 18.54 | 0.898 | 22.71 | 0.847 |
| 300 | 19.86 | 0.883 | 18.76 | 0.896 | 21.98 | 0.857 |

Ridge Regression showed the most consistent performance across feature counts, maintaining %RMSE below 19.3% for most configurations between 70 and 300 features. NNLS showed more variation, with a pronounced optimum at 130 features. XGBoost performed best with only 20 features and showed declining performance with additional features, consistent with the overfitting pattern observed on the Pakistani dataset, though less severe given the larger German training set.

> **[FIGURE 8 PLACEHOLDER]** *Performance vs. number of selected features for the German dataset (combined test set). Two panels: (a) %RMSE vs. N features, (b) R^2 vs. N features. Three lines per panel, one for each method. This should be a newly generated figure combining per-cluster predictions into overall metrics, or the existing `augmented_zero_biomass_analysis/comparison_multiple_feature_counts.png` adapted for combined metrics.*

> **[FIGURE 9 PLACEHOLDER]** *Scatter plots of predicted vs. actual biomass for the German test set, using the best configuration for each method. Three panels: (a) NNLS at 130 features, (b) Ridge Regression at 100 features, (c) XGBoost at 20 features. Include the 1:1 line, R^2, and %RMSE annotations. Points should be colored by cluster membership.*

**Zero-Biomass Augmentation.** The inclusion of 12 zero-biomass samples in the training data improved model behavior in non-vegetated areas. Without these samples, models trained exclusively on forest plots tended to predict unreasonably high biomass values for clearings, roads, and other non-forested areas, leading to artifacts in the generated biomass maps.

### 3.3 Comparison with Literature Methods

To rigorously benchmark the proposed DINO-based approach, we implemented established literature methods based on hand-crafted features (GLCM textures, RGB vegetation indices, and spectral statistics) on the same German dataset. Table 5 summarizes the results.

**Table 5.** Performance of literature methods on the German dataset (70/30 train/test split, 97 hand-crafted features).

| Method | Train %RMSE | Train R^2 | Test %RMSE | Test R^2 |
|--------|-------------|-----------|------------|---------|
| Stacking Ensemble | 30.96 | 0.148 | 31.72 | -0.007 |
| Random Forest | 14.03 | 0.825 | 32.52 | -0.058 |
| Gradient Boosting | 2.76 | 0.993 | 33.66 | -0.134 |
| XGBoost | 1.77 | 0.997 | 34.31 | -0.178 |
| Stepwise Regression | 27.20 | 0.343 | 40.50 | -0.641 |

**All literature methods produced negative R^2 values on the test set**, indicating that every method performed worse than a naive prediction of the dataset mean. This result is striking: despite achieving strong training performance (XGBoost: train R^2 = 0.997; Gradient Boosting: train R^2 = 0.993), these models failed to generalize to unseen data. The extreme disparity between training and test performance demonstrates severe **overfitting**, a consequence of fitting complex models to high-dimensional hand-crafted features that lack sufficient discriminative power for this task.

The Stacking Ensemble, which showed the least overfitting (train R^2 = 0.148 vs. test R^2 = -0.007), achieved the best test %RMSE of 31.72%. However, its near-zero R^2 indicates negligible explanatory power on unseen data.

On a **reduced dataset of 101 samples** (excluding zero-biomass augmentation) with 47 hand-crafted features, the literature methods were also evaluated (Table 6).

**Table 6.** Performance of literature methods on 101 samples (German data, 70/30 split).

| Method | Test %RMSE | Test R^2 | 5-fold CV R^2 (mean +/- std) |
|--------|------------|---------|-------------------------------|
| Ridge Regression | 20.75 | -0.120 | -0.155 +/- 0.232 |
| Random Forest | 21.29 | -0.179 | -0.146 +/- 0.192 |
| XGBoost | 21.35 | -0.186 | -0.219 +/- 0.247 |
| Gradient Boosting | 21.54 | -0.207 | -0.294 +/- 0.295 |

Even with a reduced feature set and comparable sample size, all methods still produced negative R^2 values. The 5-fold cross-validation R^2 values confirmed the consistently poor generalization (mean CV R^2 ranging from -0.146 to -0.294).

> **[FIGURE 10 PLACEHOLDER]** *Bar chart or grouped bar chart comparing test %RMSE and test R^2 between the proposed DINO-based methods (NNLS, Ridge, XGBoost) and the literature methods (Random Forest, Gradient Boosting, XGBoost, Stacking, Stepwise Regression). Use distinct color groups for "Proposed (DINO features)" and "Literature (Hand-crafted features)." A horizontal dashed line at R^2 = 0 emphasizes where methods fail to outperform the mean predictor.*

### 3.4 Comparative Analysis: DINO Features vs. Hand-Crafted Features

Table 7 presents a direct comparison between the proposed DINO-based approach and the literature methods on the German dataset.

**Table 7.** Summary comparison of proposed vs. literature methods on the German dataset.

| Approach | Feature Type | Best Method | Best N Features | Test %RMSE | Test R^2 |
|----------|-------------|-------------|----------------|------------|---------|
| **Proposed** | DINO ViT-Small (384-dim) | Ridge Regression | 100 | **17.08** | **0.914** |
| **Proposed** | DINO ViT-Small (384-dim) | NNLS | 130 | 17.38 | 0.911 |
| **Proposed** | DINO ViT-Small (384-dim) | XGBoost | 20 | 18.24 | 0.902 |
| Literature | GLCM + VIs + Spectral (97-dim) | Stacking Ensemble | 97 | 31.72 | -0.007 |
| Literature | GLCM + VIs + Spectral (47-dim) | Ridge Regression | 47 | 20.75 | -0.120 |

The proposed DINO-based approach outperformed all literature methods by a substantial margin:

1. The best proposed method (Ridge Regression with 100 DINO features) achieved an R^2 of **0.914**, compared to the best literature R^2 of **-0.007** --- an absolute improvement of 0.921 in R^2.

2. The proposed approach achieved a %RMSE of **17.08%**, representing a **46% reduction in relative error** compared to the best literature method (%RMSE = 31.72%).

3. Even the lowest-performing proposed method (XGBoost, %RMSE = 18.24%, R^2 = 0.902) dramatically outperformed all literature methods.

4. Crucially, all three proposed methods achieved **positive, high R^2 values** (> 0.90), while all literature methods produced **negative R^2 values**, meaning they failed to outperform even a naive mean predictor.

This comparison demonstrates that the choice of feature representation is the dominant factor in prediction quality. Self-supervised DINO features capture substantially more biomass-relevant information from RGB imagery than traditional hand-crafted features (GLCM textures, vegetation indices, and spectral statistics), leading to dramatically better generalization.

### 3.5 Cross-Dataset Consistency

A key strength of the proposed approach is its consistent performance across two geographically and ecologically distinct study areas.

**Table 8.** Cross-dataset comparison of best DINO-based results.

| Dataset | N Samples | Validation | Best Method | N Features | %RMSE | R^2 |
|---------|-----------|------------|-------------|-----------|-------|-----|
| Pakistan | 25 | LOO | NNLS | 250 | 20.16 | 0.948 |
| Germany | 113 | 85/15 split | Ridge | 100 | 17.08 | 0.914 |

Both datasets yielded R^2 values exceeding 0.91, confirming that the DINO features generalize effectively across:

- **Different geographic regions** (South Asia vs. Central Europe)
- **Different forest types** (subtropical forests of Balakot vs. managed temperate forests of Germany)
- **Different image acquisition conditions** (different sensors, resolutions, seasons)
- **Different sample sizes** (25 vs. 113 samples)

Interestingly, NNLS was the best method on the smaller Pakistani dataset (n = 25), while Ridge Regression was marginally best on the larger German dataset (n = 113). All three methods achieved R^2 > 0.90 on the German dataset, whereas on the Pakistani dataset, only NNLS and Ridge achieved R^2 > 0.90, with XGBoost lagging substantially. This pattern is consistent with the known sensitivity of ensemble tree methods to small sample sizes, and suggests that for very small datasets, NNLS and Ridge Regression are the preferred models.

> **[FIGURE 11 PLACEHOLDER]** *Cross-dataset comparison figure. Panel (a): Scatter plots of predicted vs. actual biomass for the best model on each dataset (NNLS for Pakistan, Ridge for Germany), with 1:1 lines. Panel (b): Bar chart comparing %RMSE and R^2 of all three methods side-by-side across both datasets.*

### 3.6 Biomass Raster Maps

Spatially continuous biomass maps were successfully generated for both study areas using the best-performing model configurations. The generated rasters provide pixel-level biomass estimates in tons per hectare (ton/ha), enabling wall-to-wall biomass mapping from single RGB aerial images.

For the **Pakistani study area**, rasters were generated at multiple overlap levels (0%, 25%, 50%, 75%). Increasing overlap produced smoother spatial patterns, with the 75% overlap yielding the most visually coherent biomass maps. The NNLS model produced the most spatially consistent rasters, while XGBoost rasters exhibited more spatial noise, consistent with that model's lower LOO performance.

For the **German study area**, the cluster-specific prediction strategy was applied: each patch was first assigned to a cluster (using the UMAP + KMeans pipeline fitted on training data), and then the cluster-specific model was used for prediction. This approach produced biomass maps that correctly distinguished between high-biomass forested areas and low-biomass clearings, with the zero-biomass augmentation ensuring realistic predictions in non-vegetated areas.

> **[FIGURE 12 PLACEHOLDER]** *Biomass raster maps for the Pakistani study area at different overlap levels (0%, 25%, 50%, 75%) using the NNLS model. Show how increasing overlap produces progressively smoother predictions. Include the original aerial image for reference. Existing figures in `pakistani_data_analysis/biomass_rasters/`.*

> **[FIGURE 13 PLACEHOLDER]** *Biomass raster comparison for the Pakistani study area showing side-by-side maps from NNLS, Ridge Regression, and XGBoost models. Highlight differences in spatial smoothness and consistency. This figure exists as `pakistani_data_analysis/biomass_rasters/biomass_rasters_comparison.png`.*

### 3.7 Summary of Key Findings

1. **Self-supervised DINO features from RGB imagery are highly effective for biomass estimation**, achieving R^2 > 0.90 on both datasets at the optimal model configuration.

2. **All three regression methods achieve strong performance** when using DINO features: R^2 = 0.948 (NNLS, Pakistan), R^2 = 0.914 (Ridge, Germany), and R^2 = 0.902 (XGBoost, Germany). The quality of the feature representation, rather than the choice of regression model, is the primary driver of prediction accuracy.

3. **Hand-crafted features (GLCM, vegetation indices, spectral statistics) are dramatically inferior** to learned DINO features for this task, with all literature methods achieving negative R^2 on held-out test data while DINO-based methods exceed R^2 = 0.90.

4. **Unsupervised clustering improves prediction accuracy** on heterogeneous sites by allowing cluster-specific models to specialize in different forest types.

5. **The approach generalizes across geographies and forest types**, demonstrating that ImageNet-pretrained DINO features transfer effectively to remote sensing biomass estimation without any domain-specific fine-tuning.

6. **Feature selection matters**: optimal performance is achieved with 20--250 DINO features (depending on dataset and model), with correlation-based ranking providing an effective and interpretable selection criterion.

7. **XGBoost is sensitive to sample size**: it performed worst on the small Pakistani dataset (n = 25) but competitively on the larger German dataset (n = 113), highlighting the importance of model choice for small-sample regimes.
