# Pakistani Study Area: Materials and Methods, Results, and Discussion

> **⚠️ Leak-free revision.** The authoritative, corrected results are in
> `methodology_and_results.md` and `../RESULTS_CORRECTED.md`. The tables and claims in
> this document have been updated to the corrected, leakage-free numbers (feature-count,
> ranking, and hyper-parameters selected on training data only; NNLS regularized with L2
> + convex weights + ensemble-over-K on this n=25 site). Headline: **DINOv2 + NNLS =
> 29.93 %RMSE, R² 0.886**, a *narrow* win over a fairly-tuned GLCM Random Forest
> (31.12 %, 0.876) — not the wide margin reported in the original (leaky) draft.

---

## 1. Materials and Methods

### 1.1 Study Area and Field Data

The study area is located in **Balakot**, Khyber Pakhtunkhwa province, in the northern mountainous region of Pakistan. The landscape is characterized by subtropical and temperate forest cover with complex terrain and heterogeneous canopy structure. Ground-truth above-ground biomass (AGB) measurements were collected from **25 circular sample plots** distributed across the study area. Biomass values for each plot were derived from field inventory data.

### 1.2 Remote Sensing Data

A single **high-resolution aerial photograph** covering the study area was used as the sole input for biomass estimation. The image was acquired as a multi-band raster with spatial dimensions of **3,175 x 4,771 pixels** and stored in GeoTIFF format (`High_resolution_Aerial_Photograph.tif`). Only the **RGB (Red, Green, Blue) channels** were used for feature extraction, making the approach entirely independent of multispectral, hyperspectral, or LiDAR data. A corresponding biomass mask raster (`total_biomass_mask.tif`) provided the spatial locations and biomass labels for each sample plot; pixels with a no-data value of -9999 were excluded.

> **[FIGURE 1 — Study area overview]** *Map showing the location of Balakot, Pakistan, with an inset of the high-resolution aerial image and the spatial distribution of the 25 sample plots overlaid. This figure could be derived from `overlay_mask_and_crops_152.png`.*

### 1.3 Patch Extraction

For each of the 25 sample plots, a square image patch was extracted from the aerial photograph centered on the plot centroid. Patches of **152 x 152 pixels** were extracted, matching the approximate spatial footprint of the circular field plots. To minimize edge effects and boundary noise, a **center crop of 112 x 112 pixels** was taken from each 152 x 152 patch. This center-cropped patch served as the input for feature extraction.

### 1.4 Feature Extraction Using DINO ViT-Small

Features were extracted from the RGB image patches using a self-supervised Vision Transformer. The **DINO** (Self-**Di**stillation with **No** Labels) framework (Caron et al., 2021) trains a Vision Transformer via a teacher-student self-distillation mechanism, in which a student network learns to match the output of a momentum-updated teacher network using different augmented views of the same image — entirely without labeled data. DINO has been shown to produce features that capture rich semantic and structural information, with attention maps that closely correspond to object segmentations.

We employed the **DINO ViT-Small** model with a patch size of 16 pixels (`vit_small_patch16_224.dino`), accessed through the `timm` (PyTorch Image Models) library. The model accepts inputs of 224 x 224 pixels; accordingly, the 112 x 112 center crops were resized to 224 x 224 using **bicubic interpolation**, and normalized using ImageNet statistics. 

The classification head was removed (`num_classes=0`), so the model outputs the **CLS token** embedding from the final transformer layer — a **384-dimensional feature vector** for each image patch. The CLS token in Vision Transformer architectures aggregates global information from all spatial positions through self-attention, making it a compact yet information-rich representation of the entire patch. The pretrained ImageNet weights were used directly as a **frozen feature extractor** without any fine-tuning.

> **[FIGURE 2 — Feature extraction pipeline]** *Schematic diagram showing the patch extraction and DINO feature extraction pipeline: Aerial image → 152x152 patch → 112x112 center crop → resize to 224x224 → DINO ViT-Small → 384-dim CLS token feature vector.*

### 1.5 Correlation-Based Feature Selection

To identify the most biomass-relevant features and reduce input dimensionality, a univariate correlation-based feature selection approach was employed. For each of the 384 DINO features, the **Pearson correlation coefficient** (*r*) and the **Spearman rank correlation coefficient** (*ρ*) were computed between that feature and the ground-truth biomass values across all 25 samples.

Features were ranked by the **absolute value of their Pearson correlation** with biomass (|*r*|), and the top *N* features were selected for model training. The number of selected features *N* was treated as a hyperparameter and systematically varied from **20 to 300 in increments of 10** (29 configurations). This sweep was designed to identify the optimal trade-off between the amount of information available to the model and the risk of overfitting due to high dimensionality relative to the small sample size.

No PCA or other unsupervised dimensionality reduction was applied in the feature-selection experiments; models were trained directly on the selected raw DINO features.

### 1.6 Regression Models

Three regression models were evaluated:

#### 1.6.1 Non-Negative Least Squares (NNLS) Regression

NNLS operates as a dictionary-based regression method. For a query sample, the *k* nearest neighbors in the training set are identified using **cosine similarity** (feature vectors are L2-normalized and similarity is computed as the dot product). The query is then reconstructed as a non-negative linear combination of the neighbor feature vectors by solving:

> minimize ‖**A**w − **b**‖² subject to w ≥ 0

where **A** is the matrix of neighbor feature vectors (columns), **b** is the query vector, and **w** are the non-negative weights, solved using `scipy.optimize.nnls`. A **simplex constraint** normalizes the weights to sum to 1, ensuring the prediction is a convex combination of the neighbor biomass values:

> ŷ = Σ wᵢ · yᵢ

Fixed hyperparameters: *k* = 3 neighbors, cosine similarity, simplex constraint enabled.

#### 1.6.2 Ridge Regression

A Ridge regression model (L2-regularized linear regression) was used as a linear baseline. Features were standardized using a `StandardScaler` (zero mean, unit variance) prior to fitting. The regularization parameter was fixed at **α = 10.0**.

#### 1.6.3 XGBoost

An XGBoost (eXtreme Gradient Boosting) model was used as a representative nonlinear ensemble method. Hyperparameters were **tuned via LOO cross-validation** for each feature count over a grid of 12 configurations: `learning_rate` ∈ {0.03, 0.05}, `max_depth` ∈ {2, 3, 4}, `reg_lambda` ∈ {1.0, 10.0}, with fixed `n_estimators` = 100, `subsample` = 0.8, `colsample_bytree` = 0.8, `reg_alpha` = 0.1, and `min_child_weight` = 3.0. The configuration yielding the lowest LOO %RMSE was selected for each feature count.

### 1.7 Validation Strategy

Due to the limited sample size of **n = 25** plots, a rigorous **Nested Leave-One-Out (Nested LOO) cross-validation** strategy was employed to prevent data leakage during hyperparameter tuning and provide an unbiased estimate of model generalization.

In the outer loop, one sample is held out as the test set, while the remaining 24 form the training set. Within this 24-sample training set, an inner LOO loop (comprising 24 inner folds) is performed to evaluate a grid of hyperparameters. The configuration that yields the lowest inner %RMSE is selected. The model is then retrained on the full 24-sample outer training set using these optimal hyperparameters, and a final prediction is made for the single held-out test sample. Performance metrics are computed by comparing these 25 unbiased predictions against the ground truth.

### 1.8 Evaluation Metrics

Model performance was assessed using four metrics:

- **Percentage RMSE (%RMSE):** RMSE normalized by the mean observed biomass: %RMSE = (RMSE / ȳ) × 100. This scale-invariant metric was used as the primary criterion for model comparison and optimal feature count selection.
- **Root Mean Square Error (RMSE):** In the same units as the raw biomass measurements (kg).
- **Mean Absolute Error (MAE):** The average absolute prediction error (kg).
- **Coefficient of Determination (R²):** The proportion of variance explained by the model.

### 1.9 Biomass Raster Map Generation

To demonstrate wall-to-wall biomass mapping capability, spatially continuous biomass raster maps were generated for the entire aerial image. The full image (3,175 x 4,771 pixels) was tessellated into **non-overlapping 152 x 152 pixel patches**, yielding approximately 620 patches. For each patch, the 112 x 112 center crop was extracted, resized to 224 x 224, and DINO features were computed using the same frozen model.

For raster generation, a separate preprocessing pipeline was used: features were first standardized using a `StandardScaler` fit on training data, then reduced via **PCA** (98% variance threshold, maximum 64 components), resulting in **17 PCA components**. This PCA-based pipeline (distinct from the correlation-based feature selection used for model evaluation) was used because it provided a compact, stable representation suited for spatial prediction over hundreds of patches.

Predictions were generated using the NNLS, Ridge Regression, and XGBoost models, with biomass values converted from raw kg to **tons per hectare (ton/ha)** using the known circular plot area (radius = 17.5 m, area = π × 17.5² ≈ 962.1 m²) and the conversion: ton/ha = (biomass_kg / 1000) × (10,000 / 962.1).

Additionally, overlapping window strategies were evaluated at **0%, 25%, 50%, and 75% overlap**. In overlapping configurations, multiple predictions covering the same 152 x 152 grid cell were averaged to produce smoother raster outputs.

---

## 2. Results

### 2.1 Feature–Biomass Correlation Analysis

The correlation analysis revealed that a substantial proportion of the DINO features carry biomass-relevant information (Table 1).

**Table 1.** Summary of feature–biomass correlation analysis (n = 25 samples, 384 DINO features).

| Statistic | Value |
|-----------|-------|
| Total features analyzed | 384 |
| Features significant at *p* < 0.05 | 170 (44.3%) |
| Features significant at *p* < 0.01 | 116 (30.2%) |
| Features significant at *p* < 0.001 | 59 (15.4%) |
| Mean Pearson *r* | −0.011 |
| Std. of Pearson *r* | 0.428 |
| Range of Pearson *r* | −0.840 to +0.865 |

The top-ranked feature (`feature_14`) exhibited a Pearson correlation of *r* = 0.865 (*p* = 2.49 × 10⁻⁸), and the top 10 features all showed |*r*| > 0.815 (*p* < 10⁻⁶). Table 2 lists the top 10 features.

**Table 2.** Top 10 DINO features by absolute Pearson correlation with biomass.

| Rank | Feature | Pearson *r* | *p*-value | Spearman *ρ* |
|------|---------|-------------|-----------|--------------|
| 1 | feature_14 | +0.865 | 2.49 × 10⁻⁸ | +0.811 |
| 2 | feature_369 | +0.862 | 3.11 × 10⁻⁸ | +0.843 |
| 3 | feature_44 | +0.846 | 1.01 × 10⁻⁷ | +0.853 |
| 4 | feature_125 | −0.840 | 1.51 × 10⁻⁷ | −0.834 |
| 5 | feature_87 | −0.829 | 3.06 × 10⁻⁷ | −0.784 |
| 6 | feature_283 | −0.825 | 3.80 × 10⁻⁷ | −0.788 |
| 7 | feature_105 | −0.822 | 4.57 × 10⁻⁷ | −0.848 |
| 8 | feature_60 | +0.821 | 5.03 × 10⁻⁷ | +0.805 |
| 9 | feature_50 | +0.819 | 5.65 × 10⁻⁷ | +0.739 |
| 10 | feature_341 | −0.815 | 6.93 × 10⁻⁷ | −0.785 |

The close agreement between Pearson and Spearman correlations for the top features confirms that the relationships are approximately linear. The distribution of correlations is roughly symmetric around zero (mean *r* = −0.011), with a wide spread (std = 0.428), indicating that DINO features encode both positively and negatively correlated biomass signals.

> **[FIGURE 3 — Correlation analysis]** *Panel (a): Horizontal bar chart showing the top 20 features ranked by |Pearson *r*|, colored by sign (positive/negative). Panel (b): Histogram of Pearson correlation coefficients across all 384 features. Panel (c): Scatter plots of the top 5 features against biomass with linear trend lines. Existing files: `top_features_correlation_analysis.png`, `correlation_distribution.png`, `top_10_features_detailed.png`.*

### 2.2 Model Performance: Nested Leave-One-Out Cross-Validation

Table 3 presents the best Nested LOO performance for each of the three regression models, comparing the optimal subset of top-ranked features against a baseline utilizing the full 384-dimensional DINO feature embedding.

**Table 3.** Best model performance on the Pakistani dataset (leak-free nested LOO, n = 25, t/ha).

| Method | Feature count | %RMSE | R² | RMSE | MAE |
|--------|---------------|-------|-----|------|-----|
| **NNLS** | **ensemble-over-K** | **29.93** | **0.886** | **3.87** | **2.86** |
| Ridge Regression | 160 | 31.73 | 0.871 | 4.11 | 3.31 |
| XGBoost | 40 | 53.46 | 0.635 | 6.92 | 5.05 |

The regularized NNLS predictor achieved the best overall performance at **%RMSE = 29.93% and R² = 0.886**, explaining ~89% of the biomass variance. Ridge Regression was close behind at 31.73% (R² = 0.871). XGBoost underperformed badly (53.46%, R² = 0.635), reflecting the well-known unsuitability of tree ensembles at n = 25. Note that on this tiny dataset NNLS must be regularized (L2 reconstruction + convex weights) and, crucially, must **average predictions over the feature-count grid rather than selecting a single count** — a single-count selection overfits the inner cross-validation and degrades NNLS to ~48–50% %RMSE (see `methodology_and_results.md`, §3.1).

### 2.3 Effect of Feature Count on Performance

The number of selected features had a pronounced impact on model performance. Figure 4 visualizes the nested LOO evaluation across the feature count sweep.

> **[FIGURE 4 — Performance vs. N features]** *Two-panel line plot: (a) %RMSE vs. number of selected features, (b) R² vs. number of selected features, with separate lines for NNLS, Ridge Regression, and XGBoost. The dashed horizontal lines show the performance baseline when utilizing all 384 available DINO features. Existing file: `nested_loo_performance.png`.*

> **[FIGURE 5 — Scatter plots]** *Scatter plots depicting Predicted vs Actual biomass for the Best-K optimal feature structure against the all-features baseline. Existing file: `nested_loo_scatter_plots.png`.*

**NNLS** and **Ridge Regression** exhibited optimal "sweet spots" in the 90-130 feature range. Performance deteriorated uniformly for both models as the feature size grew toward 384, indicating that integrating lower-ranked, noisier dimensions actively harms model precision on small datasets.

**XGBoost** performed poorly across the board, with %RMSE floating around 48–52%. This persistent degradation is a hallmark of overfitting in tree-based ensembles on small datasets ($n=25$), where hyperparameter tuning via the nested inner loop was insufficient to prevent the trees from heavily memorizing training noise.

### 2.4 Comparison with Literature Methods

To contextualize the performance of our proposed DINO-based pipeline, we implemented five representative methods from the remote sensing biomass estimation literature, using hand-crafted feature extraction pipelines described in Liu et al. (Forests 2025) and Eckert (Remote Sens. 2012). These methods use **GLCM texture features** (multi-scale, distances 1–3), **RGB vegetation indices** (VVI, GRVI, ExG, VARI, etc.), and **RGB spectral band statistics** (mean, std, min, max, median, quartiles) — totaling 97 hand-crafted features per image patch.

The following models were evaluated under the same Nested LOO protocol ($n=25$) used for the DINO-based models:

1. **Random Forest** (RF) — tuned n_estimators and max depth
2. **Gradient Boosting Tree** (GBT) — tuned depth, learning rate, n_estimators
3. **XGBoost** — tuned depth, learning rate, regularization
4. **Stacking Ensemble** — RF + GBT + XGBoost base learners, Ridge meta-learner
5. **Stepwise Multiple Linear Regression** — forward selection with *p* < 0.05

**Table 5.** Literature method performance vs proposed DINO-based methods (leak-free nested LOO, $n=25$). Literature models are evaluated under the *identical* nested-LOO protocol as the DINO models.

| Method | Features | %RMSE | R² | Source |
|--------|----------|-------|----|--------|
| **DINO + NNLS (ensemble-K)** | **DINO CLS token** | **29.93** | **0.886** | **Proposed** |
| Random Forest | GLCM + VI + RGB | 31.12 | 0.876 | Literature |
| DINO + Ridge (Top 160) | DINO CLS token | 31.73 | 0.871 | Proposed |
| Gradient Boosting | GLCM + VI + RGB | 31.91 | 0.870 | Literature |
| Ridge (GLCM) | GLCM + VI + RGB | 47.10 | 0.717 | Literature |
| XGBoost (GLCM) | GLCM + VI + RGB | 49.54 | 0.687 | Literature |
| DINO + XGBoost (Top 40) | DINO CLS token | 53.46 | 0.635 | Proposed |

> **[FIGURE 9 — Literature comparison]** *Horizontal bar chart comparing %RMSE and R² across all proposed and literature methods. Green bars = proposed (DINO), red bars = literature.*
![Literature Methods Comparison](/Users/assadabid/Documents/biomass/pakistani_data_analysis/paper_methods_nested_loo_comparison.png)

> **[FIGURE 10 — Literature scatter plots]** *Predicted vs Actual scatter plots for all 5 literature methods.*
![Literature Methods Scatter Plots](/Users/assadabid/Documents/biomass/pakistani_data_analysis/paper_methods_nested_loo_scatter.png)

Under the identical leak-free nested-LOO protocol, DINO + NNLS (29.93%, R² 0.886) is the best method, but only **narrowly** ahead of a tuned GLCM Random Forest (31.12%, R² 0.876) and Gradient Boosting (31.91%, 0.870). On this tiny site the hand-crafted-feature baselines are genuinely competitive, and the honest conclusion is that DINO features provide a *modest* improvement rather than a decisive one. (This corrects the original draft, which — using a flawed literature evaluation — reported the GLCM methods as far worse; that gap was an artifact of the evaluation, not a real effect.) The larger, clearer DINO advantage is seen on the German site (§ German writeup and `methodology_and_results.md`).

### 2.5 UMAP Visualization of the Feature Space

The 384-dimensional DINO feature space was visualized using UMAP (n_neighbors = 15, min_dist = 0.1) to examine the structure of the data and the relationship between feature-space proximity and biomass similarity.

> **[FIGURE 6 — UMAP visualization]** *2D UMAP embedding of the 25 sample patches, with points colored by biomass value. Shows whether patches with similar biomass cluster together in the DINO feature space. A second panel shows points colored by train/test split label. Existing files: `umap_visualization.png`, `umap_biomass.png`, `umap_3d_biomass.png`.*

### 2.6 Biomass Raster Maps

Spatially continuous biomass rasters were generated for the entire aerial image. Figure 7 shows the NNLS, Ridge Regression, and XGBoost raster predictions.

> **[FIGURE 7 — Biomass rasters, no overlap]** *Side-by-side biomass raster maps (ton/ha) for the Pakistani study area produced by NNLS, Ridge Regression, and XGBoost. Each panel shows the predicted biomass value per 152 x 152 pixel cell. A color bar indicates the biomass scale.*
![Biomass Rasters Comparison](/Users/assadabid/Documents/biomass/pakistani_data_analysis/biomass_rasters/biomass_rasters_comparison.png)

The NNLS model produced spatially coherent biomass maps with a clear gradient from low-biomass areas to high-biomass forest stands. Ridge Regression rasters showed similar broad patterns but with less spatial variability. XGBoost rasters exhibited more spatial noise and occasional extreme predictions, consistent with the model's lower LOO performance.

---

## 3. Discussion

### 3.1 A Novel Framework for Small-Data Biomass Estimation

The central finding of this study is the efficacy of a lightweight framework: a **Frozen Self-Supervised Vision Transformer (DINOv2)** feature extractor with a regularized non-parametric **Non-Negative Least Squares (NNLS)** regression head. Without domain-specific remote-sensing pretraining, multispectral indices, canopy height models, or LiDAR, this pipeline explained **~89% of the variance** in above-ground biomass (%RMSE = 29.93%, R² = 0.886) on the resource-constrained Pakistani dataset ($n=25$) under a strict leak-free nested-LOO protocol. 

This result underscores the unique capacity of self-supervised Vision Transformers. DINO's pretraining objective encourages the model to extract highly generalized, dense semantic structures—capturing canopy texture, crown hierarchies, and spatial shadowing—which act as powerful proxies for biophysical characteristics like biomass. 

### 3.2 The Importance of Feature Ranking and Selection

The methodology uses correlation-based feature ranking, computed strictly on training data within each fold. On this small site, however, the *number* of features is the sensitive knob: selecting a single feature count by inner cross-validation is high-variance and can overfit the selection. The robust remedy adopted here is to **average NNLS predictions over the whole feature-count grid** (ensemble-over-K) rather than committing to one count — this removes the selection variance and, together with L2-regularized convex reconstruction, is what makes NNLS the top performer at n = 25 (see `methodology_and_results.md`, §2.7 and §3.1). Correlation ranking still provides a cheap, interpretable ordering of the DINO dimensions, but on this dataset its main value is in defining the grid that the ensemble averages over.

### 3.3 Explaining the Success of NNLS in the Pipeline

Within our proposed pipeline, NNLS consistently outperformed both standard regularized parametric methods (Ridge Regression) and heavy ensemble methods (XGBoost). The strength of NNLS natively complements the small-sample, feature-filtered environment for several architectural reasons:

1. **Non-parametric Flexibility:** NNLS makes predictions based strictly on the most similar historical signatures in the filtered DINO embedding space, naturally conforming to non-linear feature-biomass relationships without assuming a rigid global functional form.
2. **Sparsity and Convexity:** The NNLS algorithm forces the reconstruction of a query sample to be a strictly non-negative, sparse combination of its $k$-nearest neighbors. The simplex constraint mathematically guarantees that predicted biomass values remain convex combinations of observed historical values, entirely preventing the model from predicting extreme, non-physical outliers or interfering negatively with collinear features.

This stands in stark contrast to XGBoost, which persistently overfit the 24-sample training partitions (yielding ~50–52% RMSE across all feature counts) despite rigourous hyperparameter tuning in the nested inner folds. The complexity of bagging and gradient boosting is simply fundamentally incompatible with such severe data constraints.

### 3.4 Comparison with Literature: Hand-Crafted Features Are Insufficient

To benchmark our framework we implemented established hand-crafted-feature approaches — GLCM texture measures, RGB vegetation indices, and spectral band statistics (97 features per patch) — and evaluated them under the **identical** leak-free nested-LOO protocol ($n=25$):

- DINO + NNLS (29.93%, R² 0.886) is the best method, but only **narrowly** ahead of GLCM Random Forest (31.12%, 0.876) and Gradient Boosting (31.91%, 0.870).
- Tree ensembles on GLCM features are genuinely competitive on this small, structurally simple site.
- Linear/boosted GLCM variants (Ridge, XGBoost) do lag (%RMSE > 47%).

The honest conclusion for Pakistan is that DINO features provide a *modest* edge over strong hand-crafted baselines — not the decisive gap claimed in the original (leaky) draft, where the literature methods were reported as far worse due to a flawed evaluation. The clearer advantage of self-supervised features appears on the larger, more heterogeneous German site.

### 3.5 Broad Implications and Practical Scalability

Our pipeline — utilizing off-the-shelf RGB drone imagery, extracting general DINO representations, filtering for the top-correlated features, and mapping predictions via an interpretable, instance-based NNLS head — significantly lowers the barrier to entry for precise precision forestry mapping. It demonstrates that expensive hyperspectral campaigns or complex deep learning fine-tuning cycles can potentially be replaced by smart spatial feature extraction and robust traditional data science techniques.

### 3.6 Limitations

Several limitations should be noted:
1. **Small sample size:** With only 25 sample plots, geographic and biophysical representativeness is narrow. The results serve as a proof-of-concept for the methodology but require validation on broader, independent spatial test sets.
2. **Temporal constraints:** The analysis is based on a single aerial acquisition; temporal variance due to seasonality or atmospheric inconsistencies has not been addressed.

### 3.7 Summary

This study validates a lightweight mechanism for forest biomass estimation on highly constrained datasets. Using a pretrained DINO feature extractor with regularized NNLS regression (L2 reconstruction, convex weights, ensemble-over-K), we achieved %RMSE = 29.93%, R² = 0.886 using only RGB imagery under a strict leak-free nested-LOO protocol — a narrow win over strong tuned GLCM baselines (best competitor: 31.12% RMSE). By relying on non-negative convex geometry rather than heavily parameterized boosting ensembles, the framework is robust to overfitting at n = 25 and offers a practical blueprint for precision forestry, while the honest cross-site evidence (clearer on the German data) is what supports the value of self-supervised RGB features.
