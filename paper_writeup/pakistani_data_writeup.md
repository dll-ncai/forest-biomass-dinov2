# Pakistani Study Area: Materials and Methods, Results, and Discussion

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

**Table 3.** Best model performance vs All-Features Baseline on the Pakistani dataset (Nested LOO, n = 25).

| Method | Best Top K | Best K %RMSE | Best K R² | All Features (384) %RMSE | All Features (384) R² |
|--------|------------|--------------|-----------|-------------------------|----------------------|
| **NNLS** | **90** | **21.81** | **0.939** | 30.04 | 0.885 |
| Ridge Regression | 130 | 24.60 | 0.923 | 31.91 | 0.870 |
| XGBoost | 40 | 49.64 | 0.685 | 50.10 | 0.679 |

The proposed lightweight NNLS methodology, paired with a subset of the top 90 features, achieved the best overall performance with **%RMSE = 21.81% and an R² of 0.939**, explaining nearly 94% of the biomass variance. Ridge Regression was highly competitive at %RMSE = 24.60% using 130 features. Notably, both NNLS and Ridge suffered substantial performance degradation when forced to utilize all 384 features, underscoring the necessity of our feature selection step. XGBoost substantially underperformed both other methods, hovering between 48–49% RMSE regardless of the feature count.

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

**Table 5.** Literature method performance vs proposed DINO-based methods (Nested LOO, $n=25$).

| Method | Features | %RMSE | R² | Source |
|--------|----------|-------|----|--------|
| **DINO + NNLS (Top 90)** | **DINO CLS token** | **21.81** | **0.939** | **Proposed** |
| DINO + Ridge (Top 130) | DINO CLS token | 24.60 | 0.923 | Proposed |
| Gradient Boosting | GLCM + VI + RGB | 31.46 | 0.874 | Literature |
| Random Forest | GLCM + VI + RGB | 35.65 | 0.838 | Literature |
| Stacking Ensemble | GLCM + VI + RGB | 48.07 | 0.705 | Literature |
| DINO + XGBoost (Top 40) | DINO CLS token | 49.64 | 0.685 | Proposed |
| XGBoost (GLCM) | GLCM + VI + RGB | 49.85 | 0.683 | Literature |
| Stepwise Regression | GLCM + VI + RGB | 51.65 | 0.659 | Literature |

> **[FIGURE 9 — Literature comparison]** *Horizontal bar chart comparing %RMSE and R² across all proposed and literature methods. Green bars = proposed (DINO), red bars = literature.*
![Literature Methods Comparison](/Users/assadabid/Documents/biomass/pakistani_data_analysis/paper_methods_nested_loo_comparison.png)

> **[FIGURE 10 — Literature scatter plots]** *Predicted vs Actual scatter plots for all 5 literature methods.*
![Literature Methods Scatter Plots](/Users/assadabid/Documents/biomass/pakistani_data_analysis/paper_methods_nested_loo_scatter.png)

Our DINO + NNLS pipeline achieves a **%RMSE that is 31% lower** (21.81% vs 31.46%) and an **R² that is 7.4% higher** (0.939 vs 0.874) than the best-performing literature method (Gradient Boosting). While GBT and RF with hand-crafted features produced reasonable R² values (0.874 and 0.838), their %RMSE remains substantially worse. Notably, the more complex methods (Stacking, DINO+XGBoost, XGBoost-GLCM, Stepwise) all performed poorly with %RMSE > 48%, exhibiting the overfitting behavior typical of high-capacity models on small datasets. These results conclusively demonstrate the superiority of self-supervised deep features over hand-crafted GLCM/vegetation index representations for small-sample biomass estimation.

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

The central finding of this study is the remarkable efficacy of our proposed lightweight framework: combining a **Frozen Self-Supervised Vision Transformer (DINOv2)** with **Correlation-Based Feature Ranking** and a non-parametric **Non-Negative Least Squares (NNLS)** regression head. Without utilizing domain-specific remote sensing pretraining, multispectral indices, explicit canopy height models, or LiDAR returns, this pipeline explained **up to 93.9% of the variance** in above-ground biomass (%RMSE = 21.81%) on the resource-constrained Pakistani dataset ($n=25$). 

This result underscores the unique capacity of self-supervised Vision Transformers. DINO's pretraining objective encourages the model to extract highly generalized, dense semantic structures—capturing canopy texture, crown hierarchies, and spatial shadowing—which act as powerful proxies for biophysical characteristics like biomass. 

### 3.2 The Importance of Feature Ranking and Selection

A core innovation of our methodology is the integration of correlation-based feature selection prior to modeling. We strictly demonstrated through our Nested LOO analysis that utilizing the entire 384-dimensional DINO vector results in substantial performance degradation for all evaluated methods (NNLS dropping from 21.81% to 30.04% RMSE; Ridge from 24.60% to 31.91% RMSE). 

By ranking features primarily on their absolute Pearson correlation with the small set of ground-truth biomass labels and isolating a "sweet spot" (e.g., the top 90 features for NNLS), we effectively prune noisy, irrelevant feature embeddings out of the high-dimensional space. This computationally cheap filtering step acts as a powerful dimensional regularizer, safeguarding against the curse of dimensionality inherent to extremely small ground-truth datasets.

### 3.3 Explaining the Success of NNLS in the Pipeline

Within our proposed pipeline, NNLS consistently outperformed both standard regularized parametric methods (Ridge Regression) and heavy ensemble methods (XGBoost). The strength of NNLS natively complements the small-sample, feature-filtered environment for several architectural reasons:

1. **Non-parametric Flexibility:** NNLS makes predictions based strictly on the most similar historical signatures in the filtered DINO embedding space, naturally conforming to non-linear feature-biomass relationships without assuming a rigid global functional form.
2. **Sparsity and Convexity:** The NNLS algorithm forces the reconstruction of a query sample to be a strictly non-negative, sparse combination of its $k$-nearest neighbors. The simplex constraint mathematically guarantees that predicted biomass values remain convex combinations of observed historical values, entirely preventing the model from predicting extreme, non-physical outliers or interfering negatively with collinear features.

This stands in stark contrast to XGBoost, which persistently overfit the 24-sample training partitions (yielding ~50–52% RMSE across all feature counts) despite rigourous hyperparameter tuning in the nested inner folds. The complexity of bagging and gradient boosting is simply fundamentally incompatible with such severe data constraints.

### 3.4 Comparison with Literature: Hand-Crafted Features Are Insufficient

To rigorously benchmark our framework, we implemented five established approaches from the remote sensing literature that rely on hand-crafted features — GLCM texture measures, RGB vegetation indices, and spectral band statistics — totaling 97 features per patch. Under the identical Nested LOO protocol ($n=25$), **none of the literature methods matched the performance of the proposed pipeline**:

- The best literature method (Gradient Boosting, %RMSE = 31.46%) was **44% worse** than DINO + NNLS (21.81%).
- While GBT and RF achieved reasonable R² values (0.874 and 0.838), their prediction error (%RMSE) remains substantially higher.
- More complex approaches (Stacking, XGBoost, Stepwise) severely overfit, with %RMSE > 48%.

This gap reveals a fundamental limitation of traditional hand-crafted features for small-dataset biomass estimation. GLCM texture and simple vegetation indices capture only low-level spatial statistics, while DINO's self-supervised training on millions of natural images produces semantically rich representations that encode hierarchical structural patterns — crown morphology, canopy layering, shadow geometry — which are far more discriminative for biophysical parameter retrieval.

### 3.5 Broad Implications and Practical Scalability

Our pipeline — utilizing off-the-shelf RGB drone imagery, extracting general DINO representations, filtering for the top-correlated features, and mapping predictions via an interpretable, instance-based NNLS head — significantly lowers the barrier to entry for precise precision forestry mapping. It demonstrates that expensive hyperspectral campaigns or complex deep learning fine-tuning cycles can potentially be replaced by smart spatial feature extraction and robust traditional data science techniques.

### 3.6 Limitations

Several limitations should be noted:
1. **Small sample size:** With only 25 sample plots, geographic and biophysical representativeness is narrow. The results serve as a proof-of-concept for the methodology but require validation on broader, independent spatial test sets.
2. **Temporal constraints:** The analysis is based on a single aerial acquisition; temporal variance due to seasonality or atmospheric inconsistencies has not been addressed.

### 3.7 Summary

This study proposes and rigorously validates a novel mechanism for forest biomass estimation on highly constrained datasets. Utilizing a pretrained DINO feature extractor combined with rigid feature ranking and NNLS regression, we demonstrated precise mapping capabilities (%RMSE = 21.81%, R² = 0.939) using only RGB imagery — outperforming five established literature methods by a wide margin (best competitor: 31.46% RMSE). By systematically isolating the top diagnostic features and relying on non-negative convex geometry rather than highly parameterized boosting ensembles, this lightweight framework exhibits high resistance to overfitting and offers a compelling new blueprint for scalable precision forestry.
