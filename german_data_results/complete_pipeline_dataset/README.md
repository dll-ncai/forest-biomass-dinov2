# Complete Pipeline Dataset

This folder contains the complete dataset for cluster-based biomass prediction, prepared with the full pipeline.

## Dataset Overview

- **Total Clusters**: 101
- **Total Plots**: 303 (grouped into clusters)
- **Mean Plots per Cluster**: 3.00
- **Train Samples**: 71 clusters
- **Test Samples**: 30 clusters
- **Features**: 50 (PCA-reduced from 384, 98.11% variance preserved)

### Biomass Statistics
- **Range**: 100.88 - 302.01 ton/ha
- **Mean**: 196.01 ton/ha
- **Std**: 43.44 ton/ha

## Files in This Folder

### 1. `dino_features_with_labels_and_split.csv`
- **Description**: Raw DINO v2 features (384 dimensions)
- **Format**: CSV with columns: `feature_0` through `feature_383`, `label`, `split`
- **Use**: For experiments with raw DINO features

### 2. `dino_features_pca_with_labels_and_split.csv` ⭐ **RECOMMENDED**
- **Description**: PCA-reduced DINO features (50 dimensions, 98.11% variance preserved)
- **Format**: CSV with columns: `feature_0` through `feature_49`, `label`, `split`
- **Use**: **Primary dataset for modeling** - optimized for maximum information retention
- **PCA Info**:
  - Original dimension: 384
  - Reduced dimension: 50
  - Explained variance: 98.11%
  - Variance threshold: 98.0%

### 3. `cluster_info.json`
- **Description**: Detailed information about each cluster
- **Contains**:
  - Cluster IDs and number of plots per cluster
  - Mean biomass per cluster
  - Individual plot biomasses used to calculate mean
  - PCA information

### 4. `pca_model.pkl`
- **Description**: Trained PCA model (pickle format)
- **Use**: For transforming new data using the same PCA transformation
- **Load**: `import pickle; pca = pickle.load(open('pca_model.pkl', 'rb'))`

### 5. `dataset_summary.txt`
- **Description**: Human-readable summary of the dataset

### 6. `crops/` (directory)
- **Description**: 101 PNG images (224x224 pixels each)
- **Naming**: `cluster_{id:03d}_biomass_{value:.2f}.png`
- **Use**: Visual inspection, debugging, or re-extracting features

## Pipeline Steps

1. **Clustering**: 303 plots → 101 clusters using spatial K-means
2. **Crop Extraction**: 224x224 image crops centered on cluster centroids
3. **Biomass Calculation**: Mean biomass of plots in each cluster
4. **Feature Extraction**: DINO v2 Small model (384 features)
5. **PCA Reduction**: 384 → 50 dimensions (98.11% variance preserved)

## Usage Example

```python
import pandas as pd
import numpy as np

# Load PCA features (recommended)
df = pd.read_csv('dino_features_pca_with_labels_and_split.csv')

# Separate features and labels
feature_cols = [c for c in df.columns if c.startswith('feature_')]
X = df[feature_cols].values
y = df['label'].values
splits = df['split'].values

# Split into train/test
X_train = X[splits == 'train']
X_test = X[splits == 'test']
y_train = y[splits == 'train']
y_test = y[splits == 'test']

print(f"Train: {len(X_train)} samples, {X_train.shape[1]} features")
print(f"Test: {len(X_test)} samples, {X_test.shape[1]} features")
```

## Key Features

✅ **Proper Cluster Aggregation**: Mean of ~3 plots per cluster (reduces noise)  
✅ **224x224 Crops**: Exact size required for DINO v2 model  
✅ **DINO Features**: Pre-trained visual features (384 dim)  
✅ **PCA Optimization**: 50 dimensions preserve 98.11% variance  
✅ **Stratified Split**: Train/test split maintains biomass distribution  

## Next Steps

1. Use `dino_features_pca_with_labels_and_split.csv` for modeling
2. Apply machine learning methods (NNLS, Ridge, XGBoost, etc.)
3. Evaluate on test set
4. Use `pca_model.pkl` to transform new data if needed

## Notes

- All crops are exactly 224x224 pixels
- Biomass values are mean of plots in each cluster
- PCA was fit only on training data (proper methodology)
- Train/test split is stratified by biomass distribution

