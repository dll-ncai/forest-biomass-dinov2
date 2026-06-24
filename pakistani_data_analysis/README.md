# Pakistani Dataset Analysis (German Methodology)

**Folder location:**  
`/home/assad/biomass/pakistani_data_analysis`

This folder applies the same methodology used for the **German data** to the **Pakistani dataset**:

1. **Feature–biomass correlation**: Pearson and Spearman correlation of each feature with biomass; rank features by absolute correlation.
2. **UMAP visualization**: 2D and 3D UMAP of the feature space, colored by biomass and by train/test (or train/val) split.

**Full features** are used by default: `dino_features_with_labels_and_split.csv` (384 DINO features: `feature_0` … `feature_383`, plus `label` and `split`), from the repo root. To use PCA-reduced features instead, pass the path to your PCA CSV when running the scripts.

## Requirements

- Python 3 with: `pandas`, `numpy`, `matplotlib`, `scipy`, `umap-learn`
- Input CSV: feature columns (e.g. `pca_0`, … or `feature_0`, …), plus `label` (or `biomass`) and optionally `split`.

## Usage

### Run full analysis (correlation + UMAP)

From this directory:

```bash
./run_analysis.sh
```

Or with a custom input CSV:

```bash
./run_analysis.sh /path/to/dino_features_reduced_pca_train_only.csv
```

Default input is `../dino_features_with_labels_and_split.csv` (full features; relative to this folder).

### Run steps separately

**1. Feature–biomass correlation and ranking**

```bash
python3 feature_biomass_correlation_analysis.py \
  --input_csv ../dino_features_with_labels_and_split.csv \
  --output_dir . \
  --top_n 20
```

Outputs:

- `feature_correlations.csv` – all features with Pearson/Spearman stats, sorted by |Pearson|
- `correlation_summary.json` – counts of significant features, top 10 list, correlation stats
- `top_features_correlation_analysis.png` – bar plots and top-5 scatter plots
- `top_10_features_detailed.png` – scatter plots for top 10 features vs biomass
- `correlation_distribution.png` – histograms of Pearson and Spearman correlations

**2. UMAP 2D and 3D**

```bash
python3 visualize_with_umap.py \
  --input_csv ../dino_features_with_labels_and_split.csv \
  --output_dir . \
  --n_neighbors 15 \
  --min_dist 0.1
```

Options:

- `--no_3d` – only 2D UMAP
- `--no_ids` – do not draw sample IDs on points

Outputs:

- `umap_visualization.png` – 2×2 panel (biomass, split, overlay, distribution)
- `umap_biomass.png` – 2D UMAP colored by biomass
- `umap_train_test.png` – 2D UMAP by train/test (if `split` column present)
- `umap_3d_biomass.png` – 3D UMAP colored by biomass
- `umap_3d_train_test.png` – 3D UMAP by split (if present)
- `umap_3d_multiple_views.png` – four 3D views

## Relation to German workflow

- **German**: `german_data_results/feature_biomass_correlation_analysis.py` and `feature_biomass_correlation_results/`; `visualize_with_umap.py` and `umap_full_features_results/`.
- **Pakistani**: This folder reuses that workflow on the Pakistani CSV and keeps all outputs here. **Default input is full features** (`dino_features_with_labels_and_split.csv`). Feature columns are detected automatically (any numeric column except `label`/`biomass`/`split`), so it works with full `feature_0`…`feature_383` or PCA `pca_0`… if you pass a different CSV.
