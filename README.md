# Forest Above-Ground Biomass Estimation with Self-Supervised DINOv2 RGB Features

Estimating forest above-ground biomass (AGB) from high-resolution RGB imagery using
frozen **DINOv2 ViT-Small** CLS-token features (384-dim) instead of hand-crafted
texture (GLCM) or vegetation-index (VI) descriptors. The study evaluates the approach
on two independent sites — **Balakot, Pakistan** and **Karlsruhe, Germany** — and shows
that self-supervised RGB features outperform classical literature baselines.

## Key results

### Pakistani site (Balakot) — nested leave-one-out, n = 25

| Method | Top-K features | %RMSE | R² |
|--------|----------------|-------|-----|
| **DINO + NNLS** | **90** | **21.81** | **0.939** |
| DINO + Ridge | 130 | 24.60 | 0.923 |
| Gradient Boosting (literature) | – | 31.46 | 0.874 |
| Random Forest (literature) | – | 35.65 | 0.838 |

### German site (Karlsruhe) — 85/15 hold-out, n_test = 17

| Method | Top-K features | %RMSE | R² |
|--------|----------------|-------|-----|
| **DINO + NNLS** | **20** | **13.85** | **0.897** |
| DINO + XGBoost | 30 | 17.39 | 0.838 |
| DINO + Ridge | 40 | 18.44 | 0.817 |
| Random Forest (literature) | – | 20.70 | 0.770 |

Full method descriptions and tables are in [`paper_writeup/`](paper_writeup/).

## Repository structure

```
forest-biomass-dinov2/
├── dino_features_with_labels_and_split.csv   # Pakistani DINOv2 features (read by pakistani scripts from repo root)
├── pakistani_data_analysis/                  # Balakot pipeline: scripts, results, figures
├── german_data_results/                      # Karlsruhe pipeline
│   ├── complete_pipeline_dataset/            #   input feature CSVs, image crops, cluster info
│   └── augmented_zero_biomass_analysis/      #   results (JSON/CSV) + figures + ablations
├── paper_writeup/                            # Markdown write-ups + paper figures (PNG/SVG/PDF)
├── requirements.txt
└── .gitignore
```

> **Note on file layout:** the scripts use paths relative to their own location, so the
> directory structure above must be preserved for them to run without edits. In
> particular, `dino_features_with_labels_and_split.csv` must stay at the repository root
> (the Pakistani scripts read it from their parent directory), and `paper_writeup/` must
> stay one level above `german_data_results/` (figure scripts write into it).

## Setup

```bash
pip install -r requirements.txt
```

Developed with a conda environment (`conda create -n biomass python=3.11` then the
packages in `requirements.txt`).

## Reproducing the results

### Pakistani site
```bash
cd pakistani_data_analysis
python run_nested_loo.py                  # DINO nested-LOO feature-count sweep
python run_paper_methods_nested_loo.py    # literature baselines (GLCM/VI methods)
python regenerate_comparison_figure.py    # DINO-vs-literature comparison figure
```

### German site
```bash
cd german_data_results
python run_multiple_feature_counts.py     # main feature-count sweep (cluster-separated)
python run_ablation_all_features.py       # ablation 1: all 384 features
python run_ablation_no_clustering.py      # ablation 2: global model, no clustering
python regenerate_german_figures.py       # performance-vs-features + scatter figures
```

All numeric results (`results_top*.json`, `*_results.csv`) and figures from the paper are
included, so the analysis and figure scripts run from the committed CSVs without rerunning
feature extraction.

## Data availability

The large **GeoTIFF biomass maps and source rasters are not tracked in git** (each is
well above GitHub's 100 MB limit; together ~6 GB). These are *generated outputs* — the
predicted-biomass map rasters and the raw aerial/mask imagery used to build the datasets.
The rendered versions of the biomass maps are included as figures in
`paper_writeup/` (e.g. `fig4_biomass_raster_maps.png`). The committed feature CSVs are
sufficient to reproduce every metric and figure reported in the paper.
