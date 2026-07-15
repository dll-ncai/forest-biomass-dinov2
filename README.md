# Forest Above-Ground Biomass Estimation with Self-Supervised DINOv2 RGB Features

Estimating forest above-ground biomass (AGB) from high-resolution RGB imagery using
frozen **DINOv2 ViT-Small** CLS-token features (384-dim) instead of hand-crafted
texture (GLCM) or vegetation-index (VI) descriptors. The study evaluates the approach
on two independent sites — **Balakot, Pakistan** and **Karlsruhe, Germany** — and shows
that self-supervised RGB features outperform classical literature baselines.

> **⚠️ Leak-free revision.** The numbers below are the **corrected, leak-free** results.
> The original pipeline inflated performance through data leakage — feature-count *K*,
> feature ranking, and the clustering were all fit using **all** samples (including the
> test set). The corrected pipeline selects everything (K, ranking, clustering, and every
> model hyper-parameter) on **training data only** (nested LOO for Pakistan; 5-fold CV on
> the train split for Germany). Full corrected tables: [`RESULTS_CORRECTED.md`](RESULTS_CORRECTED.md);
> methodology and per-site write-ups: [`paper_writeup/`](paper_writeup/) and
> [`PROJECT_NOTES.md`](PROJECT_NOTES.md). The pre-correction numbers (Pakistan 21.81 %RMSE,
> Germany 13.85 %RMSE) are **superseded and should not be cited.**

## Key results (leak-free)

### Pakistani site (Balakot) — nested leave-one-out, n = 25

| Method | K | %RMSE | R² |
|--------|----|-------|-----|
| **DINO + NNLS (ensemble-over-K)** | ensemble | **29.93** | **0.886** |
| DINO + Ridge | 160 | 31.73 | 0.871 |
| Random Forest (literature GLCM) | – | 31.12 | 0.876 |
| Gradient Boosting (literature GLCM) | – | 31.91 | 0.870 |

DINO + NNLS is the best model, but only **narrowly** ahead of the tuned GLCM baselines on
this tiny site — an honest, modest improvement rather than the decisive gap the leaky draft
reported.

### German site (Karlsruhe) — 85/15 hold-out, n_test = 17

| Method | %RMSE | R² |
|--------|-------|-----|
| **DINO + NNLS** | **17.40** | **0.837** |
| DINO + Ridge | 23.83 | 0.695 |
| Ridge (literature GLCM) | 22.79 | 0.721 |
| Random Forest (literature GLCM) | 23.70 | 0.698 |

The DINO advantage is clearer on Germany (~24 % relative error reduction over the best
literature baseline). A single 85/15 split is optimistic, however: over **20 stratified
splits** DINO + NNLS averages **23.0 ± 3.7 %RMSE**, still ahead of the best literature
baseline (24.4 ± 2.5) but by a modest, honest margin. NNLS is the best regressor on **both**
sites; DINOv2 features beat ResNet50 features throughout.

Full method descriptions and tables are in [`paper_writeup/`](paper_writeup/).

## Repository structure

```
forest-biomass-dinov2/
├── corrected_pipeline.py                      # leak-free engine (train-only K/ranking/clustering/HPs)
├── corrected_ablations.py                     # German ablations (all-features / no-clustering)
├── corrected_literature.py                    # GLCM/VI literature baselines, same protocol
├── robustness_german.py                       # German 20-split robustness sweep
├── make_corrected_figures.py                  # regenerate figures_corrected/
├── make_biomass_rasters.py                    # corrected-model biomass raster maps (Figure 4)
├── summarize_corrected.py                     # writes RESULTS_CORRECTED.md
├── run_all_corrected.sh                        # full leak-free run (logged, fault-tolerant)
├── datasets_corrected/                        # corrected input feature CSVs per site
├── results_corrected/                         # corrected result JSONs + logs
├── figures_corrected/                         # corrected figures (png/svg/pdf)
├── RESULTS_CORRECTED.md                       # human-readable corrected results table
├── PROJECT_NOTES.md                           # methodology, leak fixes, run commands, file map
├── pakistani_data_analysis/                   # ORIGINAL (leaky) Balakot pipeline — superseded
├── german_data_results/                       # ORIGINAL (leaky) Karlsruhe pipeline — superseded
├── paper_writeup/                             # Markdown write-ups + paper figures (PNG/SVG/PDF)
├── requirements.txt
└── .gitignore
```

> **Note on file layout:** the scripts use paths relative to their own location, so the
> directory structure above must be preserved for them to run without edits. The original
> `pakistani_data_analysis/` and `german_data_results/` trees are retained for provenance,
> but they produce the **superseded leaky** numbers; use the corrected pipeline below.

## Setup

```bash
pip install -r requirements.txt
```

Developed with a conda environment (`conda create -n biomass python=3.11` then the
packages in `requirements.txt`; needs `umap-learn` and `xgboost`).

## Reproducing the results (leak-free)

All choices are made on training data only. The committed `datasets_corrected/` and
`features_*.csv` are sufficient to reproduce every corrected metric without re-extracting
features.

```bash
conda activate biomass

# Full corrected pipeline (German + ablations + literature + robustness + figures + summary),
# logged to results_corrected/logs/ and fault-tolerant:
bash run_all_corrected.sh

# …or run individual pieces:
python corrected_pipeline.py --site pakistan --features dino   # nested-LOO, ensemble-K NNLS
python corrected_pipeline.py --site german   --features dino   # 85/15, cluster-separated
python corrected_ablations.py                                  # German ablations
python corrected_literature.py                                 # GLCM/VI literature baselines
python robustness_german.py --seeds 20 --profile light         # 20-split robustness
python make_corrected_figures.py                               # figures_corrected/
python summarize_corrected.py                                  # -> RESULTS_CORRECTED.md
```

The **original (leaky)** analysis can still be reproduced from the `pakistani_data_analysis/`
and `german_data_results/` scripts, but those numbers are superseded — see the leak-free
revision note above.

## Data availability

**The committed feature CSVs (`datasets_corrected/`, `features_*.csv`) and result JSONs
(`results_corrected/`) are sufficient to reproduce every corrected metric and every figure**
— the analysis and figure scripts run directly from what is in this repo, with no download
required.

The two large **aerial RGB source GeoTIFFs are not tracked in git** (each is above
GitHub's 100 MB limit). They are only needed to rebuild the German dataset from scratch
or to regenerate the aerial panels of Figure 4, and are hosted on Google Drive:

```bash
pip install gdown
python download_data.py        # fetches + extracts the rasters in place
python download_data.py --verify   # check sizes / sha256 of existing files
```

This downloads a single ~1.1 GB archive and extracts the rasters to the exact paths the
scripts expect (`High_resoltuion_Aerial_Photograph.tif` at the repo root and
`german data/karlsruhe.tif`). See [`data_manifest.json`](data_manifest.json) for the file
list and checksums. The predicted-biomass output rasters are *not* distributed
(regenerable outputs); the rendered biomass maps are already committed as figures in
`paper_writeup/` (e.g. `fig4_biomass_raster_maps.png`).
