# Forest Above-Ground Biomass Estimation from RGB with Self-Supervised DINO Features

Estimating forest above-ground biomass (AGB) from high-resolution aerial RGB imagery using
**frozen self-supervised DINO ViT-Small** CLS-token features (384-d), benchmarked against
**ResNet50** features (2048-d) and against classical hand-crafted **GLCM + vegetation-index**
literature baselines.

Two independent study sites:

| Site | Location | n plots | Evaluation protocol |
|------|----------|---------|---------------------|
| Balakot | Pakistan | 25 | Nested leave-one-out |
| Karlsruhe | Germany | 113 (96 train / 17 test) | 85/15 stratified hold-out + 20-split robustness sweep |

Every modelling choice — feature count *K*, feature ranking, clustering, and all model
hyper-parameters — is made on **training data only** (inner CV inside each fold), so the
reported metrics estimate performance on unseen plots.

## Headline results

### Balakot, Pakistan — nested LOO, n = 25

| Features | Model | %RMSE | R² | RMSE (t/ha) |
|----------|-------|-------|-----|------------|
| **DINO (384-d)** | **NNLS** | **24.71** | **0.922** | **3.20** |
| DINO (384-d) | Ridge | 31.73 | 0.871 | 4.11 |
| GLCM literature | Random Forest | 31.12 | 0.876 | 4.03 |
| GLCM literature | Gradient Boosting | 31.91 | 0.870 | 4.13 |
| ResNet50 (2048-d) | XGBoost | 41.47 | 0.780 | 5.37 |

### Karlsruhe, Germany — 85/15 hold-out, n_test = 17

| Features | Model | %RMSE | R² | RMSE (t/ha) |
|----------|-------|-------|-----|------------|
| **DINO (384-d)** | **NNLS** | **17.40** | **0.837** | **30.89** |
| DINO (384-d) | Ridge | 23.83 | 0.695 | 42.32 |
| GLCM literature | Ridge | 22.79 | 0.721 | 40.46 |
| GLCM literature | Random Forest | 23.70 | 0.698 | 42.09 |
| ResNet50 (2048-d) | Ridge | 24.13 | 0.687 | 42.85 |

A single 85/15 split is optimistic on a site this size. Over **20 stratified splits** the
German numbers settle at DINO + NNLS **22.99 ± 3.65 %RMSE** vs. the best literature baseline
(GLCM + Ridge) **24.43 ± 2.46** — a real but modest margin. Quote the 20-split figure as the
typical result, not the single split.

NNLS is the strongest regressor on **both** sites, and DINO features beat ResNet50 features
throughout. The complete table of every metric is in [`RESULTS.md`](RESULTS.md), regenerated
by `summarize_results.py`.

---

## Quick start

```bash
git clone https://github.com/dll-ncai/forest-biomass-dinov2.git
cd forest-biomass-dinov2

conda create -n biomass python=3.11 -y
conda activate biomass
pip install -r requirements.txt

bash run_all.sh          # ~1–2 h; writes results/, figures/, RESULTS.md
```

Everything needed to reproduce every number is committed in this repository — the feature
CSVs, the image crops, and the plot labels. **No download is required** for the numeric
results or the results figures. The large aerial GeoTIFFs are only needed for the biomass
raster maps (Step 7 below).

---

## Step-by-step reproduction

Each step is independent and can be run on its own. Times are for a 16-core machine.

### Step 0 — Environment

```bash
conda create -n biomass python=3.11 -y
conda activate biomass
pip install -r requirements.txt
```

`umap-learn` and `xgboost` are both required (the German pipeline clusters with UMAP +
KMeans). Verify:

```bash
python -c "import numpy, pandas, sklearn, umap, xgboost, skimage; print('env OK')"
```

To keep the four parallel jobs in `run_all.sh` from oversubscribing your CPU:

```bash
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
```

### Step 1 — Main pipeline: both sites × both backbones (~30–50 min)

```bash
python pipeline.py --site pakistan --features dino
python pipeline.py --site pakistan --features resnet50
python pipeline.py --site german   --features dino
python pipeline.py --site german   --features resnet50
```

Reads `datasets/<site>_<backbone>.csv`, writes `results/<site>_<backbone>.json`.

Pakistan runs a fully nested LOO: inside each of the 25 outer folds, *K* and the model
hyper-parameters are chosen by inner CV on the other 24 samples. Germany fits
StandardScaler → UMAP → KMeans on the training split only, then assigns test samples with
`transform`/`predict`, and selects *K* per cluster by 5-fold CV on that cluster's training
samples.

**Verify:** `results/pakistan_dino.json` → NNLS `percent_rmse` ≈ 24.71, `r2` ≈ 0.922.
`results/german_dino.json` → NNLS `percent_rmse` ≈ 17.40, `r2` ≈ 0.837.

### Step 2 — German ablations (~10 min)

```bash
python ablations.py
```

Runs two ablations against the proposed configuration, writing
`results/ablations_german_{dino,resnet50}.json`:

* `all-features` — train-only clustering, but every feature per cluster (no ranking/selection)
* `no-clustering` — one global model, train-only ranking and *K* selection

**Verify:** DINO all-features NNLS ≈ 17.90 %RMSE; DINO no-clustering NNLS ≈ 20.45 %RMSE.

### Step 3 — Literature baselines (~15 min)

```bash
python literature_baselines.py
```

Extracts 97 hand-crafted features (GLCM texture, vegetation indices, spectral statistics)
from the **same image crops** used for the deep features, so the comparison is on identical
patches, then evaluates Ridge / RF / GBT / XGBoost under the identical protocol per site.
Writes `results/literature_baselines.json`.

**Verify:** Pakistan RF ≈ 31.12 %RMSE; Germany Ridge ≈ 22.79 %RMSE.

### Step 4 — German robustness sweep (~20–30 min)

```bash
python robustness_german.py --seeds 20 --profile light
```

Repeats the whole German evaluation over 20 stratified 85/15 splits so the conclusion does
not rest on one split. `--profile light` uses a reduced hyper-parameter grid to keep the
sweep tractable; `--profile rich` uses the full grid and takes considerably longer.
Writes `results/robustness_german.json`.

**Verify:** `summary["dino|proposed|nnls"]` → `prmse_mean` ≈ 22.99, `prmse_std` ≈ 3.65.

### Step 5 — Figures (~10 s)

```bash
python make_figures.py
```

Reads only the JSONs in `results/` and writes PNG (300 dpi) + SVG + PDF into `figures/`:

| File | Content |
|------|---------|
| `fig_performance.*` | %RMSE and R² per site, DINO vs. ResNet50 vs. literature |
| `fig_scatter.*` | predicted vs. observed AGB, best model per site |
| `fig_ablation_german.*` | German ablation comparison |
| `fig_perf_vs_features.*` | German train-only CV curve vs. feature count *K* |
| `fig_robustness_german.*` | distribution over the 20 splits |

Any figure whose source JSON is missing is skipped with a warning rather than failing.

### Step 6 — Results table (~1 s)

```bash
python summarize_results.py
```

Collates every JSON in `results/` into [`RESULTS.md`](RESULTS.md). This is the file to diff
against to confirm a clean reproduction.

### Step 7 — Biomass raster maps (optional, needs the aerial GeoTIFFs)

The two source aerial rasters exceed GitHub's 100 MB file limit and are hosted separately:

```bash
pip install gdown rasterio
python download_data.py            # fetches + extracts ~1.1 GB in place
python download_data.py --verify   # check sizes and sha256 against data_manifest.json
```

They land at the exact paths the scripts expect —
`High_resoltuion_Aerial_Photograph.tif` (Balakot) at the repo root and
`german data/karlsruhe.tif` (Karlsruhe). Then:

```bash
python make_biomass_rasters.py --site both
```

This step also needs `rasterio`, which is not in `requirements.txt` because none of
Steps 1–6 use it.

### Run everything at once

```bash
bash run_all.sh
```

Runs Steps 1–6 in the right order, parallelising where it can. Every step is logged to
`results/logs/` and a failure in one step does **not** abort the rest, so a long detached
run always finishes:

```bash
screen -dmS biomass bash run_all.sh
```

`results/.RUN_COMPLETE` is touched when the script finishes. Note that `run_all.sh` activates
the `biomass` conda environment from `~/anaconda3`; edit the `source`/`conda activate` lines
at the top if your install lives elsewhere.

---

## Re-extracting features from scratch (optional)

Steps 1–6 read the committed feature CSVs, so this is not needed to reproduce the results.
To regenerate them from the image crops:

```bash
pip install torch torchvision timm
python extract_features.py --model dino     --site pakistan
python extract_features.py --model resnet50 --site pakistan
python extract_features.py --model dino     --site german
python extract_features.py --model resnet50 --site german
```

Backbones are `timm` models, frozen, classification head removed:

* `dino` → `vit_small_patch16_224.dino`, 384-d CLS token (self-supervised ViT-S/16)
* `resnet50` → `resnet50`, 2048-d global average pool (supervised CNN)

Pakistan crops are 152×152, center-cropped to 112 and resized to 224 (bicubic); German crops
are already 224×224. Both are normalised with ImageNet mean/std.

Rebuilding the German dataset (clustering the 303 field plots into 101 crops) from the raw
Karlsruhe raster additionally needs the GeoTIFF from Step 7 plus `fiona` and `shapely`:

```bash
python german_data_results/build_complete_pipeline.py
```

---

## Repository layout

```
forest-biomass-dino/
├── pipeline.py                  # main engine: train-only K / ranking / clustering / HP selection
├── ablations.py                 # German ablations (all-features, no-clustering)
├── literature_baselines.py      # GLCM + vegetation-index baselines, identical protocol
├── robustness_german.py         # 20-split German robustness sweep
├── make_figures.py              # regenerate figures/ from results/
├── summarize_results.py         # regenerate RESULTS.md from results/
├── make_biomass_rasters.py      # wall-to-wall biomass raster maps (needs GeoTIFFs)
├── extract_features.py          # DINO / ResNet50 feature extraction from crops
├── download_data.py             # fetch the large aerial GeoTIFFs
├── run_all.sh                   # Steps 1–6, logged and fault-tolerant
├── datasets/                    # model-ready feature CSVs per site and backbone
├── results/                     # result JSONs (one per experiment)
├── figures/                     # generated figures (PNG / SVG / PDF)
├── dataset_crops_152/           # 25 Balakot image crops (labels in the filenames)
├── german_data_results/         # Karlsruhe crops, dataset builder, earlier analysis
├── pakistani_data_analysis/     # earlier Balakot analysis and figures
├── RESULTS.md                   # full auto-generated results table
├── data_manifest.json           # sizes + sha256 for the downloadable GeoTIFFs
└── requirements.txt
```

Scripts resolve paths relative to their own location, so keep the directory structure
intact for them to run without edits.

`german_data_results/` and `pakistani_data_analysis/` hold the earlier, exploratory analyses
and their figures. They are kept for provenance; the numbers reported above and in
`RESULTS.md` come from the top-level scripts described in Steps 1–6.

## Method summary

1. **Crops.** One image crop per field plot (Balakot) or per plot cluster (Karlsruhe).
2. **Features.** A frozen backbone maps each crop to a fixed vector — DINO ViT-S/16 CLS
   token (384-d) or ResNet50 pooled features (2048-d). Nothing is fine-tuned.
3. **Clustering (Germany only).** StandardScaler → UMAP → KMeans, fit on the training split
   only; test samples are assigned by `transform`/`predict`. A separate model is fitted per
   cluster.
4. **Feature selection.** Features are ranked by |Pearson r| with biomass, computed on the
   training samples of the current fold only. The count *K* is chosen by inner CV.
5. **Regression.** Three model families: NNLS (local non-negative reconstruction of each test
   sample from its *k* nearest training neighbours, with L2-regularised weights, then the same
   weights applied to their biomass), Ridge, and XGBoost.
6. **Metrics.** %RMSE (RMSE as a percentage of mean observed biomass), R², RMSE and MAE in
   t/ha.

## Citation

If you use this code or the results, please cite the accompanying paper. Contact the
repository owners for the current citation.
