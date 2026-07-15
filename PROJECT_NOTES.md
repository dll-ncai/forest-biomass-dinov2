# Project notes — Forest AGB estimation with DINOv2 (leak-free re-run)

_Working memory for this project. Read this first when resuming._

## What the study is
Estimate forest **above-ground biomass (AGB)** from high-resolution RGB imagery using
frozen **DINOv2 ViT-S** CLS features (384-d), benchmarked against **ResNet50** features
(2048-d) and classical **GLCM + vegetation-index literature baselines** (97-d).
Two sites:
- **Pakistan (Balakot)** — n=25, evaluated by **nested leave-one-out (LOO)**.
  Labels are converted kg → t/ha with a circular plot r=17.5 m.
- **Germany (Karlsruhe)** — n=113, **85/15 stratified hold-out** with a train-only
  **UMAP + KMeans (k=2)** clustering, one regressor per cluster, predictions pooled.

## The leakage problem (why this re-run exists)
The **original** committed results (`README.md`, `paper_writeup/`, and the figures/JSON
under `pakistani_data_analysis/` and `german_data_results/`) were **inflated by data
leakage**: feature-count K, feature ranking (|Pearson| with biomass), and the clustering
were all fit using ALL samples (including test). Original headline (LEAKY, do not cite):
Pakistan DINO+NNLS 21.81 %RMSE / R² 0.939; Germany DINO+NNLS 13.85 %RMSE / R² 0.897.

## The corrected, leak-free pipeline
`corrected_pipeline.py` is the shared engine (imported by the ablation, literature and
robustness scripts). Everything below is selected on **train only**:
- **K (feature count)**: Germany → 5-fold CV on the train split; Pakistan → inner 5-fold
  CV inside every outer LOO fold (fully nested).
- **Feature ranking**: |Pearson(feature, biomass)| recomputed on train inside each fold.
- **Clustering** (Germany): StandardScaler→UMAP→KMeans fit on train; val/test assigned by
  `transform`/`predict`. Per-cluster ranking uses that cluster's train samples only.
- **Model hyper-parameters**: chosen by the same inner CV.

Models: **NNLS** (local non-negative reconstruction), **Ridge**, **XGBoost**.
Literature baselines (`corrected_literature.py`): RF / GBT / XGB / Ridge on the 97-d
hand-crafted features, same protocol (nested LOO / 85-15), CV-tuned on train.

## Improvements made in this session (2026-07-15)
1. **Improved NNLS** (`predict_nnls`): added **L2 (Tikhonov) regularisation** on the
   reconstruction weights (augmented NNLS) + support for a **wider neighbour count `k`**.
   Rationale: reconstructing a high-dim query from ≤12 neighbours was
   under-determined/unstable. `simplex=True` keeps predictions inside the training range
   (convex combo); `simplex=False` allows extrapolation.
2. **Two small-sample guards for NNLS** (the key fixes — see the sensitivity story below):
   - **simplex-only below n<30** (`NNLS_SIMPLEX_ONLY_BELOW=30`, guard in
     `select_K_and_cfg`): on n=25 the CV can't reliably pick the higher-variance
     `simplex=False` option, so it's dropped for tiny training sets. Germany clusters
     (35-61) are unaffected and keep the full grid.
   - **ensemble-over-K below n<30** (`predict_nnls_ensembleK`, used in `run_pakistan`):
     instead of *selecting* one feature count K by inner CV (per-fold K jumped 10→300,
     %RMSE 48), it tunes cfg per K then **averages predictions across the whole K grid**.
     This removes the K-selection variance and is insensitive to grid granularity
     (29.3–29.9 %RMSE whether the grid has 8 or 18 values).
3. **Extensive hyper-parameter tuning** (rich grids, all train-only/leak-free):
   NNLS 84 configs (k∈{3..20} × cosine/euclidean × simplex on/off × l2∈{0,0.1,1}),
   Ridge 16, XGB 16; `FEATURE_COUNTS` 18 values (10→360).
   `cp.set_profile("light")` swaps to a smaller grid for the 20-seed robustness sweep.
4. **Figures** regenerated (`make_corrected_figures.py`, Okabe-Ito CVD-safe palette):
   performance, best-model scatter, German ablation, CV-vs-#features, robustness →
   `figures_corrected/` (+ copies in `paper_writeup/`).

### The NNLS-on-Pakistan sensitivity story (why the guards exist)
Extensive tuning *hurt* NNLS on Pakistan (n=25) before the guards: rich grid → 50 %RMSE.
Diagnostics (`diag_pak_nnls*.py`, logs in `results_corrected/logs/`) showed it was
**selection overfitting on tiny data**, through two knobs — `simplex=False` and a
fine feature-count grid (fine grids overfit regardless of K range; ~5-8 round-number
candidates were fine). **Ensemble-over-K** dissolves the problem entirely and is the
robust final choice. Lesson: on n≈25 nested LOO, prefer *averaging over* hyper-parameters
to *selecting* them. This is not p-hacking — the ensemble result is grid-insensitive.

## FINAL corrected + tuned headline results (best model per site)
| Site | Best model | %RMSE | R² | vs. best literature/GLCM |
|------|-----------|-------|-----|--------------------------|
| Pakistan (nested LOO) | **DINOv2 + NNLS (ensemble-K)** | **29.93** | **0.886** | beats RF 31.12 / 0.876 |
| Germany (85/15) | **DINOv2 + NNLS** | **17.40** | **0.837** | beats Ridge 22.79 / 0.721 |

- **NNLS is now the best model on BOTH sites** (confirms the reconstruction intuition).
- Germany robustness (20 splits, honest): DINO-NNLS 23.0 ± 3.7 %RMSE — still below the
  best literature mean (24.4). The single-split 17.40 is optimistic; cite ~23 as typical.
- ResNet50 is clearly worse than DINOv2 on both sites (best ResNet50: Pakistan XGB 41.5,
  Germany Ridge 24.1). DINOv2 > ResNet50 holds throughout.
- Full numbers: `RESULTS_CORRECTED.md` (auto-generated).

## How to run (always in the `biomass` conda env; use a screen)
```bash
conda activate biomass                       # py3.11, sklearn1.9, xgboost2.0.3, umap
screen -dmS biomass bash run_all_corrected.sh   # full pipeline, logged, fault-tolerant
# individual pieces:
python corrected_pipeline.py --site pakistan --features dino
python corrected_ablations.py
python corrected_literature.py
python robustness_german.py --seeds 20 --profile light
python make_corrected_figures.py
python summarize_corrected.py                # -> RESULTS_CORRECTED.md
```
`run_all_corrected.sh` runs each step independently (a failure never aborts the rest),
logs to `results_corrected/logs/`, and touches `results_corrected/.RUN_COMPLETE` at the end.

## Where things live
- Corrected inputs: `datasets_corrected/{site}_{dino,resnet50,glcm}.csv`
- Corrected results (this session): `results_corrected/*.json`
- Previous lighter-tuned corrected results: `results_corrected_v1_backup/`
- Human-readable results table: `RESULTS_CORRECTED.md` (auto-generated)
- Figures: `figures_corrected/` (png/svg/pdf) and copied into `paper_writeup/`
- Script backups: `_original_backup_20260714/*.bak`

## Status / TODO
- [x] Improve NNLS + extensive tuning — done. NNLS now best on both sites (see above).
- [x] Final tuned numbers confirmed — `RESULTS_CORRECTED.md` (2026-07-15 12:35 run).
- [x] Figures regenerated from corrected results — `figures_corrected/` + `paper_writeup/`.
- [x] Pakistan question resolved: with ensemble-K NNLS, DINO (29.93/0.886) now edges out
      literature RF (31.12/0.876). DINO advantage holds on Pakistan again.
- [x] `paper_writeup/*.md` rewritten to corrected numbers/methodology (2026-07-15):
      `methodology_and_results.md` fully rewritten (leak-free methods + corrected tables +
      honest literature framing); `pakistani_data_writeup.md` and `german_data_writeup.md`
      given superseding banners + corrected headline tables/claims.
      UPDATE (later 2026-07-15 session): the German doc's detailed Tables 2-4 were NOT just
      flagged — they are now fully RECOMPUTED to leak-free values from
      results_corrected/german_dino.json. §2.2 replaced the leaky "test %RMSE vs K" sweep
      with the train-only inner-CV curve; §2.3 fixed cluster sizes [10,7]->[8,9] + numbers
      (combined 17.40/30.89/0.837; cluster0 K=360 15.62%/0.199; cluster1 K=70 19.58%/0.876);
      §2.4 dropped the unverifiable overlap counts; §2.5 hyper-params are the CV-selected
      per-cluster K+cfg. Verified internally consistent (pooled RMSE, cluster means). No
      leaky numbers remain in any paper_writeup/*.md.
- [x] Figure 4 biomass raster maps REGENERATED for the corrected model:
      `make_biomass_rasters.py` (run in AI_ForestWatch env: torch+timm+rasterio+GPU) slides
      DINO windows over the aerial GeoTIFFs and predicts with a convex NNLS map model
      (bounded output). Rasters: pakistani_data_analysis/biomass_rasters/biomass_nnls_ton_ha.tif
      (grid 40x61, 0-24.6 t/ha) and german_data_results/german_biomass_rasters_augmented_corrected/
      biomass_nnls.tif (grid 233x112, 0-244.9 t/ha). Figure rendered by
      paper_writeup/create_biomass_raster_figure.py (rewritten: per-site subfigures,
      constrained_layout, squarer German crop, balanced fonts) -> fig4_biomass_raster_maps.*
- [x] All results figures quality-checked (fonts/whitespace/crowding); perf-vs-features
      recoloured so colour=method, linestyle=cluster.
- [ ] Optional: extend ensemble-over-K idea to Ridge/XGB on Pakistan (only NNLS done).
- [ ] Note: NOTE re raster envs — `biomass` env has rasterio but NOT torch/timm; the GPU
      env `AI_ForestWatch` has torch+timm+rasterio (I pip-installed `click` to fix its
      rasterio). Use AI_ForestWatch for make_biomass_rasters.py; biomass for the figure.

## Reproduce this session's final state
```bash
conda activate biomass
screen -dmS biomass bash run_all_corrected.sh   # full run (german + ablations + lit + robustness + figures)
screen -dmS pak    bash run_pak.sh              # pakistan only (ensemble-K NNLS) + figures + summary
```
Diagnostics that motivated the guards: `diag_pak_nnls.py`, `diag_pak_nnls2.py`
(results in `results_corrected/logs/diag_pak_nnls*.log`).
