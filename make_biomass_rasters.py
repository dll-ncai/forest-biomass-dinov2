#!/usr/bin/env python3
"""
Generate biomass raster maps for both sites (Figure 4 inputs).

Slides NON-OVERLAPPING crop-sized windows over each aerial GeoTIFF (stride == window),
extracts frozen DINO features (same backbone/preprocessing as training), and predicts
biomass with the best NNLS model reported for THAT site, fitted on ALL of its plots.

The two sites take different branches of pipeline, so they must be mapped with
different predictors:
  * Pakistan (n=25) -> run_pakistan's small-sample branch = `predict_nnls_bestresid`,
    config k=6 / l2=10 / simplex, with per-query reconstruction-fidelity K selection.
    Convex (simplex) weights bound every mapped value to the observed biomass range.
  * Germany (n=113) -> run_german = UMAP+KMeans clustering with a PLAIN `predict_nnls`
    per cluster at that cluster's CV-selected K and cfg (k=3, simplex=False). Germany
    never uses `predict_nnls_bestresid`; mapping it with the Pakistani predictor flattens
    the map to near the training mean (see GERMAN_CLUSTER_CFG).
Non-negative weights keep both maps at or near the observed range, which is the right
behaviour for an application map.

Run in the AI_ForestWatch env (torch+timm+rasterio+GPU).  Outputs:
  pakistani_data_analysis/biomass_rasters/biomass_nnls_ton_ha.tif
  german_data_results/german_biomass_rasters_augmented_corrected/biomass_nnls.tif
"""
import os, math, argparse
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
import torch, timm
import torchvision.transforms as T
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pakistan's best-model NNLS config: the exact `predict_nnls_bestresid` setting reported in
# the paper (accuracy-optimal dense-grid setting; simplex => bounded to observed range).
NNLS_CFG_PAKISTAN = dict(k=6, l2=10.0, simplex=True)

# Germany's best-model NNLS: per-cluster K + cfg exactly as selected by the train-only inner
# CV and reported in results/german_dino.json. Clusters are keyed by mean plot AGB
# rather than by KMeans label, since the label ordering is arbitrary across refits.
#
# Using Pakistan's k=6 / l2=10 / simplex here instead (as this script previously did) halves
# the German map's dynamic range -- the heavy Tikhonov term drives the 6 convex weights
# toward uniform so every forest window collapses to ~the training mean (prediction IQR
# 26 t/ha vs 45 for the real model), and the residual criterion picks the smallest subspace
# (K=10) on 93% of windows because 6 unit vectors trivially span a 10-dim space.
GERMAN_CLUSTER_CFG = {
    "high": dict(K=360, k=3, cosine=True, simplex=False, l2=0.0),   # json cluster 0, mean ~197 t/ha
    "low":  dict(K=70,  k=3, cosine=True, simplex=False, l2=0.1),   # json cluster 1, mean ~159 t/ha
}
# Train-only feature-count sweep the per-query reconstruction-fidelity selector searches over
# (mirrors FEATURE_COUNTS in pipeline.py; values > n_features are skipped at use).
FEATURE_COUNTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160,
                  190, 220, 260, 300, 360]

SITES = {
    "pakistan": dict(
        raster=os.path.join(ROOT, "High_resoltuion_Aerial_Photograph.tif"),
        feats=os.path.join(ROOT, "datasets", "pakistan_dino.csv"),
        out=os.path.join(ROOT, "pakistani_data_analysis", "biomass_rasters", "biomass_nnls_ton_ha.tif"),
        window=152, stride=152, nodata=65536, kg_to_tha=True,
        # 152 window -> center 112 -> 224  (matches training transform)
        tf=T.Compose([T.CenterCrop(112), T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        stretch=True),
    "german": dict(
        raster=os.path.join(ROOT, "german data", "karlsruhe.tif"),
        feats=os.path.join(ROOT, "datasets", "german_dino.csv"),
        out=os.path.join(ROOT, "german_data_results", "german_biomass_rasters_augmented_corrected", "biomass_nnls.tif"),
        window=224, stride=224, nodata=255, kg_to_tha=False,
        tf=T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        stretch=False),
}


def build_dino():
    m = timm.create_model("vit_small_patch16_224.dino", pretrained=True, num_classes=0)
    return m.eval().to(DEVICE)


def render_rgb(raster, stretch, nodata):
    """Read bands 1-3 as an (H,W,3) uint8 image + a validity mask (True=data)."""
    with rasterio.open(raster) as src:
        arr = src.read([1, 2, 3]).astype(np.float32)           # (3,H,W)
        transform, crs = src.transform, src.crs
        raw = src.read(1)
    valid = raw != nodata
    if stretch:                                                # per-band 1-99% linear stretch
        rgb = np.zeros(arr.shape, np.uint8)
        for b in range(3):
            band = arr[b]; v = band[valid]
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            rgb[b] = np.clip((band - lo) * (255.0 / max(hi - lo, 1e-6)), 0, 255).astype(np.uint8)
    else:
        rgb = np.clip(arr, 0, 255).astype(np.uint8)
    return np.transpose(rgb, (1, 2, 0)), valid, transform, crs   # (H,W,3), (H,W)


@torch.no_grad()
def window_features(img, valid, cfg, batch=256):
    win, stride = cfg["window"], cfg["stride"]
    H, W = img.shape[:2]
    rows = list(range(0, H - win + 1, stride))
    cols = list(range(0, W - win + 1, stride))
    out = np.full((len(rows), len(cols)), np.nan, np.float32)   # biomass grid (filled later)
    coords, tiles = [], []
    feats = np.full((len(rows), len(cols), 384), np.nan, np.float32)
    model = build_dino()

    def flush():
        if not tiles:
            return
        x = torch.stack(tiles).to(DEVICE)
        f = model(x).cpu().numpy()
        for (ri, ci), fv in zip(coords, f):
            feats[ri, ci] = fv
        coords.clear(); tiles.clear()

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            vwin = valid[r:r + win, c:c + win]
            if vwin.mean() < 0.5:            # mostly background -> skip
                continue
            tile = img[r:r + win, c:c + win]
            tiles.append(cfg["tf"](Image.fromarray(tile)))
            coords.append((ri, ci))
            if len(tiles) >= batch:
                flush()
    flush()
    return feats, rows, cols


def rank_features_train(X, y, n):
    """Top-n feature indices by |Pearson(feature, y)| (identical to pipeline)."""
    Xc = X - X.mean(0, keepdims=True); yc = y - y.mean()
    r = (Xc.T @ yc) / np.sqrt((Xc ** 2).sum(0) * float(yc @ yc) + 1e-30)
    r[~np.isfinite(r)] = 0.0
    return np.argsort(-np.abs(r))[:n].tolist()


def fit_predictor(feats_csv, kg_to_tha):
    """Load all plots for the site as (X_all_features, y). No pre-ranking here -- the
    per-query reconstruction-fidelity selector re-ranks inside each candidate subspace."""
    df = pd.read_csv(feats_csv)
    fc = [c for c in df.columns if c.startswith("feature_")]
    X = df[fc].values.astype(np.float32); y = df["label"].values.astype(np.float32)
    if kg_to_tha:
        y = (y / 1000.0) * (10000.0 / (math.pi * 17.5 ** 2))
    return X, y


def predict_nnls_bestresid(Xtr, ytr, Xte, feature_counts, k, l2, simplex):
    """Best-model predictor, ported verbatim from pipeline.predict_nnls_bestresid.

    For each query the kNN NNLS reconstruction is solved in every candidate top-K feature
    subspace; the prediction is taken from the subspace with the smallest (L2-augmented)
    reconstruction residual. Subspace selection is unsupervised (feature geometry only)."""
    counts = [c for c in feature_counts if c <= Xtr.shape[1]]
    orders = {K: rank_features_train(Xtr, ytr, K) for K in counts}
    subspaces = {}
    for K in counts:
        idx = orders[K]
        Xs = Xtr[:, idx]
        Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-10)
        subspaces[K] = (idx, Xn)
    out = np.empty(len(Xte), np.float32)
    for qi, x in enumerate(Xte):
        best_p, best_r = 0.0, np.inf
        for K in counts:
            idx, Xn = subspaces[K]
            q = x[idx] / (np.linalg.norm(x[idx]) + 1e-10)
            d = 1 - Xn @ q
            kk = min(k, len(Xn))
            nn = np.argsort(d)[:kk]
            A = Xn[nn].T
            b = q
            if l2 > 0:
                A = np.vstack([A, np.sqrt(l2) * np.eye(kk)])
                b = np.concatenate([q, np.zeros(kk)])
            try:
                w, _ = nnls(A, b, maxiter=50000)
            except Exception:
                w = np.ones(kk) / kk
            r = float(np.linalg.norm(A @ w - b))
            if r < best_r:
                ws = w / w.sum() if (simplex and w.sum() > 0) else w
                best_r, best_p = r, float(ws @ ytr[nn])
        out[qi] = best_p
    return out


def predict_nnls_plain(Xtr, ytr, Xte, K, k, cosine=True, simplex=False, l2=0.0):
    """Plain local NNLS reconstruction, ported verbatim from pipeline.predict_nnls.

    This is the predictor the German site actually uses: the query is reconstructed from its
    k nearest training plots inside the top-K correlation-ranked subspace, and the same
    non-negative weights carry the neighbours' biomass. With ``simplex=False`` the raw NNLS
    weights pass through, so the reconstruction magnitude modulates the prediction and the
    map keeps its dynamic range."""
    idx = rank_features_train(Xtr, ytr, K)
    Xs = Xtr[:, idx]
    if cosine:
        Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-10)
    else:
        sc = StandardScaler().fit(Xs)
        Xn = sc.transform(Xs)
    out = np.empty(len(Xte), np.float32)
    for qi, x in enumerate(Xte):
        if cosine:
            q = x[idx] / (np.linalg.norm(x[idx]) + 1e-10)
            d = 1 - Xn @ q
        else:
            q = sc.transform(x[None, idx])[0]
            d = np.linalg.norm(Xn - q, axis=1)
        kk = min(k, len(Xn))
        nn = np.argsort(d)[:kk]
        A = Xn[nn].T
        b = q
        if l2 > 0:
            A = np.vstack([A, np.sqrt(l2) * np.eye(kk)])
            b = np.concatenate([q, np.zeros(kk)])
        try:
            w, _ = nnls(A, b, maxiter=50000)
        except Exception:
            w = np.ones(kk) / kk
        if simplex and w.sum() > 0:
            w = w / w.sum()
        out[qi] = float(w @ ytr[nn])
    return out


def predict_german_clustered(Xtr, ytr, Xte):
    """German best model: UMAP+KMeans clustering with a plain per-cluster NNLS at that
    cluster's CV-selected K and cfg.

    The clustering is refitted on all plots (a wall-to-wall map is not a held-out evaluation,
    so every plot is training data here) and each window is assigned through the same
    UMAP->KMeans transform. Clusters are matched to their reported configs by mean plot AGB,
    because KMeans label ids are arbitrary."""
    sc = StandardScaler().fit(Xtr)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2,
                        random_state=42, metric="euclidean").fit(sc.transform(Xtr))
    km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(reducer.embedding_)
    plot_lab = km.labels_
    win_lab = km.predict(reducer.transform(sc.transform(Xte)))

    means = {c: float(ytr[plot_lab == c].mean()) for c in (0, 1)}
    role = {max(means, key=means.get): "high", min(means, key=means.get): "low"}

    out = np.empty(len(Xte), np.float32)
    for c in (0, 1):
        cfg = GERMAN_CLUSTER_CFG[role[c]]
        sel = win_lab == c
        print(f"    cluster {c} ({role[c]}, mean {means[c]:.1f} t/ha): "
              f"{int((plot_lab == c).sum())} plots, {int(sel.sum())} windows, "
              f"K={cfg['K']} k={cfg['k']} simplex={cfg['simplex']} l2={cfg['l2']}", flush=True)
        if sel.any():
            out[sel] = predict_nnls_plain(Xtr[plot_lab == c], ytr[plot_lab == c], Xte[sel], **cfg)
    return out


def predict_windows(feats, Xtr, ytr, site):
    """Apply that site's best-model predictor to every valid window in the (R,C,384) grid."""
    R, C = feats.shape[:2]
    flat = feats.reshape(-1, feats.shape[2])
    valid = np.isfinite(flat[:, 0])
    grid = np.full(flat.shape[0], np.nan, np.float32)
    if valid.any():
        if site == "pakistan":
            grid[valid] = predict_nnls_bestresid(Xtr, ytr, flat[valid], FEATURE_COUNTS,
                                                 **NNLS_CFG_PAKISTAN)
        else:
            grid[valid] = predict_german_clustered(Xtr, ytr, flat[valid])
    return grid.reshape(R, C)


def write_raster(out_path, grid, rows, cols, cfg, transform, crs):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    win, stride = cfg["window"], cfg["stride"]
    # each output pixel = one window; georeference to window centres
    out_tf = transform * Affine.translation(win / 2.0, win / 2.0) * Affine.scale(stride, stride)
    with rasterio.open(out_path, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1],
                       count=1, dtype="float32", crs=crs, transform=out_tf, nodata=np.nan) as dst:
        dst.write(grid.astype(np.float32), 1)
    print(f"  wrote {out_path}  grid={grid.shape}  "
          f"biomass range [{np.nanmin(grid):.1f}, {np.nanmax(grid):.1f}] "
          f"({np.isfinite(grid).sum()} valid cells)")


def run(site):
    cfg = SITES[site]
    print(f"[{site}] rendering RGB from {os.path.basename(cfg['raster'])} ...", flush=True)
    img, valid, transform, crs = render_rgb(cfg["raster"], cfg["stretch"], cfg["nodata"])
    print(f"[{site}] image {img.shape}  valid {valid.mean()*100:.0f}%  extracting DINO features ...", flush=True)
    feats, rows, cols = window_features(img, valid, cfg)
    print(f"[{site}] grid {len(rows)}x{len(cols)}  fitting best-model NNLS on plots ...", flush=True)
    Xtr, ytr = fit_predictor(cfg["feats"], cfg["kg_to_tha"])
    grid = predict_windows(feats, Xtr, ytr, site)
    write_raster(cfg["out"], grid, rows, cols, cfg, transform, crs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["pakistan", "german", "both"], default="both")
    a = ap.parse_args()
    for s in (["pakistan", "german"] if a.site == "both" else [a.site]):
        run(s)
    print("RASTERS_DONE")


if __name__ == "__main__":
    main()
