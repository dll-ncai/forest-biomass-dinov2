#!/usr/bin/env python3
"""
Generate corrected-model biomass raster maps for both sites (Figure 4 inputs).

Slides crop-sized windows over each aerial GeoTIFF, extracts frozen DINOv2 features
(same backbone/preprocessing as training), and predicts biomass with the CORRECTED
NNLS reconstruction model trained on ALL plots of that site. Because the predictor uses
convex (simplex) weights, every mapped value is a weighted average of training-plot
biomass -> bounded to the observed range, which is the right behaviour for an
application map.

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
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGENET_MEAN = [0.485, 0.456, 0.406]; IMAGENET_STD = [0.229, 0.224, 0.225]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Corrected NNLS map predictor config (representative of the tuned model; simplex => bounded)
MAP_CFG = dict(k=10, l2=1.0, K=100)

SITES = {
    "pakistan": dict(
        raster=os.path.join(ROOT, "High_resoltuion_Aerial_Photograph.tif"),
        feats=os.path.join(ROOT, "datasets_corrected", "pakistan_dino.csv"),
        out=os.path.join(ROOT, "pakistani_data_analysis", "biomass_rasters", "biomass_nnls_ton_ha.tif"),
        window=152, stride=76, nodata=65536, kg_to_tha=True,
        # 152 window -> center 112 -> 224  (matches training transform)
        tf=T.Compose([T.CenterCrop(112), T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]),
        stretch=True),
    "german": dict(
        raster=os.path.join(ROOT, "german data", "karlsruhe.tif"),
        feats=os.path.join(ROOT, "datasets_corrected", "german_dino.csv"),
        out=os.path.join(ROOT, "german_data_results", "german_biomass_rasters_augmented_corrected", "biomass_nnls.tif"),
        window=224, stride=112, nodata=255, kg_to_tha=False,
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


def fit_predictor(feats_csv, kg_to_tha):
    df = pd.read_csv(feats_csv)
    fc = [c for c in df.columns if c.startswith("feature_")]
    X = df[fc].values.astype(np.float32); y = df["label"].values.astype(np.float32)
    if kg_to_tha:
        y = (y / 1000.0) * (10000.0 / (math.pi * 17.5 ** 2))
    # rank top-K features by |Pearson| on all plots
    Xc = X - X.mean(0, keepdims=True); yc = y - y.mean()
    r = (Xc.T @ yc) / np.sqrt((Xc ** 2).sum(0) * (yc @ yc) + 1e-30)
    r[~np.isfinite(r)] = 0
    idx = np.argsort(-np.abs(r))[:MAP_CFG["K"]]
    Xs = X[:, idx]
    Xn = Xs / (np.linalg.norm(Xs, axis=1, keepdims=True) + 1e-10)  # cosine space
    return idx, Xn, y


def predict_windows(feats, idx, Xn, ytr):
    k, l2 = MAP_CFG["k"], MAP_CFG["l2"]
    R, C = feats.shape[:2]
    out = np.full((R, C), np.nan, np.float32)
    for ri in range(R):
        for ci in range(C):
            fv = feats[ri, ci]
            if not np.isfinite(fv[0]):
                continue
            q = fv[idx]; q = q / (np.linalg.norm(q) + 1e-10)
            d = 1 - Xn @ q
            nn = np.argsort(d)[:k]
            A = Xn[nn].T
            A = np.vstack([A, np.sqrt(l2) * np.eye(len(nn))])
            b = np.concatenate([q, np.zeros(len(nn))])
            try:
                w, _ = nnls(A, b, maxiter=50000)
            except Exception:
                w = np.ones(len(nn))
            if w.sum() > 0:
                w = w / w.sum()                 # simplex -> bounded prediction
            out[ri, ci] = float(w @ ytr[nn])
    return out


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
    print(f"[{site}] grid {len(rows)}x{len(cols)}  fitting NNLS on plots ...", flush=True)
    idx, Xn, ytr = fit_predictor(cfg["feats"], cfg["kg_to_tha"])
    grid = predict_windows(feats, idx, Xn, ytr)
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
