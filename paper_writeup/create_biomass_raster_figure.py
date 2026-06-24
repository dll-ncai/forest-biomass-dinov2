#!/usr/bin/env python3
"""
Figure 4: Biomass Raster Maps for Both Study Areas.

2 rows × 2 columns:
  (a) Pakistan aerial RGB          (b) Pakistan NNLS biomass
  (c) Germany aerial RGB           (d) Germany NNLS biomass

Subplot labels (a)–(d) are placed inside the top-left corner of each panel
(white text with black stroke) because axes are image panels with no frame.
Colorbar labels use Times New Roman at 16pt.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import Normalize
import rasterio

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 28,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAK_RGB = os.path.join(ROOT, "High_resoltuion_Aerial_Photograph.tif")
PAK_BIO = os.path.join(ROOT, "pakistani_data_analysis",
                        "biomass_rasters", "biomass_nnls_ton_ha.tif")
GER_RGB = os.path.join(ROOT, "german data", "karlsruhe.tif")
GER_BIO = os.path.join(ROOT, "german_data_results",
                        "german_biomass_rasters_augmented_corrected",
                        "biomass_nnls.tif")

BIOMASS_CMAP = "viridis"


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_rgb(path, bands=(1, 2, 3), max_dim=4000):
    """Read RGB bands, downsampling so the longest axis ≤ max_dim."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = max(1, max(h, w) // max_dim)
        out_h, out_w = h // scale, w // scale
        img = np.stack(
            [src.read(b, out_shape=(out_h, out_w)) for b in bands],
            axis=-1,
        ).astype(np.float64)
    return img


def contrast_stretch(img, lo_pct=1, hi_pct=99):
    out = np.zeros_like(img, dtype=np.float64)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo = np.nanpercentile(ch, lo_pct)
        hi = np.nanpercentile(ch, hi_pct)
        if hi - lo < 1e-6:
            hi = lo + 1
        out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def read_biomass(path, max_dim=4000):
    """Read single-band biomass raster at display resolution."""
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        scale = max(1, max(h, w) // max_dim)
        out_h, out_w = h // scale, w // scale
        data = src.read(1, out_shape=(out_h, out_w)).astype(np.float64)
        nd = src.nodata
    if nd is not None:
        data[data == nd] = np.nan
    data = np.where(np.isnan(data), np.nan, np.clip(data, 0, None))
    return data


def valid_bbox(arr):
    valid = ~np.isnan(arr)
    rows  = np.any(valid, axis=1)
    cols  = np.any(valid, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return r0, r1 + 1, c0, c1 + 1


def panel_label(ax, letter):
    """Bold white letter with black stroke, top-left inside the image."""
    ax.text(0.025, 0.965, f"({letter})",
            transform=ax.transAxes,
            fontsize=44, fontweight="bold", color="white",
            va="top", ha="left", zorder=20,
            path_effects=[
                pe.Stroke(linewidth=3.5, foreground="black"),
                pe.Normal(),
            ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Reading Pakistan RGB …")
    pak_rgb = contrast_stretch(read_rgb(PAK_RGB))
    print("Reading Pakistan biomass …")
    pak_bio = read_biomass(PAK_BIO)
    print("Reading Germany RGB …")
    ger_rgb = contrast_stretch(read_rgb(GER_RGB))
    print("Reading Germany biomass …")
    ger_bio = read_biomass(GER_BIO)

    pak_vmax = np.nanmax(pak_bio)
    ger_vmax = np.nanmax(ger_bio)
    print(f"Pakistan biomass 0 – {pak_vmax:.1f} t/ha")
    print(f"Germany  biomass 0 – {ger_vmax:.1f} t/ha")

    # Crop to valid extent of biomass raster
    pr0, pr1, pc0, pc1 = valid_bbox(pak_bio)
    pak_bio = pak_bio[pr0:pr1, pc0:pc1]
    pak_rgb = pak_rgb[pr0:pr1, pc0:pc1]

    gr0, gr1, gc0, gc1 = valid_bbox(ger_bio)
    ger_bio = ger_bio[gr0:gr1, gc0:gc1]
    ger_rgb = ger_rgb[gr0:gr1, gc0:gc1]

    ph, pw = pak_bio.shape
    gh, gw = ger_bio.shape
    pak_aspect  = ph / pw
    ger_aspect  = gh / gw
    height_ratio = (ger_aspect / pak_aspect) * 0.5

    fig_w       = 16
    col_w       = fig_w * 0.44
    pak_row_h   = col_w * pak_aspect
    ger_row_h   = col_w * ger_aspect * 0.5
    fig_h       = pak_row_h + ger_row_h + 2.5

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(2, 3,
                            height_ratios=[1, height_ratio],
                            width_ratios=[1, 1, 0.05],
                            wspace=0.12, hspace=0.06)

    # ── (a) Pakistan RGB ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(pak_rgb, interpolation="bilinear", aspect="equal")
    panel_label(ax, "a")
    ax.axis("off")

    # ── (b) Pakistan NNLS biomass ──────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    im_pak = ax.imshow(np.ma.masked_invalid(pak_bio),
                       cmap=BIOMASS_CMAP, vmin=0, vmax=pak_vmax,
                       interpolation="nearest", aspect="equal")
    panel_label(ax, "b")
    ax.axis("off")
    cax_pak = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im_pak, cax=cax_pak)
    cb.set_label("Predicted AGB (t/ha)", fontsize=32, labelpad=8)
    cb.ax.tick_params(labelsize=28)

    # ── (c) Germany RGB ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(ger_rgb, interpolation="bilinear", aspect="equal")
    panel_label(ax, "c")
    ax.axis("off")

    # ── (d) Germany NNLS biomass ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    im_ger = ax.imshow(np.ma.masked_invalid(ger_bio),
                       cmap=BIOMASS_CMAP, vmin=0, vmax=ger_vmax,
                       interpolation="nearest", aspect="equal")
    panel_label(ax, "d")
    ax.axis("off")
    cax_ger = fig.add_subplot(gs[1, 2])
    cb2 = fig.colorbar(im_ger, cax=cax_ger)
    cb2.set_label("Predicted AGB (t/ha)", fontsize=32, labelpad=8)
    cb2.ax.tick_params(labelsize=28)

    # ── Save ──────────────────────────────────────────────────────────────────
    for ext in ("svg", "png", "pdf"):
        path = os.path.join(out_dir, f"fig4_biomass_raster_maps.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.15)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
