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
    "font.size": 20,
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
    ax.text(0.03, 0.955, f"({letter})",
            transform=ax.transAxes,
            fontsize=26, fontweight="bold", color="white",
            va="top", ha="left", zorder=20,
            path_effects=[
                pe.Stroke(linewidth=3.0, foreground="black"),
                pe.Normal(),
            ])


def square_crop(bio, rgb, target_aspect=1.15):
    """Crop a tall raster pair to a squarer, representative top region so the
    panels sit compactly next to the (wide) Pakistan panels without whitespace."""
    h, w = bio.shape
    want_h = int(round(w * target_aspect))
    if want_h < h:
        # find the contiguous top band with the most valid data
        bio, rgb = bio[:want_h], rgb[:want_h]
    return bio, rgb


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
    # Germany is a tall strip -> crop to a squarer, representative top region so it
    # sits compactly beside the wide Pakistan panels (avoids large side whitespace).
    ger_bio, ger_rgb = square_crop(ger_bio, ger_rgb, target_aspect=1.15)

    LAB = "Predicted AGB (t/ha)"
    CB_KW = dict(fraction=0.9, pad=0.02)

    def add_row(subfig, rgb, bio, vmax, letters, title):
        axes = subfig.subplots(1, 2)
        axes[0].imshow(rgb, interpolation="bilinear", aspect="equal")
        panel_label(axes[0], letters[0]); axes[0].axis("off")
        im = axes[1].imshow(np.ma.masked_invalid(bio), cmap=BIOMASS_CMAP,
                            vmin=0, vmax=vmax, interpolation="nearest", aspect="equal")
        panel_label(axes[1], letters[1]); axes[1].axis("off")
        cb = subfig.colorbar(im, ax=axes[1], **CB_KW)
        cb.set_label(LAB, fontsize=22, labelpad=6)
        cb.ax.tick_params(labelsize=18)
        subfig.suptitle(title, fontsize=23, y=0.99)

    # Portrait figure; per-site subfigures keep each colorbar next to its own map.
    pak_aspect = pak_bio.shape[0] / pak_bio.shape[1]
    ger_aspect = ger_bio.shape[0] / ger_bio.shape[1]
    fig = plt.figure(figsize=(11, 12), constrained_layout=True)
    subfigs = fig.subfigures(2, 1, height_ratios=[pak_aspect, ger_aspect * 1.02])
    add_row(subfigs[0], pak_rgb, pak_bio, pak_vmax, ("a", "b"),
            "Pakistan (Balakot) — aerial RGB and predicted AGB")
    add_row(subfigs[1], ger_rgb, ger_bio, ger_vmax, ("c", "d"),
            "Germany (Karlsruhe) — aerial RGB and predicted AGB")

    for ext in ("svg", "png", "pdf"):
        path = os.path.join(out_dir, f"fig4_biomass_raster_maps.{ext}")
        fig.savefig(path, dpi=200, facecolor="white", edgecolor="none")
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
