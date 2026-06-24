#!/usr/bin/env python3
"""
Create a publication-quality study area overview figure (Fig. 1).

Two panels:
  (a) Balakot, Pakistan — aerial RGB with 25 sample plot locations from biomass mask
  (b) Karlsruhe, Germany — aerial RGB with 101 plot locations (cluster centers;
      biomass mask has 303 sample plots in clusters of 3; 101 circular plots of 35 m radius)
      + zero-biomass extraction regions highlighted from zero_biomass_area.shp

Pakistan: plot locations from biomass mask (connected-component centroids).
Germany: 303 plots from biomass_karlsruhe.shp grouped into 101 clusters (K-means);
         each cluster center is the center of a circular plot with 35 m radius.

This script is designed so that all scale bars and colorbars (AGB colormaps)
are drawn **outside** the imagery, on the surrounding white space.
"""

import os
import csv as csvmod
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 28,
})
from matplotlib.patches import Circle, Rectangle, Polygon as MplPolygon
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage as ndi

import rasterio
from rasterio.crs import CRS as RasterioCRS
from rasterio.warp import transform as warp_transform
import fiona
from shapely.geometry import shape


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAK_HIGHRES = os.path.join(ROOT, "High_resoltuion_Aerial_Photograph.tif")
PAK_MASK = os.path.join(ROOT, "total_biomass_mask.tif")
PAK_CSV = os.path.join(ROOT, "dino_features_with_labels_and_split.csv")

GER_HIGHRES = os.path.join(ROOT, "german data", "karlsruhe.tif")
GER_MASK = os.path.join(ROOT, "german data", "biomass mask.tif")
GER_PLOTS_SHP = os.path.join(
    ROOT, "german data", "biomass_karlsruhe.shp"
)  # 303 plots in 101 clusters of 3
GER_ZERO_SHP = os.path.join(ROOT, "german data", "zero_biomass_area.shp")
GER_CSV = os.path.join(
    ROOT,
    "german_data_results",
    "complete_pipeline_dataset",
    "dino_features_with_labels_and_split.csv",
)

OUTPUT_DIR = os.path.join(ROOT, "paper_writeup")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "fig1_study_area_overview.png")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "fig1_study_area_overview.pdf")
OUTPUT_SVG = os.path.join(OUTPUT_DIR, "fig1_study_area_overview.svg")


# ── Constants ────────────────────────────────────────────────────────────────
PAK_PLOT_RADIUS_M = 17.5
PAK_PIXEL_SIZE_M = 0.23
PAK_PLOT_AREA_M2 = math.pi * PAK_PLOT_RADIUS_M**2

GER_PLOT_RADIUS_M = 35.0
GER_PIXEL_SIZE_M = 0.177

NODATA = -9999.0


# ── Helpers ──────────────────────────────────────────────────────────────────


def to_rgb_uint8(bands: np.ndarray) -> np.ndarray:
    """(C,H,W) raster → (H,W,3) uint8 with percentile stretch."""
    if bands.ndim == 3:
        rgb = np.transpose(bands[:3], (1, 2, 0)).astype(np.float64)
    else:
        # single-band → stack to RGB
        arr = bands.astype(np.float64)
        rgb = np.stack([arr, arr, arr], axis=-1)

    valid = rgb[rgb > 0]
    if valid.size == 0:
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)

    lo, hi = np.percentile(valid, [2, 98])
    if hi <= lo:
        lo, hi = float(rgb.min()), float(rgb.max())

    rgb = np.clip((rgb - lo) / (hi - lo) * 255, 0, 255)
    return rgb.astype(np.uint8)


def geo_to_pixel(x_geo, y_geo, transform):
    """Convert geographic (x, y) → image pixel (col, row) using rasterio transform."""
    # For an Affine transform: x = a*col + b*row + c; y = d*col + e*row + f
    # Solve approximately via inverse transform when possible.
    try:
        col, row = ~transform * (x_geo, y_geo)
    except Exception:
        # Fallback for simple north-up affine
        col = (x_geo - transform.c) / transform.a
        row = (y_geo - transform.f) / transform.e
    return col, row


def pixel_to_geo(row, col, transform):
    """Convert mask pixel (row, col) → geographic (x, y)."""
    x, y = transform * (col, row)
    return x, y


def find_mask_centroids_in_image_pixels(
    mask_path, image_transform, nodata=-9999.0, use_components=False
):
    """
    Find plot centroids from a biomass mask raster and convert to the
    aerial image's pixel coordinates.

    When use_components=True (Pakistan) do connected-component analysis per value.
    When use_components=False (Germany-style) treat each unique value as one plot,
    centroid = mean of all pixels with that value.

    Returns list of (img_col, img_row, biomass_value).
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        mask_transform = src.transform

    vals = mask.astype(float)
    unique_vals = np.unique(vals)
    unique_vals = unique_vals[unique_vals != nodata]
    unique_vals = unique_vals[~np.isnan(unique_vals)]
    print(f"    Mask unique values (excl nodata): {len(unique_vals)}")

    centroids = []

    if use_components:
        # Full connected-component analysis (small mask, Pakistan)
        structure = np.ones((3, 3), dtype=int)
        for val in unique_vals:
            lbl_mask = mask == val
            if not np.any(lbl_mask):
                continue
            labeled_arr, n_comp = ndi.label(lbl_mask, structure=structure)
            for comp_idx in range(1, n_comp + 1):
                coords = ndi.center_of_mass(lbl_mask, labels=labeled_arr, index=comp_idx)
                if coords is None:
                    continue
                m_row, m_col = coords
                x_geo, y_geo = pixel_to_geo(m_row, m_col, mask_transform)
                img_col, img_row = geo_to_pixel(x_geo, y_geo, image_transform)
                centroids.append((float(img_col), float(img_row), float(val)))
    else:
        # Fast path: each unique value = one plot → centroid = mean of pixel positions
        valid_mask = vals != nodata
        valid_mask &= ~np.isnan(vals)
        rows, cols = np.where(valid_mask)
        pixel_vals = vals[rows, cols]

        for val in unique_vals:
            idx = pixel_vals == val
            if not np.any(idx):
                continue
            m_row = float(np.mean(rows[idx]))
            m_col = float(np.mean(cols[idx]))
            x_geo, y_geo = pixel_to_geo(m_row, m_col, mask_transform)
            img_col, img_row = geo_to_pixel(x_geo, y_geo, image_transform)
            centroids.append((float(img_col), float(img_row), float(val)))

    return centroids


def filter_centroids_to_csv(centroids, csv_labels, tol=5.0):
    """
    Filter mask centroids to match the labels present in the CSV dataset.
    Greedy nearest-neighbor matching in biomass space.
    """
    csv_sorted = sorted(csv_labels)
    cent_sorted = sorted(centroids, key=lambda c: c[2])
    used = [False] * len(cent_sorted)

    result = []
    for csv_lbl in csv_sorted:
        best_idx, best_diff = -1, float("inf")
        for i, (col, row, val) in enumerate(cent_sorted):
            if used[i]:
                continue
            diff = abs(val - csv_lbl)
            if diff < best_diff:
                best_idx, best_diff = i, diff
        if best_idx >= 0 and best_diff <= tol:
            used[best_idx] = True
            col, row, val = cent_sorted[best_idx]
            result.append((col, row, val))
    return result


def get_german_101_cluster_centroids(
    shapefile_path, image_transform, image_crs, n_clusters=101
):
    """
    Load 303 plot polygons from the German shapefile, group them into 101 clusters
    of 3 (K-means on plot centroids), and return 101 cluster centers in image pixel
    coordinates, with mean biomass per cluster.

    Returns list of dicts:
      [{\"col\": float, \"row\": float, \"biomass_tha\": float}, ...]
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            f"scikit-learn required for clustering 303→101 plots: {e}"
        )

    plots = []
    with fiona.open(shapefile_path, "r") as shp:
        shp_crs = shp.crs
        for feat in shp:
            geom = shape(feat["geometry"])
            cent = geom.centroid
            biom = feat["properties"].get("biom_t_ha_", None)
            try:
                biom_f = float(biom) if biom is not None else None
            except (TypeError, ValueError):
                biom_f = None
            plots.append({"centroid": (cent.x, cent.y), "biomass": biom_f})

    if not plots:
        return []

    centroids_xy = np.array([p["centroid"] for p in plots], dtype=float)
    biomasses = np.array(
        [p["biomass"] if p["biomass"] is not None else np.nan for p in plots],
        dtype=float,
    )

    # Reproject to image CRS if needed
    shp_crs_wkt = RasterioCRS.from_user_input(shp_crs).wkt if shp_crs else None
    img_crs_wkt = RasterioCRS.from_user_input(image_crs).wkt if image_crs else None
    crs_differ = bool(shp_crs_wkt and img_crs_wkt and shp_crs_wkt != img_crs_wkt)

    if crs_differ:
        xs = centroids_xy[:, 0].tolist()
        ys = centroids_xy[:, 1].tolist()
        xs_t, ys_t = warp_transform(shp_crs, image_crs, xs, ys)
        centroids_xy = np.column_stack([xs_t, ys_t])

    # Cluster centroids into 101 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(centroids_xy)

    result = []
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        cx = float(np.mean(centroids_xy[mask, 0]))
        cy = float(np.mean(centroids_xy[mask, 1]))
        b = float(np.nanmean(biomasses[mask])) if np.any(~np.isnan(biomasses[mask])) else 0.0

        img_col, img_row = geo_to_pixel(cx, cy, image_transform)
        result.append({"col": float(img_col), "row": float(img_row), "biomass_tha": b})

    return result


# ── Load Pakistan ────────────────────────────────────────────────────────────


def load_pakistan():
    print("Loading Pakistan aerial image...")
    with rasterio.open(PAK_HIGHRES) as src:
        img = src.read()
        img_transform = src.transform
    rgb = to_rgb_uint8(img)

    print("Finding Pakistan plot centroids from biomass mask...")
    all_centroids = find_mask_centroids_in_image_pixels(
        PAK_MASK, img_transform, NODATA, use_components=True
    )
    print(f"  Raw centroids from mask: {len(all_centroids)}")

    # Filter to match the 25 CSV samples
    csv_labels = []
    with open(PAK_CSV, "r") as f:
        for row in csvmod.DictReader(f):
            csv_labels.append(float(row["label"]))
    centroids = filter_centroids_to_csv(all_centroids, csv_labels, tol=5.0)
    print(f"  After CSV matching: {len(centroids)}")

    plots = []
    for col, row, raw_val in centroids:
        if raw_val == 0.0:
            tha = 0.0
        else:
            # Convert from per-plot value back to t/ha (matching original script)
            tha = (raw_val * 10000.0 / PAK_PLOT_AREA_M2) / 1000.0
        plots.append({"col": col, "row": row, "biomass_tha": tha})

    return rgb, plots


# ── Load Germany ─────────────────────────────────────────────────────────────


def load_germany():
    print("Loading Germany aerial image (downsampled)...")
    with rasterio.open(GER_HIGHRES) as src:
        full_h, full_w = src.height, src.width
        img_transform = src.transform
        img_crs = src.crs
        # Downsample ~6x for display
        scale = max(1, full_h // 4000)
        out_h, out_w = full_h // scale, full_w // scale
        img = src.read([1, 2, 3], out_shape=(3, out_h, out_w))
    rgb = to_rgb_uint8(img)
    print(f"  Display: {rgb.shape[0]}x{rgb.shape[1]}  (scale={scale})")

    # 303 plots in shapefile are in clusters of 3; 101 plots = 101 cluster centers (35 m radius each)
    print("Loading 303 plots from shapefile and grouping into 101 cluster centers...")
    plots_geo = get_german_101_cluster_centroids(
        GER_PLOTS_SHP, img_transform, img_crs, n_clusters=101
    )
    print(f"  Got {len(plots_geo)} cluster centers (101 expected)")

    # Scale centroids to display resolution
    plots = []
    for p in plots_geo:
        plots.append(
            {
                "col": p["col"] / scale,
                "row": p["row"] / scale,
                "biomass_tha": p["biomass_tha"],
            }
        )

    # Load zero-biomass extraction areas, project to image/display pixels
    print("Loading zero-biomass extraction regions...")
    zero_polygons = []
    with fiona.open(GER_ZERO_SHP, "r") as shp:
        zero_crs = shp.crs
        zero_crs_wkt = RasterioCRS.from_user_input(zero_crs).wkt if zero_crs else None
        img_crs_wkt = RasterioCRS.from_user_input(img_crs).wkt if img_crs else None
        crs_diff = bool(zero_crs_wkt and img_crs_wkt and zero_crs_wkt != img_crs_wkt)

        for feat in shp:
            geom = shape(feat["geometry"])
            if geom.is_empty:
                continue

            def _reproject_coords(coords):
                xs, ys = zip(*coords)
                if crs_diff:
                    xs_t, ys_t = warp_transform(zero_crs, img_crs, list(xs), list(ys))
                else:
                    xs_t, ys_t = xs, ys
                pts = []
                for x, y in zip(xs_t, ys_t):
                    pc, pr = geo_to_pixel(x, y, img_transform)
                    pts.append((pc / scale, pr / scale))
                return pts

            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                zero_polygons.append(_reproject_coords(coords))
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    zero_polygons.append(_reproject_coords(coords))

    print(f"  Loaded {len(zero_polygons)} zero-biomass regions")

    return rgb, plots, zero_polygons, scale


# ── Small drawing helpers ────────────────────────────────────────────────────


def _north(ax, rel_x, rel_y):
    """Draw a north arrow using axes-relative coordinates."""
    ax.annotate(
        "N",
        xy=(rel_x, rel_y + 0.04),
        xytext=(rel_x, rel_y),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=18,
        fontweight="bold",
        color="white",
        ha="center",
        va="bottom",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8),
    )


def _draw_scalebar_axis(ax, bar_length_m, pixel_size_m, image_width_px, scale=1):
    """
    Draw a clean scale bar in a dedicated thin axis below the image.
    The bar is centered and labeled with the distance.
    """
    # Convert bar to display pixels
    bar_px = bar_length_m / pixel_size_m / scale

    ax.set_xlim(0, image_width_px)
    ax.set_ylim(0, 1)
    ax.set_aspect("auto")
    ax.axis("off")

    # Center the bar
    x_center = image_width_px / 2.0
    x0 = x_center - bar_px / 2.0
    y0 = 0.3

    bar_h = 0.35
    rect = Rectangle((x0, y0), bar_px, bar_h, fc="black", ec="black", lw=0.8)
    ax.add_patch(rect)

    # End ticks
    tick_h = 0.15
    for x in (x0, x0 + bar_px):
        ax.plot([x, x], [y0 - tick_h, y0 + bar_h + tick_h], color="black", lw=0.8)

    # Label
    ax.text(
        x_center,
        y0 + bar_h + 0.20,
        f"{bar_length_m} m",
        ha="center",
        va="bottom",
        fontsize=26,
        fontweight="bold",
        color="black",
    )


# ── Draw figure ──────────────────────────────────────────────────────────────


def create_figure():
    pak_rgb, pak_plots = load_pakistan()
    ger_rgb, ger_plots, ger_zero_polys, ger_scale = load_germany()

    cmap = plt.cm.YlOrRd

    # Figure layout:
    #   Row 0: images (Pakistan, Germany)
    #   Row 1: scale bars
    #   Row 2: horizontal colorbars (in white space)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[1.0, 0.05, 0.14],
        hspace=0.06,
        wspace=0.10,
    )

    ax_pak = fig.add_subplot(gs[0, 0])
    ax_ger = fig.add_subplot(gs[0, 1])
    ax_sb_pak = fig.add_subplot(gs[1, 0])
    ax_sb_ger = fig.add_subplot(gs[1, 1])
    ax_cb_pak = fig.add_subplot(gs[2, 0])
    ax_cb_ger = fig.add_subplot(gs[2, 1])

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (a): Pakistan
    # ══════════════════════════════════════════════════════════════════════
    ax_pak.imshow(pak_rgb, aspect="equal")
    ax_pak.set_xlim(0, pak_rgb.shape[1])
    ax_pak.set_ylim(pak_rgb.shape[0], 0)

    pak_circle_r_px = PAK_PLOT_RADIUS_M / PAK_PIXEL_SIZE_M

    pak_forested = [p for p in pak_plots if p["biomass_tha"] > 0]
    pak_nonforest = [p for p in pak_plots if p["biomass_tha"] == 0]

    pak_bm_max = max((p["biomass_tha"] for p in pak_forested), default=1.0)
    pak_norm = Normalize(vmin=0.0, vmax=pak_bm_max)

    for p in pak_forested:
        c = Circle(
            (p["col"], p["row"]),
            pak_circle_r_px,
            lw=1.2,
            ec="white",
            fc=cmap(pak_norm(p["biomass_tha"])),
            alpha=0.85,
        )
        ax_pak.add_patch(c)

    for p in pak_nonforest:
        c = Circle(
            (p["col"], p["row"]),
            pak_circle_r_px,
            lw=1.5,
            ec="cyan",
            fc="none",
            ls="--",
            alpha=0.95,
        )
        ax_pak.add_patch(c)

    # North arrow (upper-right)
    _north(ax_pak, 0.93, 0.88)

    # Subplot label (a) — top-left, outside plot area
    ax_pak.text(-0.01, 1.03, "(a)",
                transform=ax_pak.transAxes,
                fontsize=40, fontweight="bold",
                va="bottom", ha="right",
                color="black")
    ax_pak.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False
    )
    for sp in ax_pak.spines.values():
        sp.set_edgecolor("#333")
        sp.set_linewidth(1.0)

    # Legend (forested vs non-forested)
    leg_pak = ax_pak.legend(
        handles=[
            Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor=cmap(0.6), markeredgecolor="white",
                markersize=10, label="Forested plot",
            ),
            Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor="none", markeredgecolor="cyan",
                markersize=10, ls="--", label="Non-forested plot",
            ),
        ],
        loc="upper left",
        fontsize=24,
        framealpha=0.75,
        edgecolor="gray",
        fancybox=True,
        borderpad=0.6,
        handletextpad=0.4,
    )
    leg_pak.get_frame().set_facecolor("black")
    for t in leg_pak.get_texts():
        t.set_color("white")

    # Info box
    ax_pak.text(
        0.02, 0.02,
        f"n = {len(pak_plots)} plots  |  r = {PAK_PLOT_RADIUS_M} m  |  GSD = {PAK_PIXEL_SIZE_M} m",
        transform=ax_pak.transAxes,
        fontsize=22,
        color="white",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.65,
                  ec="gray", lw=0.7),
    )

    # Scale bar axis below Pakistan (white space)
    _draw_scalebar_axis(
        ax_sb_pak,
        bar_length_m=200,
        pixel_size_m=PAK_PIXEL_SIZE_M,
        image_width_px=pak_rgb.shape[1],
        scale=1,
    )

    # ══════════════════════════════════════════════════════════════════════
    # PANEL (b): Germany
    # ══════════════════════════════════════════════════════════════════════
    ax_ger.imshow(ger_rgb, aspect="equal")
    ax_ger.set_xlim(0, ger_rgb.shape[1])
    ax_ger.set_ylim(ger_rgb.shape[0], 0)

    ger_circle_r_px = GER_PLOT_RADIUS_M / GER_PIXEL_SIZE_M / ger_scale

    ger_bm_max = max((p["biomass_tha"] for p in ger_plots), default=1.0)
    ger_norm = Normalize(vmin=0.0, vmax=ger_bm_max)

    # Draw zero-biomass extraction regions first (below plots)
    for poly_coords in ger_zero_polys:
        patch = MplPolygon(
            poly_coords,
            closed=True,
            lw=1.2,
            ec="cyan",
            fc="cyan",
            alpha=0.25,
        )
        ax_ger.add_patch(patch)
        patch_border = MplPolygon(
            poly_coords,
            closed=True,
            lw=1.0,
            ec="cyan",
            fc="none",
            ls="-",
            alpha=0.8,
        )
        ax_ger.add_patch(patch_border)

    # Draw inventory plots
    for p in ger_plots:
        c = Circle(
            (p["col"], p["row"]),
            ger_circle_r_px,
            lw=0.8,
            ec="white",
            fc=cmap(ger_norm(p["biomass_tha"])),
            alpha=0.85,
        )
        ax_ger.add_patch(c)

    # North arrow (upper-right)
    _north(ax_ger, 0.95, 0.90)

    # Subplot label (b) — top-left, outside plot area
    ax_ger.text(-0.01, 1.03, "(b)",
                transform=ax_ger.transAxes,
                fontsize=40, fontweight="bold",
                va="bottom", ha="right",
                color="black")
    ax_ger.tick_params(
        left=False, bottom=False, labelleft=False, labelbottom=False
    )
    for sp in ax_ger.spines.values():
        sp.set_edgecolor("#333")
        sp.set_linewidth(1.0)

    # Legend
    leg_ger = ax_ger.legend(
        handles=[
            Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor=cmap(0.6), markeredgecolor="white",
                markersize=10, label="Forest inventory plot",
            ),
            MplPolygon(
                [(0, 0)], closed=True,
                fc="cyan", ec="cyan", alpha=0.4,
                label="Zero-biomass extraction area",
            ),
        ],
        loc="upper left",
        fontsize=24,
        framealpha=0.75,
        edgecolor="gray",
        fancybox=True,
        borderpad=0.6,
        handletextpad=0.4,
    )
    leg_ger.get_frame().set_facecolor("black")
    for t in leg_ger.get_texts():
        t.set_color("white")

    # Info box
    ax_ger.text(
        0.02, 0.02,
        f"n = {len(ger_plots)} plots  |  r = {GER_PLOT_RADIUS_M:.0f} m  |  GSD = {GER_PIXEL_SIZE_M} m",
        transform=ax_ger.transAxes,
        fontsize=22,
        color="white",
        ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", fc="black", alpha=0.65,
                  ec="gray", lw=0.7),
    )

    # Scale bar axis below Germany (white space)
    _draw_scalebar_axis(
        ax_sb_ger,
        bar_length_m=500,
        pixel_size_m=GER_PIXEL_SIZE_M,
        image_width_px=ger_rgb.shape[1],
        scale=ger_scale,
    )

    # ══════════════════════════════════════════════════════════════════════
    # COLORBARS (outside images, on white space)
    # ══════════════════════════════════════════════════════════════════════
    sm_pak = plt.cm.ScalarMappable(cmap=cmap, norm=pak_norm)
    sm_pak.set_array([])
    cb_pak = plt.colorbar(sm_pak, cax=ax_cb_pak, orientation="horizontal")
    cb_pak.set_label("AGB (t/ha)", fontsize=28, labelpad=6)
    cb_pak.ax.tick_params(labelsize=24)

    sm_ger = plt.cm.ScalarMappable(cmap=cmap, norm=ger_norm)
    sm_ger.set_array([])
    cb_ger = plt.colorbar(sm_ger, cax=ax_cb_ger, orientation="horizontal")
    cb_ger.set_label("AGB (t/ha)", fontsize=28, labelpad=6)
    cb_ger.ax.tick_params(labelsize=24)

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_kwargs = dict(bbox_inches="tight", pad_inches=0.15, facecolor="white")
    fig.savefig(OUTPUT_SVG, **save_kwargs)
    print(f"\nSaved SVG: {OUTPUT_SVG}")
    fig.savefig(OUTPUT_PNG, dpi=300, **save_kwargs)
    print(f"Saved PNG: {OUTPUT_PNG}")
    fig.savefig(OUTPUT_PDF, **save_kwargs)
    print(f"Saved PDF: {OUTPUT_PDF}")
    plt.close(fig)
    print("Done!")


# ── Entry point ──────────────────────────────────────────────────────────────


if __name__ == "__main__":
    create_figure()

