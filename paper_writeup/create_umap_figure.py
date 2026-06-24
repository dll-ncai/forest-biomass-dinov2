#!/usr/bin/env python3
"""
Create Figure 3: UMAP embeddings of DINO feature vectors (2D + 3D).

Top row  (2D):  (a) German – clusters, (b) German – biomass, (c) Pakistani – biomass
Bottom row (3D): (d) German – clusters, (e) German – biomass, (f) Pakistani – biomass

Layout: 2×6 gridspec with dedicated colorbar columns so every data panel
        has identical width.  Columns: [d_a, cb_a*, d_b, cb_b, d_c, cb_c]
        (* cb_a and cb_d are not created — the reserved space equalises widths)

UMAP parameters: n_neighbors=15, min_dist=0.1, metric=euclidean, random_state=42
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

import umap

# ── Publication style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 28,
    "axes.labelsize": 32,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

CLUSTER_COLORS = ["#2E86C1", "#E74C3C"]
BIOMASS_CMAP   = "YlOrRd"
ELEV, AZIM     = 25, 225


def load_features(csv_path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feat_cols].values.astype(np.float64)
    y = df["label"].values.astype(np.float64)
    print(f"  Loaded {len(X)} samples, {len(feat_cols)} features, "
          f"biomass {y.min():.1f}–{y.max():.1f} t/ha")
    return X, y


def run_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, seed=42):
    X_std = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        n_components=n_components, metric="euclidean",
        random_state=seed,
    )
    return reducer.fit_transform(X_std)


def subplot_label_2d(ax, letter):
    ax.text(-0.08, 1.05, f"({letter})",
            transform=ax.transAxes,
            fontsize=40, fontweight="bold",
            va="bottom", ha="left")


def subplot_label_3d(ax, letter):
    ax.text2D(-0.06, 1.04, f"({letter})",
              transform=ax.transAxes,
              fontsize=40, fontweight="bold",
              va="bottom", ha="left")


def _add_cb(fig, cax, mappable, label="AGB (t/ha)"):
    """Add colorbar into a pre-allocated cax — no space stolen from data axes."""
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=30, labelpad=6)
    cb.ax.tick_params(labelsize=26)


# ══════════════════════════════════════════════════════════════════════════════
def main():
    base = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(base)

    # ── data ──────────────────────────────────────────────────────────────
    print("German dataset:")
    ger_csv = os.path.join(
        root, "german_data_results", "complete_pipeline_dataset",
        "dino_features_with_labels_and_split_augmented.csv")
    X_ger, y_ger = load_features(ger_csv)

    print("Pakistani dataset:")
    pak_csv = os.path.join(root, "dino_features_with_labels_and_split.csv")
    X_pak, y_pak = load_features(pak_csv)
    y_pak = y_pak / 1000.0
    print(f"  Converted to t/ha: {y_pak.min():.2f}–{y_pak.max():.2f}")

    # ── UMAP ─────────────────────────────────────────────────────────────
    print("\nUMAP 2D (German)...")
    emb2_ger = run_umap(X_ger, n_components=2)
    print("UMAP 2D (Pakistani)...")
    emb2_pak = run_umap(X_pak, n_components=2)
    print("UMAP 3D (German)...")
    emb3_ger = run_umap(X_ger, n_components=3)
    print("UMAP 3D (Pakistani)...")
    emb3_pak = run_umap(X_pak, n_components=3)

    # ── K-Means clusters ──────────────────────────────────────────────────
    km   = KMeans(n_clusters=2, n_init=10, random_state=42)
    cl_ger = km.fit_predict(emb2_ger)
    km3  = KMeans(n_clusters=2, n_init=10, random_state=42)
    cl_ger_3d = km3.fit_predict(emb3_ger)
    print(f"Clusters 2D: {np.bincount(cl_ger)},  3D: {np.bincount(cl_ger_3d)}")

    # ── Figure layout: 2 rows × 6 cols ────────────────────────────────────
    # Columns: [data_a, cb*_a, data_b, cb_b, data_c, cb_c]
    # cb*_a and cb*_d are not created — reserved space keeps widths equal.
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(
        2, 6,
        width_ratios=[10, 0.9, 10, 0.9, 10, 0.9],
        height_ratios=[1, 1.25],
        hspace=0.40, wspace=0.10,
        left=0.07, right=0.97, top=0.95, bottom=0.05,
    )

    # 2D subplot axes
    ax_a  = fig.add_subplot(gs[0, 0])
    # gs[0, 1]: colorbar column for (a) — left empty (reserved space only)
    ax_b  = fig.add_subplot(gs[0, 2])
    cax_b = fig.add_subplot(gs[0, 3])
    ax_c  = fig.add_subplot(gs[0, 4])
    cax_c = fig.add_subplot(gs[0, 5])

    # 3D subplot axes
    ax_d  = fig.add_subplot(gs[1, 0], projection="3d")
    # gs[1, 1]: colorbar column for (d) — left empty (reserved space only)
    ax_e  = fig.add_subplot(gs[1, 2], projection="3d")
    cax_e = fig.add_subplot(gs[1, 3])
    ax_f  = fig.add_subplot(gs[1, 4], projection="3d")
    cax_f = fig.add_subplot(gs[1, 5])

    mk2 = dict(s=80, edgecolors="white", linewidths=0.5, zorder=3)
    mk3 = dict(s=60, edgecolors="white", linewidths=0.3, depthshade=True)

    norm_g = Normalize(vmin=y_ger.min(), vmax=y_ger.max())
    norm_p = Normalize(vmin=y_pak.min(), vmax=y_pak.max())

    # ─────────── (a) German 2D – clusters ─────────────────────────────────
    for ci in range(2):
        m = cl_ger == ci
        ax_a.scatter(emb2_ger[m, 0], emb2_ger[m, 1],
                     c=CLUSTER_COLORS[ci], label=f"Cluster {ci + 1}", **mk2)
    ax_a.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC",
                loc="best", markerscale=1.3)
    _fmt2d(ax_a)
    subplot_label_2d(ax_a, "a")

    # ─────────── (b) German 2D – biomass ──────────────────────────────────
    sc = ax_b.scatter(emb2_ger[:, 0], emb2_ger[:, 1],
                      c=y_ger, cmap=BIOMASS_CMAP, norm=norm_g, **mk2)
    _add_cb(fig, cax_b, sc)
    _fmt2d(ax_b)
    subplot_label_2d(ax_b, "b")

    # ─────────── (c) Pakistani 2D – biomass ───────────────────────────────
    sc2 = ax_c.scatter(emb2_pak[:, 0], emb2_pak[:, 1],
                       c=y_pak, cmap=BIOMASS_CMAP, norm=norm_p, **mk2)
    _add_cb(fig, cax_c, sc2)
    _fmt2d(ax_c)
    subplot_label_2d(ax_c, "c")

    # ─────────── (d) German 3D – clusters ─────────────────────────────────
    for ci in range(2):
        m = cl_ger_3d == ci
        ax_d.scatter(emb3_ger[m, 0], emb3_ger[m, 1], emb3_ger[m, 2],
                     c=CLUSTER_COLORS[ci], label=f"Cluster {ci + 1}", **mk3)
    ax_d.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC",
                loc="upper left", markerscale=1.3)
    _fmt3d(ax_d)
    subplot_label_3d(ax_d, "d")

    # ─────────── (e) German 3D – biomass ──────────────────────────────────
    sc3 = ax_e.scatter(emb3_ger[:, 0], emb3_ger[:, 1], emb3_ger[:, 2],
                       c=y_ger, cmap=BIOMASS_CMAP, norm=norm_g, **mk3)
    _add_cb(fig, cax_e, sc3)
    _fmt3d(ax_e)
    subplot_label_3d(ax_e, "e")

    # ─────────── (f) Pakistani 3D – biomass ───────────────────────────────
    sc4 = ax_f.scatter(emb3_pak[:, 0], emb3_pak[:, 1], emb3_pak[:, 2],
                       c=y_pak, cmap=BIOMASS_CMAP, norm=norm_p, **mk3)
    _add_cb(fig, cax_f, sc4)
    _fmt3d(ax_f)
    subplot_label_3d(ax_f, "f")

    # ── save ──────────────────────────────────────────────────────────────
    for ext in ("svg", "png", "pdf"):
        out = os.path.join(base, f"fig3_umap_embeddings.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.15)
        print(f"Saved: {out}")
    plt.close(fig)


# ── formatting helpers ─────────────────────────────────────────────────────────

def _fmt2d(ax):
    ax.set_xlabel("UMAP 1", fontsize=32, labelpad=6)
    ax.set_ylabel("UMAP 2", fontsize=32, labelpad=6)
    ax.tick_params(labelsize=26)
    ax.grid(True, lw=0.4, alpha=0.4)


def _fmt3d(ax):
    ax.set_xlabel("UMAP 1", fontsize=30, labelpad=10)
    ax.set_ylabel("UMAP 2", fontsize=30, labelpad=10)
    ax.set_zlabel("UMAP 3", fontsize=30, labelpad=10)
    ax.tick_params(labelsize=24, pad=4)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#DDDDDD")
    ax.yaxis.pane.set_edgecolor("#DDDDDD")
    ax.zaxis.pane.set_edgecolor("#DDDDDD")
    ax.grid(True, lw=0.3, alpha=0.3)


if __name__ == "__main__":
    main()
