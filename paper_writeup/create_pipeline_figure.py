#!/usr/bin/env python3
"""
Create a publication-quality methodological pipeline figure (Fig. 2).

Horizontal left-to-right flow with 5 main stages connected by arrows.
After stage 4 (clustering), a decision diamond checks whether the data
shows multiple clusters. Both datasets go through UMAP + K-Means, but
only multi-cluster data gets per-cluster modelling.

Designed for full-page width in a two-column journal.
Grayscale-safe, clean lines, no gradients.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
import numpy as np
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 16,
    "text.color": "black",
})

# Palette
STAGE_FILL   = "#D0E4F5"
STAGE_BORDER = "#1B4F72"
ARROW_COLOR  = "#444444"
FROZEN_COLOR = "#D35400"
OUTPUT_FILL  = "#E8F8E8"
OUTPUT_BORDER = "#27AE60"
INPUT_FILL   = "#F5F5F5"
INPUT_BORDER = "#777777"
DECISION_FILL = "#FFF3CD"
YES_COLOR    = "#27AE60"
NO_COLOR     = "#2E86C1"


# ── drawing primitives ────────────────────────────────────────────────────

def stage_box(ax, x, y, w, h, title, subtitle=None, number=None,
              section_ref=None, facecolor=STAGE_FILL,
              edgecolor=STAGE_BORDER, fontsize=19):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=1.8, zorder=2, clip_on=False))
    lbl = f"{number}. {title}" if number else title
    ty = y + h - 0.15 if subtitle else y + h - 0.22
    ax.text(x + w/2, ty, lbl, ha="center", va="top",
            fontsize=fontsize, fontweight="bold", zorder=10, clip_on=False)
    if subtitle:
        ax.text(x + w/2, ty - 0.30, subtitle, ha="center", va="top",
                fontsize=14, color="#444", style="italic",
                zorder=10, clip_on=False)
    if section_ref:
        ax.text(x + w/2, y - 0.14, section_ref, ha="center", va="top",
                fontsize=13, color="#666", zorder=10, clip_on=False)


def small_box(ax, x, y, w, h, facecolor="#FFF", edgecolor="#AAA", lw=0.8):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, zorder=4, clip_on=False))


def arrow_h(ax, x1, y1, x2, y2, color=ARROW_COLOR, lw=1.5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=14, zorder=1, clip_on=False))


def arrow_curved(ax, x1, y1, x2, y2, rad=0.3, color=ARROW_COLOR, lw=1.5):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", color=color,
        linewidth=lw, mutation_scale=14,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1, clip_on=False))


def draw_image_icon(ax, x, y, s=0.4):
    cg = np.array([
        [[.30,.50,.20],[.35,.55,.25],[.40,.45,.20],[.30,.50,.30]],
        [[.35,.50,.25],[.50,.55,.30],[.45,.50,.25],[.35,.45,.20]],
        [[.40,.45,.20],[.35,.50,.25],[.30,.55,.30],[.40,.50,.25]],
        [[.30,.50,.30],[.40,.45,.20],[.35,.50,.25],[.35,.55,.20]],
    ])
    ax.imshow(cg, extent=[x, x+s, y, y+s], aspect="auto",
              zorder=4, interpolation="nearest", clip_on=False)
    ax.add_patch(Rectangle((x, y), s, s, lw=0.8, edgecolor="#555",
                            facecolor="none", zorder=5, clip_on=False))


def draw_plot_icon(ax, x, y, s=0.4):
    ax.add_patch(Rectangle((x, y), s, s, lw=0.8, edgecolor="#555",
                            facecolor="#EDEDED", zorder=4, clip_on=False))
    np.random.seed(7)
    for _ in range(7):
        ax.plot(x + np.random.uniform(.05, s-.05),
                y + np.random.uniform(.05, s-.05),
                "o", color="#C0392B", ms=2.5, zorder=6, clip_on=False)


def draw_lock(ax, x, y, s=0.16):
    bw, bh = s*0.65, s*0.45
    bx = x - bw/2
    ax.add_patch(Rectangle((bx, y), bw, bh,
                            facecolor=FROZEN_COLOR, edgecolor="#7B3F00",
                            lw=0.7, zorder=7, clip_on=False))
    th = np.linspace(0, np.pi, 30)
    r = bw * 0.38
    ax.plot(x + r*np.cos(th), y + bh + np.abs(r*np.sin(th))*0.7,
            color="#7B3F00", lw=1.2, zorder=7, clip_on=False)


def draw_transformer(ax, x, y, w=0.55, h=0.8):
    n = 5
    bh = h / (n*1.35)
    bw = w * 0.75
    bx = x + (w-bw)/2
    gap = (h - n*bh) / (n+1)
    for i in range(n):
        by = y + gap + i*(bh+gap)
        shade = 0.62 + i*0.04
        ax.add_patch(FancyBboxPatch(
            (bx, by), bw, bh, boxstyle="round,pad=0.01",
            facecolor=(shade, shade, shade+0.08),
            edgecolor="#555", lw=0.5, zorder=4, clip_on=False))
        if i < n-1:
            ax.plot([bx+bw/2]*2, [by+bh, by+bh+gap*0.6],
                    color="#888", lw=0.4, zorder=3, clip_on=False)
            ax.plot(bx+bw/2, by+bh+gap*0.6, "^",
                    color="#888", ms=2, zorder=3, clip_on=False)


def draw_feature_vector(ax, x, y, w=0.12, h=0.7):
    """384-d feature vector shown as a narrow colored bar."""
    np.random.seed(42)
    ax.imshow(np.random.rand(20, 1), extent=[x, x+w, y, y+h],
              aspect="auto", cmap="Blues", zorder=4,
              interpolation="nearest", clip_on=False)
    ax.add_patch(Rectangle((x, y), w, h, lw=0.8, edgecolor="#1A5276",
                            facecolor="none", zorder=5, clip_on=False))


def draw_feature_bars(ax, x, y, w, h, n, faded=False):
    np.random.seed(321 if faded else 654)
    bw = w / (n*1.4)
    gap = bw * 0.4
    heights = np.random.uniform(0.2, 0.95, n) * h
    alpha = 0.35 if faded else 0.85
    color = "#85C1E9" if faded else "#1A5276"
    for i in range(n):
        ax.add_patch(Rectangle((x + i*(bw+gap), y), bw, heights[i],
                                facecolor=color, alpha=alpha,
                                edgecolor="none", zorder=4, clip_on=False))


def draw_two_clusters(ax, x, y, w, h):
    """UMAP scatter with two well-separated clusters."""
    ax.add_patch(Rectangle((x, y), w, h, lw=0.6, edgecolor="#999",
                            facecolor="white", zorder=3, clip_on=False))
    np.random.seed(55)
    ax.scatter(np.random.normal(x+w*.28, w*.07, 18),
               np.random.normal(y+h*.65, h*.08, 18),
               s=9, c="#2E86C1", zorder=5, edgecolors="none", clip_on=False,
               label="Cluster A")
    ax.scatter(np.random.normal(x+w*.72, w*.07, 18),
               np.random.normal(y+h*.35, h*.08, 18),
               s=9, c="#E74C3C", zorder=5, edgecolors="none", clip_on=False,
               label="Cluster B")


def draw_one_cluster(ax, x, y, w, h):
    """UMAP scatter with a single cluster."""
    ax.add_patch(Rectangle((x, y), w, h, lw=0.6, edgecolor="#999",
                            facecolor="white", zorder=3, clip_on=False))
    np.random.seed(88)
    ax.scatter(np.random.normal(x+w*.5, w*.12, 25),
               np.random.normal(y+h*.5, h*.12, 25),
               s=9, c="#2E86C1", zorder=5, edgecolors="none", clip_on=False)


def draw_pred_scatter(ax, x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, lw=0.5, edgecolor="#888",
                            facecolor="white", zorder=4, clip_on=False))
    np.random.seed(99)
    n = 12
    xs = np.sort(np.random.uniform(x+.02, x+w-.02, n))
    ys = np.linspace(y+.03, y+h-.03, n) + np.random.normal(0, h*.06, n)
    ax.scatter(xs, ys, s=4, c="#2E86C1", zorder=6, edgecolors="none",
               clip_on=False)
    ax.plot([x+.01, x+w-.01], [y+.01, y+h-.01], "-", color="#E74C3C",
            lw=0.6, zorder=5, clip_on=False)


def draw_heatmap(ax, x, y, w, h):
    np.random.seed(77)
    ax.imshow(np.random.rand(5, 5), extent=[x, x+w, y, y+h],
              aspect="auto", cmap="YlGn", zorder=4,
              interpolation="nearest", clip_on=False)
    ax.add_patch(Rectangle((x, y), w, h, lw=0.6, edgecolor="#888",
                            facecolor="none", zorder=5, clip_on=False))


def draw_diamond(ax, cx, cy, w=0.45, h=0.55):
    verts = [(cx-w/2, cy), (cx, cy+h/2),
             (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    ax.add_patch(Polygon(verts, closed=True, facecolor=DECISION_FILL,
                         edgecolor=STAGE_BORDER, lw=1.4,
                         zorder=3, clip_on=False))
    ax.text(cx, cy+0.02, "Multiple\nclusters?",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=STAGE_BORDER, zorder=10, clip_on=False)


def draw_crop_visual(ax, x, y, imgw=0.55, imgh=0.45):
    ax.add_patch(Rectangle((x, y), imgw, imgh, lw=0.6,
                            edgecolor="#888", facecolor="#D5D5D5",
                            zorder=3, clip_on=False))
    for gx in np.linspace(x, x+imgw, 5):
        ax.plot([gx]*2, [y, y+imgh], color="#BBB", lw=0.3,
                zorder=3, clip_on=False)
    for gy in np.linspace(y, y+imgh, 4):
        ax.plot([x, x+imgw], [gy]*2, color="#BBB", lw=0.3,
                zorder=3, clip_on=False)
    cs = 0.2
    cx_, cy_ = x + imgw*0.5, y + imgh*0.35
    ax.add_patch(Rectangle((cx_, cy_), cs, cs, lw=1.2,
                            edgecolor=FROZEN_COLOR, facecolor="none",
                            ls="--", zorder=5, clip_on=False))
    arrow_h(ax, cx_+cs+0.02, cy_+cs/2, x+imgw+0.12, cy_+cs/2,
            color=FROZEN_COLOR, lw=0.8)
    ax.add_patch(Rectangle((x+imgw+0.14, cy_), cs, cs, lw=0.8,
                            edgecolor=FROZEN_COLOR, facecolor="#F8D7DA",
                            zorder=5, clip_on=False))
    ax.text(x+imgw+0.14+cs/2, cy_+cs/2, "224\u00B2",
            ha="center", va="center", fontsize=9, color=FROZEN_COLOR,
            fontweight="bold", zorder=6, clip_on=False)


# ── main figure ───────────────────────────────────────────────────────────

def create_pipeline_figure():
    fig, ax = plt.subplots(1, 1, figsize=(18, 6.5))
    ax.set_xlim(-0.5, 18.8)
    ax.set_ylim(-1.6, 5.8)
    ax.set_aspect("equal")
    ax.axis("off")

    BOX_H = 2.3
    MY  = 0.3            # main-row bottom y
    MID = MY + BOX_H/2   # centerline y

    # ── x-positions ──
    INP_X,  INP_W = 0.0,  1.6
    S1_X,   S1_W  = 2.2,  1.8
    S2_X,   S2_W  = 4.7,  1.9
    S3_X,   S3_W  = 7.3,  2.1
    S4_X,   S4_W  = 10.1, 2.2
    DIA_CX        = 13.1          # diamond center
    S5_X,   S5_W  = 14.8, 2.1
    OUT_X,  OUT_W = 17.4, 1.3

    # ──────────────────────────────────────────────────────────────────────
    # INPUT BLOCK
    # ──────────────────────────────────────────────────────────────────────
    stage_box(ax, INP_X, MY, INP_W, BOX_H, "Input Data",
              facecolor=INPUT_FILL, edgecolor=INPUT_BORDER, fontsize=16)

    draw_image_icon(ax, INP_X+0.12, MY+1.15, s=0.45)
    ax.text(INP_X+0.65, MY+1.38, "High-resolution\naerial RGB image",
            ha="left", va="center", fontsize=13, zorder=10, clip_on=False)

    draw_plot_icon(ax, INP_X+0.12, MY+0.22, s=0.45)
    ax.text(INP_X+0.65, MY+0.46, "Field plot locations\n+ biomass labels",
            ha="left", va="center", fontsize=13, zorder=10, clip_on=False)

    arrow_h(ax, INP_X+INP_W+0.06, MID, S1_X-0.06, MID)

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 1: Patch Extraction
    # ──────────────────────────────────────────────────────────────────────
    stage_box(ax, S1_X, MY, S1_W, BOX_H,
              "Patch Extraction", number=1, section_ref="Section 2.3")

    draw_crop_visual(ax, S1_X+0.25, MY+0.6)

    ax.text(S1_X+S1_W/2, MY+0.4,
            "152\u00D7152 px (Pakistan) / 224\u00D7224 px (Germany)",
            ha="center", va="center", fontsize=11, color="#444",
            zorder=10, clip_on=False)
    ax.text(S1_X+S1_W/2, MY+0.18,
            "Center crop \u2192 resize to 224\u00D7224 \u2192 normalize",
            ha="center", va="center", fontsize=10, color="#666",
            style="italic", zorder=10, clip_on=False)

    arrow_h(ax, S1_X+S1_W+0.06, MID, S2_X-0.06, MID)

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 2: Feature Extraction
    # ──────────────────────────────────────────────────────────────────────
    stage_box(ax, S2_X, MY, S2_W, BOX_H,
              "Feature Extraction",
              subtitle="DINO ViT-Small  (self-supervised)",
              number=2, section_ref="Section 2.4")

    draw_transformer(ax, S2_X+0.12, MY+0.15, w=0.65, h=1.35)

    # Output vector
    draw_feature_vector(ax, S2_X+0.95, MY+0.28, w=0.14, h=1.1)
    ax.text(S2_X+1.02, MY+0.13, "384-dim\nfeature\nvector",
            ha="center", va="top", fontsize=12, fontweight="bold",
            color="#1A5276", zorder=10, clip_on=False)

    # Frozen badge
    lx = S2_X+S2_W-0.35
    ly = MY+BOX_H-0.62
    draw_lock(ax, lx, ly, s=0.19)
    ax.text(lx, ly-0.07, "Pre-trained\n(no fine-tuning)",
            ha="center", va="top", fontsize=11, fontweight="bold",
            color=FROZEN_COLOR, zorder=10, clip_on=False)

    arrow_h(ax, S2_X+S2_W+0.06, MID, S3_X-0.06, MID)

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 3: Feature Selection
    # ──────────────────────────────────────────────────────────────────────
    stage_box(ax, S3_X, MY, S3_W, BOX_H,
              "Feature Selection",
              subtitle="Correlation with biomass",
              number=3, section_ref="Section 2.5")

    fb_y = MY + 0.6
    draw_feature_bars(ax, S3_X+0.12, fb_y, w=0.6, h=0.7, n=20, faded=True)
    ax.annotate("", xy=(S3_X+1.15, fb_y+0.35),
                xytext=(S3_X+0.82, fb_y+0.35),
                arrowprops=dict(arrowstyle="-|>", color=STAGE_BORDER, lw=1.2),
                zorder=6, clip_on=False)
    draw_feature_bars(ax, S3_X+1.2, fb_y, w=0.6, h=0.7, n=8, faded=False)

    ax.text(S3_X+0.42, fb_y-0.1, "all 384",
            ha="center", va="top", fontsize=12, color="#888",
            zorder=10, clip_on=False)
    ax.text(S3_X+1.5, fb_y-0.1, "top N",
            ha="center", va="top", fontsize=12, color="#1A5276",
            fontweight="bold", zorder=10, clip_on=False)

    ax.text(S3_X+S3_W/2, MY+0.3,
            "Rank features by correlation with biomass",
            ha="center", va="center", fontsize=11, color="#444",
            zorder=10, clip_on=False)
    ax.text(S3_X+S3_W/2, MY+0.12,
            "Keep top N features (N = 20\u2013300)",
            ha="center", va="center", fontsize=10, color="#666",
            style="italic", zorder=10, clip_on=False)

    arrow_h(ax, S3_X+S3_W+0.06, MID, S4_X-0.06, MID)

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 4: UMAP + K-Means  (applied to ALL data)
    # ──────────────────────────────────────────────────────────────────────
    stage_box(ax, S4_X, MY, S4_W, BOX_H,
              "Clustering Analysis",
              subtitle="UMAP + K-Means (applied to both sites)",
              number=4, section_ref="Section 2.6")

    # Two mini-scatter panels side by side showing different outcomes
    panel_w, panel_h = 0.6, 0.55
    gap = 0.22

    # Left panel: two clusters (Germany)
    lp_x = S4_X + 0.15
    lp_y = MY + 0.38
    draw_two_clusters(ax, lp_x, lp_y, panel_w, panel_h)
    ax.text(lp_x + panel_w/2, lp_y - 0.08, "Germany:\n2 clusters",
            ha="center", va="top", fontsize=14, color="#444",
            fontweight="bold", zorder=10, clip_on=False)

    # Right panel: single cluster (Pakistan)
    rp_x = lp_x + panel_w + gap
    draw_one_cluster(ax, rp_x, lp_y, panel_w, panel_h)
    ax.text(rp_x + panel_w/2, lp_y - 0.08, "Pakistan:\n1 cluster",
            ha="center", va="top", fontsize=14, color="#444",
            fontweight="bold", zorder=10, clip_on=False)

    arrow_h(ax, S4_X+S4_W+0.06, MID, DIA_CX-0.24, MID)

    # ──────────────────────────────────────────────────────────────────────
    # DECISION DIAMOND: Multiple clusters?
    # ──────────────────────────────────────────────────────────────────────
    draw_diamond(ax, DIA_CX, MID, w=0.55, h=0.70)

    # ── YES path (up, then right into Stage 5 top)
    yes_mid_y = MID + 1.65
    # Vertical up from diamond
    ax.add_patch(FancyArrowPatch(
        (DIA_CX, MID+0.36), (DIA_CX, yes_mid_y),
        arrowstyle="-", color=YES_COLOR,
        linewidth=1.6, zorder=1, clip_on=False))
    # Horizontal right
    ax.add_patch(FancyArrowPatch(
        (DIA_CX, yes_mid_y), (S5_X-0.06, yes_mid_y),
        arrowstyle="-|>", color=YES_COLOR,
        linewidth=1.6, mutation_scale=14, zorder=1, clip_on=False))
    # "Yes" label
    ax.text(DIA_CX+0.08, MID+0.65, "Yes",
            ha="left", va="bottom", fontsize=18, fontweight="bold",
            color=YES_COLOR, zorder=10, clip_on=False)
    # Description on the horizontal arm
    ax.text((DIA_CX + S5_X)/2, yes_mid_y + 0.12,
            "Per-cluster feature selection & models",
            ha="center", va="bottom", fontsize=14,
            color=YES_COLOR, fontweight="bold", zorder=10, clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))

    # ── NO path (down, then right into Stage 5 bottom)
    no_mid_y = MID - 1.45
    # Vertical down from diamond
    ax.add_patch(FancyArrowPatch(
        (DIA_CX, MID-0.36), (DIA_CX, no_mid_y),
        arrowstyle="-", color=NO_COLOR,
        linewidth=1.6, zorder=1, clip_on=False))
    # Horizontal right
    ax.add_patch(FancyArrowPatch(
        (DIA_CX, no_mid_y), (S5_X-0.06, no_mid_y),
        arrowstyle="-|>", color=NO_COLOR,
        linewidth=1.6, mutation_scale=14, zorder=1, clip_on=False))
    # "No" label
    ax.text(DIA_CX+0.08, MID-0.65, "No",
            ha="left", va="top", fontsize=18, fontweight="bold",
            color=NO_COLOR, zorder=10, clip_on=False)
    # Description on the horizontal arm
    ax.text((DIA_CX + S5_X)/2, no_mid_y - 0.12,
            "Single model on all data",
            ha="center", va="top", fontsize=14,
            color=NO_COLOR, fontweight="bold", zorder=10, clip_on=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))

    # ──────────────────────────────────────────────────────────────────────
    # STAGE 5: Biomass Prediction
    # ──────────────────────────────────────────────────────────────────────
    stage_box(ax, S5_X, MY, S5_W, BOX_H,
              "Biomass Prediction", number=5, section_ref="Section 2.7")

    methods = [
        ("NNLS",    "Neighbor-weighted",  "#D5E8D4", "#82B366"),
        ("Ridge",   "Linear regression",  "#DAE8FC", "#6C8EBF"),
        ("XGBoost", "Gradient boosting",  "#FFF2CC", "#D6B656"),
    ]
    sub_w = S5_W * 0.88
    sub_h = 0.36
    sub_x = S5_X + (S5_W - sub_w)/2
    total = len(methods)*sub_h + (len(methods)-1)*0.14
    start_y = MY + (BOX_H - total)/2 - 0.1

    for i, (name, desc, fc, ec) in enumerate(methods):
        sy = start_y + i*(sub_h + 0.14)
        small_box(ax, sub_x, sy, sub_w, sub_h, facecolor=fc, edgecolor=ec, lw=1.0)
        ax.text(sub_x + sub_w*0.28, sy + sub_h/2, name,
                ha="center", va="center", fontsize=18,
                fontweight="bold", zorder=6, clip_on=False)
        ax.text(sub_x + sub_w*0.70, sy + sub_h/2, desc,
                ha="center", va="center", fontsize=13,
                color="#555", zorder=6, clip_on=False)

    # Both yes/no arrows should visually merge into stage 5
    # (the L-shaped paths already point at the stage 5 box edges)

    arrow_h(ax, S5_X+S5_W+0.06, MID, OUT_X-0.06, MID)

    # ──────────────────────────────────────────────────────────────────────
    # OUTPUT BLOCK
    # ──────────────────────────────────────────────────────────────────────
    stage_box(ax, OUT_X, MY, OUT_W, BOX_H, "Output",
              facecolor=OUTPUT_FILL, edgecolor=OUTPUT_BORDER, fontsize=16)

    draw_pred_scatter(ax, OUT_X+0.18, MY+1.2, w=0.36, h=0.36)
    ax.text(OUT_X+OUT_W/2, MY+1.08,
            "Plot-level AGB\nestimate (t/ha)",
            ha="center", va="top", fontsize=13, zorder=10, clip_on=False)

    draw_heatmap(ax, OUT_X+0.18, MY+0.22, w=0.36, h=0.36)
    ax.text(OUT_X+OUT_W/2, MY+0.1,
            "Wall-to-wall\nbiomass map",
            ha="center", va="top", fontsize=13, zorder=10, clip_on=False)

    # ── save ──
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for ext in ("svg", "png", "pdf"):
        path = os.path.join(out_dir, f"fig2_methodological_pipeline.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none", pad_inches=0.2)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    create_pipeline_figure()
