#!/usr/bin/env python3
"""
Regenerate the Pakistani nested LOO comparison figure from existing CSVs.
Reads the verified result CSVs and produces:
  - paper_methods_nested_loo_comparison.svg  (publication-ready)
  - paper_methods_nested_loo_comparison.png  (backup raster)
  - paper_methods_nested_loo_comparison.csv  (updated sorted comparison table)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── Publication style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 27,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "axes.linewidth": 1.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

dino_csv = os.path.join(SCRIPT_DIR, "nested_loo_best_vs_all.csv")
lit_csv  = os.path.join(SCRIPT_DIR, "paper_methods_nested_loo_results.csv")
out_dir  = SCRIPT_DIR

dino_df = pd.read_csv(dino_csv)
lit_df  = pd.read_csv(lit_csv)

# ── Build unified comparison rows ────────────────────────────────────────────
rows = []

method_label = {
    "Linear Regression": "Ridge Regression",
    "XGBoost": "XGBoost",
    "NNLS": "NNLS",
}
for _, r in dino_df.iterrows():
    label = method_label.get(r["Method"], r["Method"])
    rows.append({
        "Method": f"DINO + {label} (Top {int(r['Best Top K'])})",
        "%RMSE": round(r["Best K %RMSE"], 2),
        "R²":    round(r["Best K R²"],    3),
        "Source": "Proposed (DINO)",
    })

for _, r in lit_df.iterrows():
    rows.append({
        "Method": r["Method"],
        "%RMSE": round(r["%RMSE"], 2),
        "R²":    round(r["R²"],    3),
        "Source": "Literature",
    })

comp_df = pd.DataFrame(rows).sort_values("%RMSE").reset_index(drop=True)
comp_df.to_csv(os.path.join(out_dir, "paper_methods_nested_loo_comparison.csv"), index=False)

print("Comparison table (sorted by %RMSE):")
print(comp_df[["Method", "%RMSE", "R²", "Source"]].to_string(index=False))

# ── Bar chart ────────────────────────────────────────────────────────────────
methods = comp_df["Method"].tolist()
prmse   = comp_df["%RMSE"].tolist()
r2_vals = comp_df["R²"].tolist()
n       = len(methods)

COLOR_DINO = "#2ecc71"
COLOR_LIT  = "#e74c3c"
colors = [COLOR_DINO if s == "Proposed (DINO)" else COLOR_LIT
          for s in comp_df["Source"]]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.subplots_adjust(left=0.32, right=0.97, top=0.90, bottom=0.08, wspace=0.45)

bar_h = 0.55

# ── Panel (a): %RMSE ─────────────────────────────────────────────────────────
ax1 = axes[0]
y_pos = list(range(n))
bars1 = ax1.barh(y_pos, prmse, height=bar_h, color=colors, edgecolor="k",
                 linewidth=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(methods, fontsize=17)
ax1.invert_yaxis()
ax1.set_xlabel("%RMSE (%)", fontsize=27, labelpad=8)
ax1.grid(axis="x", alpha=0.35, linewidth=0.6)
ax1.set_axisbelow(True)
x_max = max(prmse) * 1.18
ax1.set_xlim(0, x_max)
for bar, val in zip(bars1, prmse):
    ax1.text(val + x_max * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f"{val:.1f}%", va="center", fontsize=16)

# Subplot label (a)
ax1.text(-0.02, 1.04, "(a)", transform=ax1.transAxes,
         fontsize=30, fontweight="bold", va="bottom", ha="right")

# ── Panel (b): R² ────────────────────────────────────────────────────────────
ax2 = axes[1]
bars2 = ax2.barh(y_pos, r2_vals, height=bar_h, color=colors, edgecolor="k",
                 linewidth=0.7)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(methods, fontsize=17)
ax2.invert_yaxis()
ax2.set_xlabel("R²", fontsize=27, labelpad=8)
ax2.grid(axis="x", alpha=0.35, linewidth=0.6)
ax2.set_axisbelow(True)
x_max2 = 1.12
ax2.set_xlim(0, x_max2)
for bar, val in zip(bars2, r2_vals):
    ax2.text(val + x_max2 * 0.01,
             bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=16)

# Subplot label (b)
ax2.text(-0.02, 1.04, "(b)", transform=ax2.transAxes,
         fontsize=30, fontweight="bold", va="bottom", ha="right")

# ── Shared legend ────────────────────────────────────────────────────────────
legend_els = [
    Patch(facecolor=COLOR_DINO, edgecolor="k", linewidth=0.7,
          label="Proposed (DINO)"),
    Patch(facecolor=COLOR_LIT,  edgecolor="k", linewidth=0.7,
          label="Literature baseline"),
]
fig.legend(handles=legend_els, loc="upper center", ncol=2,
           fontsize=24, frameon=True, edgecolor="#AAAAAA",
           bbox_to_anchor=(0.64, 0.97))

# ── Save ─────────────────────────────────────────────────────────────────────
for ext in ("svg", "png"):
    out = os.path.join(out_dir, f"paper_methods_nested_loo_comparison.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

paper_dir = os.path.join(SCRIPT_DIR, "..", "paper_writeup")
for ext in ("svg", "png"):
    out = os.path.join(paper_dir, f"paper_methods_nested_loo_comparison.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

plt.close()
