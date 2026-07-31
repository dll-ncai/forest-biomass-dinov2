#!/usr/bin/env python3
"""
Regenerate every results figure from the experiment JSONs in results/.
Figures are written to figures/ as PNG (300 dpi) + SVG + PDF, and the headline
ones are copied into paper_writeup/ when that directory exists.

Reads only the committed JSONs, so it re-runs in seconds. Any figure whose
source JSON is missing is skipped with a warning (e.g. robustness before it finishes).

Colour = identity of the feature backbone, fixed order, Okabe-Ito (colour-blind safe):
    DINO = blue,  ResNet50 = orange,  Literature(GLCM) = bluish-green.
"""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "paper_writeup")
os.makedirs(OUT, exist_ok=True)

# ---- Okabe-Ito colour-blind-safe categorical palette (fixed identity order) ----
C = {"dino": "#0072B2", "resnet50": "#E69F00", "literature": "#009E73",
     "nnls": "#0072B2", "ridge": "#D55E00", "xgb": "#CC79A7",
     "proposed": "#0072B2", "no_clustering": "#E69F00", "all_features": "#009E73"}
GRID = "#D9D9D9"; INK = "#222222"; MUTED = "#666666"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "font.size": 11,
    "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
})


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        warnings.warn(f"missing {name}; skipping figures that need it")
        return None
    try:
        return json.load(open(p))
    except Exception as e:
        warnings.warn(f"could not parse {name}: {e}")
        return None


def save(fig, stem, also_paper=False):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(os.path.join(OUT, f"{stem}.{ext}"), bbox_inches="tight")
    if also_paper and os.path.isdir(PAPER):
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(PAPER, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem} (png/svg/pdf)")


def best_of(methods_metrics):
    """methods_metrics: {name:{percent_rmse,r2}}. Return (name, dict) w/ min %RMSE."""
    items = [(n, m) for n, m in methods_metrics.items() if m and np.isfinite(m.get("percent_rmse", np.inf))]
    return min(items, key=lambda kv: kv[1]["percent_rmse"]) if items else (None, None)


def barlabels(ax, bars, fmt="{:.1f}"):
    for b in bars:
        h = b.get_height()
        if np.isfinite(h):
            ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h),
                    ha="center", va="bottom", fontsize=8.5, color=INK)


# ───────────────── data assembly ─────────────────
def pak_methods(feat):
    d = load(f"pakistan_{feat}.json")
    return {m: r["test"] for m, r in d["methods"].items()} if d else {}

def ger_methods(feat):
    d = load(f"german_{feat}.json")
    return {m: r["combined_test"] for m, r in d["methods"].items()} if d else {}

lit = load("literature_baselines.json") or {}


# ───────────────── FIG 1: performance comparison (both sites) ─────────────────
def fig_performance():
    methods = ["nnls", "ridge", "xgb"]
    mlabel = {"nnls": "NNLS", "ridge": "Ridge", "xgb": "XGBoost"}
    sites = [("Pakistan (Balakot) — nested LOO", pak_methods("dino"), pak_methods("resnet50"),
              lit.get("pakistan", {})),
             ("Germany (Karlsruhe) — 85/15 hold-out", ger_methods("dino"), ger_methods("resnet50"),
              lit.get("german", {}))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    handles = None
    for ax, (title, dino, resnet, litd) in zip(axes, sites):
        x = np.arange(len(methods)); w = 0.38
        dv = [dino.get(m, {}).get("percent_rmse", np.nan) for m in methods]
        rv = [resnet.get(m, {}).get("percent_rmse", np.nan) for m in methods]
        b1 = ax.bar(x - w/2, dv, w, color=C["dino"], label="DINO (384-d)")
        b2 = ax.bar(x + w/2, rv, w, color=C["resnet50"], label="ResNet50 (2048-d)")
        barlabels(ax, b1); barlabels(ax, b2)
        _, lbest = best_of(litd)
        if lbest:
            lv = lbest["percent_rmse"]
            ax.axhline(lv, ls="--", lw=1.6, color=C["literature"], zorder=1)
            ax.text(len(methods) - 0.5, lv, f" best literature/GLCM = {lv:.1f}",
                    va="bottom", ha="right", fontsize=8.5, color=C["literature"])
        ax.set_xticks(x); ax.set_xticklabels([mlabel[m] for m in methods])
        ax.set_ylabel("%RMSE  (lower is better)")
        ax.set_title(title, fontsize=10.5)
        top = np.nanmax(dv + rv)
        ax.set_ylim(0, top * 1.16)
        ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
        handles = [b1, b2]
    fig.legend(handles, ["DINO (384-d)", "ResNet50 (2048-d)"],
               loc="upper center", ncol=2, fontsize=9.5, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Biomass regression — %RMSE by backbone & model",
                 fontsize=12, y=1.06)
    save(fig, "fig_performance", also_paper=True)


# ───────────────── FIG 2: best-model scatter per site ─────────────────
def fig_scatter():
    panels = []
    dp = load("pakistan_dino.json")
    if dp:
        bn, _ = best_of({m: r["test"] for m, r in dp["methods"].items()})
        r = dp["methods"][bn]
        panels.append(("Pakistan — DINO + %s" % bn.upper(), r["y_true"], r["y_pred"],
                       r["test"], "Biomass (t/ha)"))
    dg = load("german_dino.json")
    if dg:
        bn, _ = best_of({m: rr["combined_test"] for m, rr in dg["methods"].items()})
        r = dg["methods"][bn]
        panels.append(("Germany — DINO + %s" % bn.upper(), r["pooled_true"], r["pooled_pred"],
                       r["combined_test"], "Biomass (t/ha)"))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.3 * len(panels), 4.6))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, yt, yp, mt, unit) in zip(axes, panels):
        yt, yp = np.asarray(yt, float), np.asarray(yp, float)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1, ls="--",
                label="1:1")
        ax.scatter(yt, yp, s=46, color=C["dino"], edgecolor="white", linewidth=0.7, zorder=3)
        ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel(f"Observed {unit}"); ax.set_ylabel(f"Predicted {unit}")
        ax.set_title(title, fontsize=10.5)
        ax.text(0.05, 0.95, f"R$^2$ = {mt['r2']:.3f}\n%RMSE = {mt['percent_rmse']:.2f}\n"
                f"RMSE = {mt['rmse']:.2f}\nn = {len(yt)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID))
        ax.legend(loc="lower right", fontsize=9)
        ax.set_aspect("equal", "box")
    fig.suptitle("Best-model predictions vs. observations", fontsize=12, y=1.02)
    save(fig, "fig_scatter", also_paper=True)


# ───────────────── FIG 3: German ablation (DINO) ─────────────────
def fig_ablation():
    main = load("german_dino.json"); abl = load("ablations_german_dino.json")
    if not (main and abl):
        return
    methods = ["nnls", "ridge", "xgb"]; mlabel = {"nnls": "NNLS", "ridge": "Ridge", "xgb": "XGBoost"}
    proposed = {m: main["methods"][m]["combined_test"]["percent_rmse"] for m in methods
                if m in main["methods"]}
    allf = {m: abl["all_features"][m]["percent_rmse"] for m in methods if m in abl["all_features"]}
    noclu = {m: abl["no_clustering"][m]["test"]["percent_rmse"] for m in methods
             if m in abl["no_clustering"]}
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(methods)); w = 0.26
    b1 = ax.bar(x - w, [proposed.get(m, np.nan) for m in methods], w,
                color=C["proposed"], label="Proposed (cluster + train-only selection)")
    b2 = ax.bar(x, [noclu.get(m, np.nan) for m in methods], w,
                color=C["no_clustering"], label="No clustering (global model)")
    b3 = ax.bar(x + w, [allf.get(m, np.nan) for m in methods], w,
                color=C["all_features"], label="All 384 features (no selection)")
    for bs in (b1, b2, b3):
        barlabels(ax, bs)
    ax.set_xticks(x); ax.set_xticklabels([mlabel[m] for m in methods])
    ax.set_ylabel("%RMSE  (lower is better)")
    ax.set_title("Germany ablation (DINO)", fontsize=11)
    ax.legend(fontsize=8.5); ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    save(fig, "fig_ablation_german", also_paper=True)


# ───────────────── FIG 4: performance vs. #features (German DINO CV curve) ─────
def fig_perf_vs_features():
    d = load("german_dino.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # colour = method (identity); line style = cluster (solid cluster 0, dashed cluster 1)
    STYLE = {"0": "-", "1": "--"}; MRK = {"0": "o", "1": "s"}
    drew = False
    from matplotlib.lines import Line2D
    for m, col in [("nnls", C["nnls"]), ("ridge", C["ridge"]), ("xgb", C["xgb"])]:
        r = d["methods"].get(m)
        if not r:
            continue
        for cid, pc in r["per_cluster"].items():
            cid = str(cid)
            if isinstance(pc, dict) and pc.get("cv_curve"):
                ks = sorted(int(k) for k in pc["cv_curve"])
                vs = [pc["cv_curve"][str(k)] for k in ks]
                ax.plot(ks, vs, STYLE.get(cid, "-"), marker=MRK.get(cid, "o"), ms=3.5,
                        lw=1.7, color=col, alpha=0.9)
                drew = True
    if not drew:
        plt.close(fig); return
    method_handles = [Line2D([], [], color=C[m], lw=2.4, label=lab)
                      for m, lab in [("nnls", "NNLS"), ("ridge", "Ridge"), ("xgb", "XGBoost")]]
    cluster_handles = [Line2D([], [], color=MUTED, lw=1.7, ls=STYLE[c], marker=MRK[c],
                              label=f"cluster {c}") for c in ("0", "1")]
    leg1 = ax.legend(handles=method_handles, title="model (colour)", fontsize=9,
                     title_fontsize=9, loc="upper left")
    ax.add_artist(leg1)
    ax.legend(handles=cluster_handles, title="cluster (style)", fontsize=9,
              title_fontsize=9, loc="upper right")
    ax.set_xlabel("Number of top-|corr| features (K), selected on train")
    ax.set_ylabel("Inner-CV %RMSE")
    ax.set_title("Germany DINO — train-only CV vs. feature count", fontsize=11)
    ax.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    save(fig, "fig_perf_vs_features", also_paper=True)


# ───────────────── FIG 5: robustness over 20 splits (German) ─────────────────
def fig_robustness():
    d = load("robustness_german.json")
    if not d or "summary" not in d:
        return
    s = d["summary"]
    backbones = [("dino", "DINO", C["dino"]), ("resnet50", "ResNet50", C["resnet50"])]
    methods = ["nnls", "ridge", "xgb"]; mlabel = {"nnls": "NNLS", "ridge": "Ridge", "xgb": "XGBoost"}
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    x = np.arange(len(methods)); w = 0.36
    for i, (bk, blab, col) in enumerate(backbones):
        means = [s.get(f"{bk}|proposed|{m}", {}).get("prmse_mean", np.nan) for m in methods]
        stds = [s.get(f"{bk}|proposed|{m}", {}).get("prmse_std", np.nan) for m in methods]
        bars = ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, capsize=4,
                      color=col, label=blab, error_kw=dict(ecolor=MUTED, lw=1))
        barlabels(ax, bars)
    litmean = np.nanmin([s.get(f"glcm|literature|{m}", {}).get("prmse_mean", np.nan)
                         for m in ["rf", "gbt", "xgb", "ridge"]])
    if np.isfinite(litmean):
        ax.axhline(litmean, ls="--", lw=1.6, color=C["literature"],
                   label=f"Best literature/GLCM ({litmean:.1f})")
    ax.set_xticks(x); ax.set_xticklabels([mlabel[m] for m in methods])
    ax.set_ylabel("%RMSE  (mean ± std over splits)")
    ax.set_title(f"Germany robustness — proposed pipeline over {d.get('seeds','?')} "
                 f"stratified 85/15 splits", fontsize=11)
    ax.legend(fontsize=8.5); ax.yaxis.grid(True, color=GRID, lw=0.7); ax.set_axisbelow(True)
    save(fig, "fig_robustness_german", also_paper=True)


def main():
    print("Generating results figures -> figures/")
    for fn in (fig_performance, fig_scatter, fig_ablation, fig_perf_vs_features, fig_robustness):
        try:
            fn()
        except Exception as e:
            warnings.warn(f"{fn.__name__} failed: {e}")
    print("FIGURES_DONE")


if __name__ == "__main__":
    main()
