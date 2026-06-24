#!/usr/bin/env python3
"""
Nested LOO evaluation for models using top-ranked features vs. all features.
Outer loop: LOO validation (25 folds). Left out: 1 test sample, 24 train samples.
Inner loop (inside hyperparameter tuning functions): LOO validation (24 folds) 
on the 24 train samples to find the best hyperparameters.
The best model is run on the test sample.
Generates metrics, a comparison table, and scatter plots for best K vs all.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Re-use methods from the original script
from find_best_models_top_features import (
    get_feature_and_label_columns,
    load_ranked_feature_indices,
    compute_metrics,
    tune_nnls_loo,
    tune_ridge_loo,
    tune_xgb_loo,
    predict_nnls,
    predict_linear,
    predict_xgb,
    METHOD_DISPLAY, METHOD_COLORS, HAS_XGB, METHODS
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
FEATURE_COUNTS = list(range(20, 251, 10))

def run_nested_loo_for_features(X: np.ndarray, y: np.ndarray, method: str) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, List[Dict]]:
    n = X.shape[0]
    preds = []
    truths = []
    outer_best_configs = []
    
    for i in range(n):
        # Outer splits (24 train, 1 test)
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]
        X_test, y_test = X[~mask], y[~mask]
        
        # Inner loop: tune hyperparameters on the 24 train samples
        if method == "xgb":
            best_cfg, _, _, _ = tune_xgb_loo(X_train, y_train)
            p = predict_xgb(X_train, y_train, X_test, best_cfg)
        elif method == "nnls":
            best_cfg, _, _, _ = tune_nnls_loo(X_train, y_train)
            p = predict_nnls(X_train, y_train, X_test, 
                             best_cfg.get("neighbor_k", 3), 
                             best_cfg.get("cosine", True), 
                             best_cfg.get("simplex", True))
        elif method == "linear":
            best_cfg, _, _, _ = tune_ridge_loo(X_train, y_train)
            p = predict_linear(X_train, y_train, X_test, 
                               best_cfg.get("scale", True), 
                               best_cfg.get("alpha", 10.0))
        else:
            raise ValueError(method)
            
        preds.append(float(p[0]))
        truths.append(float(y_test[0]))
        outer_best_configs.append(best_cfg)

    preds = np.array(preds)
    truths = np.array(truths)
    metrics = compute_metrics(truths, preds)
    return metrics, truths, preds, outer_best_configs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Nested LOO validation on top features")
    parser.add_argument("--input_csv", type=str, default=None, help="Full features CSV")
    parser.add_argument("--correlations_csv", type=str, default=None, help="feature_correlations.csv from correlation analysis")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: script dir)")
    args = parser.parse_args()
    
    output_dir = args.output_dir or SCRIPT_DIR
    input_csv = args.input_csv or os.path.join(BASE_DIR, "dino_features_with_labels_and_split.csv")
    correlations_csv = args.correlations_csv or os.path.join(SCRIPT_DIR, "feature_correlations.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("NESTED LOO EVALUATION")
    print("=" * 60)
    print("\nLoading data...")
    df = pd.read_csv(input_csv)
    feature_cols, label_col = get_feature_and_label_columns(df)
    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df[label_col].values.astype(np.float32)
    # Convert Total Biomass (kg) to Biomass Density (t/ha)
    # Circular field plot: radius = 17.5 m, area = π × 17.5² ≈ 962.1 m² = 0.09621 ha
    # t/ha = (kg / 1000) / 0.09621 = kg × 0.01040
    import math
    circle_area_m2 = math.pi * 17.5 ** 2   # ≈ 962.1 m²
    y_full = (y_full / 1000.0) * (10000.0 / circle_area_m2)  # kg → t/ha
    
    n_total, n_features_total = X_full.shape
    print(f"  Samples: {n_total}, Full features: {n_features_total}")
    print(f"  Biomass range (t/ha): {y_full.min():.2f} – {y_full.max():.2f}")
    
    print("\nLoading feature ranking...")
    ranked_indices = load_ranked_feature_indices(correlations_csv, feature_cols)
    print(f"  Ranked {len(ranked_indices)} features")
    
    feature_counts = [n for n in FEATURE_COUNTS if n <= len(ranked_indices)]
    counts_to_eval = feature_counts + ["all"]
    print(f"\nEvaluating Feature counts: {counts_to_eval}")
    
    all_results = {}
    rows = []
    
    # dict to store predictions for scatter plots
    # dict[method][count] = (metrics, y_true, y_pred)
    scatter_data = {m: {} for m in METHODS}
    
    for count in counts_to_eval:
        # Skip XGBoost for > 60 features to save compute time (we already know it degrades fast)
        # This condition is checked for each method inside the inner loop, but we want to skip the whole count for XGB if applicable.
        # So, we'll move the check for XGBoost inside the method loop.
        
        print(f"\n  Evaluating structure: {count} features ...")
        if count == "all":
            X = X_full
            n_feat = len(ranked_indices)
        else:
            top_idx = ranked_indices[:int(count)]
            X = X_full[:, top_idx]
            n_feat = int(count)
            
        all_results[count] = {}
        
        for method in METHODS:
            if method == "xgb" and not HAS_XGB:
                continue
                
            if method == "xgb" and str(count) != "all" and isinstance(count, int) and count > 60:
                continue
            
            metrics, y_true, y_pred, outer_configs = run_nested_loo_for_features(X, y_full, method)
            
            all_results[count][method] = {
                "rmse": metrics["rmse"], 
                "%rmse": metrics["%rmse"],
                "mae": metrics["mae"], 
                "r2": metrics["r2"],
                "y_true": y_true.tolist(), 
                "y_pred": y_pred.tolist(),
                "configs_per_fold": outer_configs
            }
            rows.append({
                "n_features": n_feat if count != "all" else "all",
                "method": METHOD_DISPLAY[method],
                "rmse": metrics["rmse"],
                "percent_rmse": metrics["%rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"]
            })
            scatter_data[method][count] = (metrics, y_true, y_pred)
            print(f"    {METHOD_DISPLAY[method]}: %RMSE={metrics['%rmse']:.2f}, R²={metrics['r2']:.4f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "nested_loo_results.csv"), index=False)
    
    # Find Best vs All Comparison
    comparison_rows = []
    best_config_for_plot = {} # dict[method] = best_count
    
    for method in METHODS:
        if method == "xgb" and not HAS_XGB:
            continue
        sub = results_df[results_df["method"] == METHOD_DISPLAY[method]]
        
        # Find best numerical count
        numeric_sub = sub[sub["n_features"] != "all"].copy()
        best_row = numeric_sub.loc[numeric_sub["percent_rmse"].idxmin()]
        best_count = best_row["n_features"]
        
        best_config_for_plot[method] = best_count
        
        all_row = sub[sub["n_features"] == "all"].iloc[0]
        
        comparison_rows.append({
            "Method": METHOD_DISPLAY[method],
            "Best Top K": int(best_count),
            "Best K %RMSE": best_row["percent_rmse"],
            "Best K R²": best_row["r2"],
            "All Features %RMSE": all_row["percent_rmse"],
            "All Features R²": all_row["r2"],
        })
        
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(output_dir, "nested_loo_best_vs_all.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("BEST VS ALL COMPARISON (Nested LOO)")
    print("=" * 60)
    print(comp_df.to_string(index=False))

    # --- Plot 1: Performance vs n_features (Outer Nested LOO curve) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method in METHODS:
        if method == "xgb" and not HAS_XGB: continue
        sub = results_df[results_df["method"] == METHOD_DISPLAY[method]]
        # Filter for integer n_features
        sub_num = sub[sub["n_features"] != "all"].copy()
        sub_num["n_features"] = sub_num["n_features"].astype(int)
        sub_num = sub_num.sort_values("n_features")
        
        axes[0].plot(sub_num["n_features"], sub_num["percent_rmse"], marker="o", label=METHOD_DISPLAY[method],
                     color=METHOD_COLORS.get(method, "gray"), linewidth=2, markersize=5)
        axes[1].plot(sub_num["n_features"], sub_num["r2"], marker="o", label=METHOD_DISPLAY[method],
                     color=METHOD_COLORS.get(method, "gray"), linewidth=2, markersize=5)

        # Plot the "all" point as a dashed line
        all_row = sub[sub["n_features"] == "all"].iloc[0]
        axes[0].axhline(y=all_row["percent_rmse"], color=METHOD_COLORS.get(method, "gray"), linestyle="--", alpha=0.5)
        axes[1].axhline(y=all_row["r2"], color=METHOD_COLORS.get(method, "gray"), linestyle="--", alpha=0.5)

    axes[0].set_xlabel("Number of top features", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("% RMSE", fontsize=12, fontweight="bold")
    axes[0].set_title("Nested LOO % RMSE vs Top Features", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Number of top features", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("R²", fontsize=12, fontweight="bold")
    axes[1].set_title("Nested LOO R² vs Top Features", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle("Model Performance vs Number of Features (Nested LOO)\nDashed lines indicate 'All Features' baseline", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nested_loo_performance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # --- Plot 2: Scatter plots (Predicted vs Actual) ---
    methods_present = [m for m in METHODS if m in best_config_for_plot]
    fig, axes = plt.subplots(len(methods_present), 2, figsize=(12, 5 * len(methods_present)))
    if len(methods_present) == 1: axes = np.array([axes])

    for i, method in enumerate(methods_present):
        best_count = best_config_for_plot[method]
        
        # Best K Scatter
        ax1 = axes[i, 0]
        mets_best, yt_best, yp_best = scatter_data[method][best_count]
        ax1.scatter(yt_best, yp_best, color=METHOD_COLORS.get(method, "gray"), alpha=0.7, edgecolors="k")
        ax1.plot([min(yt_best), max(yt_best)], [min(yt_best), max(yt_best)], "k--", lw=2)
        ax1.set_xlabel("Actual Biomass (t/ha)")
        ax1.set_ylabel("Predicted Biomass (t/ha)")
        ax1.set_title(f"{METHOD_DISPLAY[method]} (Best K: {best_count})\n%RMSE={mets_best['%rmse']:.2f}, R²={mets_best['r2']:.4f}")
        ax1.grid(True, alpha=0.3)
        
        # All Features Scatter
        ax2 = axes[i, 1]
        mets_all, yt_all, yp_all = scatter_data[method]["all"]
        ax2.scatter(yt_all, yp_all, color=METHOD_COLORS.get(method, "gray"), alpha=0.7, edgecolors="k")
        ax2.plot([min(yt_all), max(yt_all)], [min(yt_all), max(yt_all)], "k--", lw=2)
        ax2.set_xlabel("Actual Biomass (t/ha)")
        ax2.set_ylabel("Predicted Biomass (t/ha)")
        ax2.set_title(f"{METHOD_DISPLAY[method]} (All Features)\n%RMSE={mets_all['%rmse']:.2f}, R²={mets_all['r2']:.4f}")
        ax2.grid(True, alpha=0.3)

    plt.suptitle("Nested LOO Scatter Plots: Predicted vs Actual\nBest Top-K vs All Features", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nested_loo_scatter_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 2.5: Separate Scatter plot for ONLY the best configs ---
    fig, axes = plt.subplots(1, len(methods_present), figsize=(5 * len(methods_present), 5))
    if len(methods_present) == 1: axes = [axes]

    for i, method in enumerate(methods_present):
        best_count = best_config_for_plot[method]
        ax = axes[i]
        mets_best, yt_best, yp_best = scatter_data[method][best_count]
        ax.scatter(yt_best, yp_best, color=METHOD_COLORS.get(method, "gray"), alpha=0.7, edgecolors="k")
        ax.plot([min(yt_best), max(yt_best)], [min(yt_best), max(yt_best)], "k--", lw=2)
        ax.set_xlabel("Actual Biomass (t/ha)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Biomass (t/ha)", fontsize=11, fontweight="bold")
        ax.set_title(f"{METHOD_DISPLAY[method]} (Best K: {best_count})\n%RMSE={mets_best['%rmse']:.2f}, R²={mets_best['r2']:.4f}", fontsize=12)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Nested LOO Scatter Plots: Best Feature Count per Model", fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nested_loo_best_scatter_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot 3: Summary text table ---
    fig, ax = plt.subplots(figsize=(12, max(3, len(comp_df) * 0.5)))
    ax.axis("off")
    # Format floats for display
    disp_df = comp_df.copy()
    disp_df["Best K %RMSE"] = disp_df["Best K %RMSE"].apply(lambda x: f"{x:.2f}%")
    disp_df["Best K R²"] = disp_df["Best K R²"].apply(lambda x: f"{x:.4f}")
    disp_df["All Features %RMSE"] = disp_df["All Features %RMSE"].apply(lambda x: f"{x:.2f}%")
    disp_df["All Features R²"] = disp_df["All Features R²"].apply(lambda x: f"{x:.4f}")

    table = ax.table(
        cellText=disp_df.values,
        colLabels=disp_df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#e0e0e0"] * len(disp_df.columns),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5)
    plt.title("Best Top-K vs All Features (Nested LOO)\nPakistani Dataset", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nested_loo_best_vs_all_table.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n✅ All nested LOO artifacts saved successfully in {output_dir}")

if __name__ == "__main__":
    main()
