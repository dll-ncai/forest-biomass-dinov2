#!/usr/bin/env python3
"""
Analyze which features in the feature vector correlate most strongly with biomass.
Computes Pearson and Spearman correlations for each feature and ranks them.
Uses the same methodology as the German data analysis.
Works with any feature columns (e.g. pca_0, pca_1, ... or feature_0, feature_1, ...).
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from scipy.stats import pearsonr, spearmanr


def get_feature_and_label_columns(df: pd.DataFrame):
    """Get feature columns and label column. Supports 'label' or 'biomass'."""
    label_col = None
    if "label" in df.columns:
        label_col = "label"
    elif "biomass" in df.columns:
        label_col = "biomass"
    else:
        raise ValueError("CSV must have 'label' or 'biomass' column.")
    exclude = {label_col, "split"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols, label_col


def compute_feature_correlations(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Compute correlations between each feature and biomass.
    Returns DataFrame with feature indices, names, correlations, and p-values.
    """
    print("Computing feature-biomass correlations...")
    print(f"  Input: {X.shape[0]} samples, {X.shape[1]} features")
    n_features = X.shape[1]
    correlations = []
    for i in range(n_features):
        feature_values = X[:, i]
        if np.std(feature_values) < 1e-10:
            continue
        pearson_r, pearson_p = pearsonr(feature_values, y)
        spearman_r, spearman_p = spearmanr(feature_values, y)
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        correlations.append({
            "feature_idx": i,
            "feature_name": name,
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "pearson_abs": float(abs(pearson_r)),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "spearman_abs": float(abs(spearman_r)),
            "feature_mean": float(np.mean(feature_values)),
            "feature_std": float(np.std(feature_values)),
            "feature_min": float(np.min(feature_values)),
            "feature_max": float(np.max(feature_values)),
        })
    df = pd.DataFrame(correlations)
    print(f"  Computed correlations for {len(df)} features")
    if len(df) > 0:
        print(f"  Pearson |r| range: {df['pearson_abs'].min():.4f} - {df['pearson_abs'].max():.4f}")
        print(f"  Spearman |r| range: {df['spearman_abs'].min():.4f} - {df['spearman_abs'].max():.4f}")
    return df


def visualize_top_features(
    X: np.ndarray,
    y: np.ndarray,
    correlations_df: pd.DataFrame,
    output_dir: str,
    top_n: int = 20,
):
    """Create visualizations for top correlated features."""
    top_pearson = correlations_df.nlargest(min(top_n, len(correlations_df)), "pearson_abs")
    top_spearman = correlations_df.nlargest(min(top_n, len(correlations_df)), "spearman_abs")
    fig = plt.figure(figsize=(20, 16))
    # Plot 1: Top features by Pearson (bar)
    ax1 = plt.subplot(3, 3, 1)
    top_pearson_sorted = top_pearson.sort_values("pearson_r", ascending=True)
    colors = ["red" if r < 0 else "blue" for r in top_pearson_sorted["pearson_r"]]
    ax1.barh(range(len(top_pearson_sorted)), top_pearson_sorted["pearson_r"], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_pearson_sorted)))
    ax1.set_yticklabels(list(top_pearson_sorted["feature_name"]), fontsize=8)
    ax1.set_xlabel("Pearson Correlation (r)", fontsize=11, fontweight="bold")
    ax1.set_title(f"Top {len(top_pearson_sorted)} Features - Pearson Correlation", fontsize=12, fontweight="bold")
    ax1.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis="x")
    # Plot 2: Top features by Spearman (bar)
    ax2 = plt.subplot(3, 3, 2)
    top_spearman_sorted = top_spearman.sort_values("spearman_r", ascending=True)
    colors = ["red" if r < 0 else "blue" for r in top_spearman_sorted["spearman_r"]]
    ax2.barh(range(len(top_spearman_sorted)), top_spearman_sorted["spearman_r"], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(top_spearman_sorted)))
    ax2.set_yticklabels(list(top_spearman_sorted["feature_name"]), fontsize=8)
    ax2.set_xlabel("Spearman Correlation (ρ)", fontsize=11, fontweight="bold")
    ax2.set_title(f"Top {len(top_spearman_sorted)} Features - Spearman Correlation", fontsize=12, fontweight="bold")
    ax2.axvline(x=0, color="black", linestyle="--", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="x")
    # Plot 3: Pearson vs Spearman scatter
    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(correlations_df["pearson_r"], correlations_df["spearman_r"], alpha=0.6, s=50, edgecolors="black", linewidths=0.3)
    ax3.plot([-1, 1], [-1, 1], "r--", linewidth=1, alpha=0.5, label="y=x")
    ax3.set_xlabel("Pearson Correlation (r)", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Spearman Correlation (ρ)", fontsize=11, fontweight="bold")
    ax3.set_title("Pearson vs Spearman Correlations", fontsize=12, fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([-1, 1])
    ax3.set_ylim([-1, 1])
    # Plots 4–8: Scatter for top 5 features (Pearson)
    top_5_pearson = top_pearson.nlargest(min(5, len(top_pearson)), "pearson_abs")
    for idx, (_, row) in enumerate(top_5_pearson.iterrows()):
        ax = plt.subplot(3, 3, 4 + idx)
        fi = int(row["feature_idx"])
        fv = X[:, fi]
        ax.scatter(fv, y, alpha=0.7, s=80, edgecolors="black", linewidths=0.3)
        ax.set_xlabel(row["feature_name"], fontsize=10, fontweight="bold")
        ax.set_ylabel("Biomass (label)", fontsize=10, fontweight="bold")
        ax.set_title(f"{row['feature_name']}\nPearson r={row['pearson_r']:.4f}, p={row['pearson_p']:.4f}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        z = np.polyfit(fv, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(fv.min(), fv.max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    plt.suptitle("Feature-Biomass Correlation Analysis (Pakistani Dataset)\nTop Correlated Features", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "top_features_correlation_analysis.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved top features analysis: {out_path}")
    plt.close()
    # Top 10 detailed scatter
    n_show = min(10, len(top_pearson))
    if n_show > 0:
        nrows = 3
        ncols = 4
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
        axes = axes.flatten()
        top_n_list = top_pearson.nlargest(n_show, "pearson_abs")
        for idx, (_, row) in enumerate(top_n_list.iterrows()):
            ax = axes[idx]
            fi = int(row["feature_idx"])
            fv = X[:, fi]
            ax.scatter(fv, y, alpha=0.7, s=120, edgecolors="black", linewidths=0.5, c=fv, cmap="viridis")
            ax.set_xlabel(f"{row['feature_name']} Value", fontsize=16)
            ax.set_ylabel("Biomass (t/ha)", fontsize=16)
            sig = "***" if row["pearson_p"] < 0.001 else "**" if row["pearson_p"] < 0.01 else "*" if row["pearson_p"] < 0.05 else ""
            ax.set_title(f"{row['feature_name']}\nPearson r={row['pearson_r']:.4f}{sig}, Spearman ρ={row['spearman_r']:.4f}", fontsize=13)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, alpha=0.25, linewidth=0.5, ls=':')
            z = np.polyfit(fv, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(fv.min(), fv.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Linear fit")
            ax.legend(fontsize=12)
        for j in range(len(top_n_list), len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Top Features — Detailed Scatter Plots with Biomass (Pakistani Dataset)", fontsize=20)
        plt.tight_layout()
        out_path = os.path.join(output_dir, "top_10_features_detailed.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved top 10 features detailed: {out_path}")
        plt.close()
    # Correlation distribution histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].hist(correlations_df["pearson_r"], bins=50, alpha=0.7, edgecolor="black", linewidth=0.5)
    axes[0].axvline(x=0, color="red", linestyle="--", linewidth=2, label="r=0")
    axes[0].set_xlabel("Pearson Correlation (r)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[0].set_title("Distribution of Pearson Correlations (all features)", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[1].hist(correlations_df["spearman_r"], bins=50, alpha=0.7, edgecolor="black", linewidth=0.5, color="orange")
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2, label="ρ=0")
    axes[1].set_xlabel("Spearman Correlation (ρ)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Frequency", fontsize=12, fontweight="bold")
    axes[1].set_title("Distribution of Spearman Correlations (all features)", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "correlation_distribution.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved correlation distribution: {out_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze feature-biomass correlations (Pakistani dataset)")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV (default: ../dino_features_with_labels_and_split.csv = full features)")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory (default: current dir)")
    parser.add_argument("--top_n", type=int, default=20, help="Number of top features to visualize")
    args = parser.parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    input_csv = args.input_csv
    if input_csv is None:
        input_csv = os.path.join(base_dir, "dino_features_with_labels_and_split.csv")
    if not os.path.isabs(input_csv):
        input_csv = os.path.normpath(os.path.join(script_dir, input_csv))
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(script_dir, output_dir))
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("FEATURE-BIOMASS CORRELATION ANALYSIS (Pakistani Dataset)")
    print("=" * 60)
    print("\nLoading data...")
    df = pd.read_csv(input_csv)
    feature_cols, label_col = get_feature_and_label_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")
    print(f"Label range: {y.min():.2f} - {y.max():.2f}")
    correlations_df = compute_feature_correlations(X, y, feature_cols)
    if len(correlations_df) == 0:
        print("No features with variation found. Exiting.")
        return
    correlations_csv = os.path.join(output_dir, "feature_correlations.csv")
    correlations_df.sort_values("pearson_abs", ascending=False).to_csv(correlations_csv, index=False)
    print(f"\nSaved correlations to: {correlations_csv}")
    print("\n" + "=" * 60)
    print("TOP FEATURES BY ABSOLUTE PEARSON CORRELATION")
    print("=" * 60)
    top_n = min(args.top_n, len(correlations_df))
    top_df = correlations_df.nlargest(top_n, "pearson_abs")
    print(f"\n{'Rank':<6} {'Feature':<12} {'Pearson r':<12} {'p-value':<12} {'Spearman ρ':<12}")
    print("-" * 60)
    for rank, (_, row) in enumerate(top_df.iterrows(), 1):
        sig = "***" if row["pearson_p"] < 0.001 else "**" if row["pearson_p"] < 0.01 else "*" if row["pearson_p"] < 0.05 else ""
        print(f"{rank:<6} {row['feature_name']:<12} {row['pearson_r']:>10.4f}{sig:<2} {row['pearson_p']:>10.4f}    {row['spearman_r']:>10.4f}")
    print("\nCreating visualizations...")
    visualize_top_features(X, y, correlations_df, output_dir, top_n=args.top_n)
    summary = {
        "total_features": len(correlations_df),
        "features_with_sig_correlation_001": int(np.sum(correlations_df["pearson_p"] < 0.001)),
        "features_with_sig_correlation_01": int(np.sum(correlations_df["pearson_p"] < 0.01)),
        "features_with_sig_correlation_05": int(np.sum(correlations_df["pearson_p"] < 0.05)),
        "top_10_features": [
            {
                "feature_idx": int(row["feature_idx"]),
                "feature_name": str(row["feature_name"]),
                "pearson_r": float(row["pearson_r"]),
                "pearson_p": float(row["pearson_p"]),
                "spearman_r": float(row["spearman_r"]),
                "spearman_p": float(row["spearman_p"]),
            }
            for _, row in correlations_df.nlargest(10, "pearson_abs").iterrows()
        ],
        "pearson_correlation_stats": {
            "mean": float(correlations_df["pearson_r"].mean()),
            "std": float(correlations_df["pearson_r"].std()),
            "min": float(correlations_df["pearson_r"].min()),
            "max": float(correlations_df["pearson_r"].max()),
            "median": float(correlations_df["pearson_r"].median()),
        },
    }
    summary_path = os.path.join(output_dir, "correlation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")
    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
