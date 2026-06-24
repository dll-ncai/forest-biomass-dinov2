#!/usr/bin/env python3
"""
Visualize Pakistani dataset features using UMAP dimensionality reduction (2D and 3D).
Same methodology as German data: feature space colored by biomass and by train/test (or train/val) split.
Uses the same previously extracted features (e.g. pca_0, pca_1, ...).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not available. Install with: pip install umap-learn")


def get_feature_and_label_columns(df: pd.DataFrame):
    """Get feature columns and label column."""
    label_col = "label" if "label" in df.columns else ("biomass" if "biomass" in df.columns else None)
    if label_col is None:
        raise ValueError("CSV must have 'label' or 'biomass' column.")
    exclude = {label_col, "split"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols, label_col


def load_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
    """Load data from CSV. Returns X, y, splits, feature_cols, sample_ids."""
    df = pd.read_csv(csv_path)
    feature_cols, label_col = get_feature_and_label_columns(df)
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
    if "split" in df.columns:
        splits = df["split"].astype(str).values
    else:
        splits = np.array(["all"] * len(df))
    sample_ids = np.arange(len(df))
    return X, y, splits, feature_cols, sample_ids


def visualize_umap(
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
    sample_ids: np.ndarray,
    output_dir: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_components: int = 2,
    show_ids: bool = True,
    create_3d: bool = True,
    label_unit: str = "Biomass (label)",
):
    """Apply UMAP and create 2D and 3D visualizations."""
    if not HAS_UMAP:
        raise ImportError("umap-learn required. Install with: pip install umap-learn")
    print(f"Applying UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    print(f"  Input shape: {X.shape}")
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, len(X) - 1),
        min_dist=min_dist,
        n_components=n_components,
        random_state=42,
        metric="euclidean",
    )
    X_umap = reducer.fit_transform(X)
    print(f"  Output shape: {X_umap.shape}")
    has_split = len(np.unique(splits)) > 1
    if has_split:
        train_mask = np.array([s.lower() in ("train", "train_only") for s in splits])
        test_mask = ~train_mask
    else:
        train_mask = np.ones(len(X), dtype=bool)
        test_mask = np.zeros(len(X), dtype=bool)
    fig = plt.figure(figsize=(20, 10))
    # 1: Colored by biomass
    ax1 = plt.subplot(2, 2, 1)
    scatter1 = ax1.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="viridis", s=100, alpha=0.7, edgecolors="black", linewidths=0.5)
    if show_ids and len(X) <= 50:
        for i, (x, y_pos, sid) in enumerate(zip(X_umap[:, 0], X_umap[:, 1], sample_ids)):
            ax1.annotate(str(int(sid)), (x, y_pos), fontsize=7, ha="center", va="center", color="white", weight="bold",
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"))
    ax1.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
    ax1.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
    ax1.set_title(f"Feature Space Colored by {label_unit}", fontsize=14, fontweight="bold")
    plt.colorbar(scatter1, ax=ax1, label=label_unit)
    ax1.grid(True, alpha=0.3)
    # 2: Colored by train/test split (if available)
    ax2 = plt.subplot(2, 2, 2)
    if has_split:
        ax2.scatter(X_umap[train_mask, 0], X_umap[train_mask, 1], c="blue", label="Train", s=100, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax2.scatter(X_umap[test_mask, 0], X_umap[test_mask, 1], c="red", label="Test/Val", s=100, alpha=0.7, edgecolors="black", linewidths=0.5)
        if show_ids and len(X) <= 50:
            for i, (x, y_pos, sid) in enumerate(zip(X_umap[:, 0], X_umap[:, 1], sample_ids)):
                color = "white" if train_mask[i] else "yellow"
                ax2.annotate(str(int(sid)), (x, y_pos), fontsize=7, ha="center", va="center", color=color, weight="bold",
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6, edgecolor="none"))
        ax2.set_title("Feature Space: Train/Test Split", fontsize=14, fontweight="bold")
    else:
        ax2.scatter(X_umap[:, 0], X_umap[:, 1], c="steelblue", s=100, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax2.set_title("Feature Space (no split)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
    ax2.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    # 3: Biomass + split overlay
    ax3 = plt.subplot(2, 2, 3)
    scatter3 = ax3.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="viridis", s=100, alpha=0.6, edgecolors="black", linewidths=0.5)
    if has_split:
        ax3.scatter(X_umap[train_mask, 0], X_umap[train_mask, 1], marker="o", s=150, facecolors="none", edgecolors="blue", linewidths=2, label="Train")
        ax3.scatter(X_umap[test_mask, 0], X_umap[test_mask, 1], marker="s", s=150, facecolors="none", edgecolors="red", linewidths=2, label="Test/Val")
    plt.colorbar(scatter3, ax=ax3, label=label_unit)
    ax3.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
    ax3.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
    ax3.set_title(f"Biomass + Split" if has_split else "Feature Space", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    # 4: Biomass distribution
    ax4 = plt.subplot(2, 2, 4)
    if has_split:
        ax4.hist(y[train_mask], bins=min(15, max(3, len(np.unique(y)))), alpha=0.7, label="Train", color="blue", edgecolor="black")
        ax4.hist(y[test_mask], bins=min(15, max(3, len(np.unique(y)))), alpha=0.7, label="Test/Val", color="red", edgecolor="black")
    else:
        ax4.hist(y, bins=min(15, max(3, len(np.unique(y)))), alpha=0.7, color="steelblue", edgecolor="black")
    ax4.set_xlabel(label_unit, fontsize=12, fontweight="bold")
    ax4.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax4.set_title("Biomass Distribution", fontsize=14, fontweight="bold")
    if has_split:
        ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3, axis="y")
    plt.suptitle(f"UMAP Visualization - Pakistani Dataset\n({X.shape[0]} samples, {X.shape[1]} features → 2D)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "umap_visualization.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved UMAP visualization: {out_path}")
    plt.close()
    # Standalone 2D plots
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap="viridis", s=150, alpha=0.8, edgecolors="black", linewidths=1)
    if show_ids and len(X) <= 50:
        for i, (x, y_pos, sid) in enumerate(zip(X_umap[:, 0], X_umap[:, 1], sample_ids)):
            ax.annotate(str(int(sid)), (x, y_pos), fontsize=8, ha="center", va="center", color="white", weight="bold",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7, edgecolor="white", linewidth=0.5))
    ax.set_xlabel("UMAP Dimension 1", fontsize=14, fontweight="bold")
    ax.set_ylabel("UMAP Dimension 2", fontsize=14, fontweight="bold")
    ax.set_title(f"Feature Space (UMAP) - Pakistani Dataset\nColored by {label_unit}", fontsize=16, fontweight="bold")
    plt.colorbar(scatter, ax=ax, label=label_unit, pad=0.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_biomass.png"), dpi=200, bbox_inches="tight")
    plt.close()
    if has_split:
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(X_umap[train_mask, 0], X_umap[train_mask, 1], c="blue", label=f"Train ({np.sum(train_mask)} samples)", s=150, alpha=0.7, edgecolors="black", linewidths=1)
        ax.scatter(X_umap[test_mask, 0], X_umap[test_mask, 1], c="red", label=f"Test/Val ({np.sum(test_mask)} samples)", s=150, alpha=0.7, edgecolors="black", linewidths=1)
        ax.set_xlabel("UMAP Dimension 1", fontsize=14, fontweight="bold")
        ax.set_ylabel("UMAP Dimension 2", fontsize=14, fontweight="bold")
        ax.set_title("Feature Space (UMAP) - Train/Test Split", fontsize=16, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_train_test.png"), dpi=200, bbox_inches="tight")
        plt.close()
    # 3D UMAP
    if create_3d:
        print("Creating 3D UMAP...")
        reducer_3d = umap.UMAP(
            n_neighbors=min(n_neighbors, len(X) - 1),
            min_dist=min_dist,
            n_components=3,
            random_state=42,
            metric="euclidean",
        )
        X_umap_3d = reducer_3d.fit_transform(X)
        # 3D biomass
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=y, cmap="viridis", s=100, alpha=0.7, edgecolors="black", linewidths=0.5)
        ax.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
        ax.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
        ax.set_zlabel("UMAP Dimension 3", fontsize=12, fontweight="bold")
        ax.set_title(f"3D UMAP: Feature Space Colored by {label_unit}", fontsize=14, fontweight="bold")
        plt.colorbar(scatter, ax=ax, label=label_unit, pad=0.1, shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_3d_biomass.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved umap_3d_biomass.png")
        if has_split:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(X_umap_3d[train_mask, 0], X_umap_3d[train_mask, 1], X_umap_3d[train_mask, 2],
                      c="blue", label=f"Train ({np.sum(train_mask)})", s=100, alpha=0.7, edgecolors="black", linewidths=0.5)
            ax.scatter(X_umap_3d[test_mask, 0], X_umap_3d[test_mask, 1], X_umap_3d[test_mask, 2],
                      c="red", label=f"Test/Val ({np.sum(test_mask)})", s=100, alpha=0.7, edgecolors="black", linewidths=0.5)
            ax.set_xlabel("UMAP Dimension 1", fontsize=12, fontweight="bold")
            ax.set_ylabel("UMAP Dimension 2", fontsize=12, fontweight="bold")
            ax.set_zlabel("UMAP Dimension 3", fontsize=12, fontweight="bold")
            ax.set_title("3D UMAP: Train/Test Split", fontsize=14, fontweight="bold")
            ax.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "umap_3d_train_test.png"), dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Saved umap_3d_train_test.png")
        # Multiple 3D views
        fig = plt.figure(figsize=(20, 20))
        angles = [(30, 45), (30, 135), (60, 45), (60, 135)]
        for idx, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
            ax.scatter(X_umap_3d[:, 0], X_umap_3d[:, 1], X_umap_3d[:, 2], c=y, cmap="viridis", s=80, alpha=0.7, edgecolors="black", linewidths=0.3)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            ax.set_zlabel("Dim 3", fontsize=10)
            ax.set_title(f"View {idx+1}: elev={elev}°, azim={azim}°", fontsize=11, fontweight="bold")
        plt.suptitle("3D UMAP - Multiple Views (Pakistani Dataset)", fontsize=16, fontweight="bold", y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "umap_3d_multiple_views.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved umap_3d_multiple_views.png")
    return X_umap, reducer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="UMAP visualization for Pakistani dataset")
    parser.add_argument("--input_csv", type=str, default=None, help="Path to input CSV (default: ../dino_features_with_labels_and_split.csv = full features)")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--n_neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min_dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--no_3d", action="store_true", help="Skip 3D plots")
    parser.add_argument("--no_ids", action="store_true", help="Do not show sample IDs on plots")
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
    try:
        import umap
    except ImportError:
        print("Installing umap-learn...")
        import subprocess
        subprocess.check_call(["pip", "install", "umap-learn", "-q"])
        import umap
    print("Loading data...")
    X, y, splits, feature_cols, sample_ids = load_data(input_csv)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label range: {y.min():.2f} - {y.max():.2f}")
    if len(np.unique(splits)) > 1:
        print(f"Split: {dict(zip(*np.unique(splits, return_counts=True)))}")
    print("\nCreating UMAP visualizations (2D and 3D)...")
    visualize_umap(
        X, y, splits, sample_ids, output_dir,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        show_ids=not args.no_ids,
        create_3d=not args.no_3d,
    )
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
