#!/usr/bin/env python3
"""
Complete pipeline for cluster-based biomass prediction:
1. Group 303 plots into ~101 clusters (mean of ~3 plots per cluster)
2. Extract 224x224 image crops centered on each cluster
3. Calculate mean biomass of plots in each cluster
4. Extract DINO v2 features from crops
5. Apply PCA to preserve maximum information
6. Save complete dataset in organized folder structure
"""
import os
import json
import csv
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import timm
import torch
from torchvision import transforms
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import fiona
    from shapely.geometry import shape
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False
    print("Warning: fiona/shapely not available. Install with: pip install fiona shapely")


def load_dino_v2_small(pretrained: bool = True):
    """Load DINO v2 Small model (patch16) - uses 224x224 input."""
    model = timm.create_model('vit_small_patch16_224.dino', pretrained=pretrained, num_classes=0)
    model.eval()
    return model


def extract_features(model, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
    """Extract features using DINO v2, processing in batches."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features: List[np.ndarray] = []
    n_images = len(image_paths)
    
    print(f"Extracting DINO features from {n_images} images...")
    for i in range(0, n_images, batch_size):
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Processed {i}/{n_images} images...")
        
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            tensor = transform(img)
            batch_images.append(tensor)
        
        batch_tensor = torch.stack(batch_images).to(device)
        with torch.no_grad():
            batch_feat = model(batch_tensor)
        features.append(batch_feat.cpu().numpy())
    
    all_features = np.vstack(features)
    print(f"Extracted DINO features shape: {all_features.shape}")
    return all_features


def group_plots_into_clusters(shapefile_path: str, n_clusters: int = 101) -> Dict[int, List[Dict]]:
    """
    Group plots from shapefile into clusters.
    Returns dict: {cluster_id: [list of plot features]}
    """
    if not HAS_FIONA:
        raise ImportError("fiona and shapely required. Install with: pip install fiona shapely")
    
    plots = []
    with fiona.open(shapefile_path, 'r') as shp:
        for feature in shp:
            geom = shape(feature['geometry'])
            centroid = geom.centroid
            biomass = feature['properties'].get('biom_t_ha_', None)
            plots.append({
                'id': feature['id'],
                'geometry': feature['geometry'],
                'centroid': (centroid.x, centroid.y),
                'biomass': float(biomass) if biomass is not None else None,
                'properties': feature['properties']
            })
    
    print(f"Loaded {len(plots)} plots from shapefile")
    
    # Use spatial clustering (K-means on centroids)
    try:
        from sklearn.cluster import KMeans
        centroids = np.array([plot['centroid'] for plot in plots])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(centroids)
        
        clusters = defaultdict(list)
        for plot, label in zip(plots, cluster_labels):
            clusters[label].append(plot)
        
        # Calculate statistics
        cluster_sizes = [len(p) for p in clusters.values()]
        print(f"Grouped {len(plots)} plots into {len(clusters)} clusters")
        print(f"Plots per cluster: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
              f"mean={np.mean(cluster_sizes):.2f}, median={np.median(cluster_sizes):.1f}")
        
        return dict(clusters)
    except ImportError:
        raise ImportError("sklearn required for clustering. Install with: pip install scikit-learn")


def get_cluster_centroid(cluster_plots: List[Dict]) -> Tuple[float, float]:
    """Get centroid of a cluster (mean of plot centroids)."""
    centroids = [plot['centroid'] for plot in cluster_plots]
    mean_x = np.mean([c[0] for c in centroids])
    mean_y = np.mean([c[1] for c in centroids])
    return (mean_x, mean_y)


def get_cluster_biomass(cluster_plots: List[Dict]) -> Optional[float]:
    """Get mean biomass value for a cluster (mean of plot biomasses)."""
    biomasses = [plot['biomass'] for plot in cluster_plots if plot['biomass'] is not None]
    if len(biomasses) == 0:
        return None
    return float(np.mean(biomasses))


def extract_cluster_crop(
    raster_path: str,
    cluster_centroid: Tuple[float, float],
    crop_size: int = 224,
    output_path: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Extract a 224x224 crop centered on cluster centroid from the raster.
    Returns crop array (224, 224, 3) as uint8, or None if failed.
    """
    try:
        with rasterio.open(raster_path) as src:
            # Get row/col from coordinates
            centroid_x, centroid_y = cluster_centroid
            row, col = src.index(centroid_x, centroid_y)
            
            # Calculate crop bounds (centered on center point)
            half_size = crop_size // 2
            row_start = max(0, row - half_size)
            row_end = min(src.height, row + half_size)
            col_start = max(0, col - half_size)
            col_end = min(src.width, col + half_size)
            
            # Ensure we have exactly crop_size x crop_size
            actual_h = row_end - row_start
            actual_w = col_end - col_start
            
            if actual_h < crop_size or actual_w < crop_size:
                # Adjust to get full size
                row_start = max(0, row - half_size)
                row_end = min(src.height, row_start + crop_size)
                col_start = max(0, col - half_size)
                col_end = min(src.width, col_start + crop_size)
            
            # Read crop
            window = Window.from_slices(
                (row_start, row_end),
                (col_start, col_end)
            )
            
            crop_data = src.read(window=window)
            
            # Handle multi-band
            if crop_data.ndim == 3:
                # Take first 3 bands for RGB
                if crop_data.shape[0] >= 3:
                    crop_data = crop_data[:3, :, :]
                else:
                    # Pad if less than 3 bands
                    pad = np.zeros((3 - crop_data.shape[0], crop_data.shape[1], crop_data.shape[2]), dtype=crop_data.dtype)
                    crop_data = np.vstack([crop_data, pad])
                crop_data = np.transpose(crop_data, (1, 2, 0))  # (H, W, 3)
            else:
                # Single band - convert to RGB
                crop_data = np.stack([crop_data[0], crop_data[0], crop_data[0]], axis=-1)
            
            # Resize to exactly 224x224 if needed
            if crop_data.shape[0] != crop_size or crop_data.shape[1] != crop_size:
                img = Image.fromarray(crop_data.astype(np.uint8) if crop_data.dtype == np.uint8 else crop_data)
                img = img.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
                crop_data = np.array(img)
            
            # Normalize to 0-255 if needed
            if crop_data.max() > 255 or crop_data.dtype != np.uint8:
                if crop_data.max() > 1.0:
                    # Normalize
                    cmin, cmax = np.percentile(crop_data, [1, 99])
                    if cmax > cmin:
                        crop_data = np.clip((crop_data - cmin) * (255.0 / (cmax - cmin)), 0, 255)
                    else:
                        crop_data = np.clip(crop_data, 0, 255)
                crop_data = crop_data.astype(np.uint8)
            
            # Save if path provided
            if output_path:
                img = Image.fromarray(crop_data)
                img.save(output_path)
            
            return crop_data
            
    except Exception as e:
        print(f"Error extracting crop at {cluster_centroid}: {e}")
        return None


def apply_pca(X_train: np.ndarray, X_test: np.ndarray, variance_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Apply PCA to preserve maximum information.
    Returns (X_train_pca, X_test_pca, pca_model)
    """
    print(f"\nApplying PCA to preserve maximum information...")
    print(f"  Original feature dimension: {X_train.shape[1]}")
    
    # Fit PCA on training data
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    
    # Calculate cumulative explained variance
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Find number of components to preserve variance_threshold
    n_components = np.argmax(cumsum_var >= variance_threshold) + 1
    if n_components == 1 and cumsum_var[0] < variance_threshold:
        n_components = len(cumsum_var)
    
    print(f"  Components needed for {variance_threshold*100:.1f}% variance: {n_components}")
    print(f"  Actual variance preserved: {cumsum_var[n_components-1]*100:.2f}%")
    
    # Refit with selected components
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"  Reduced feature dimension: {X_train_pca.shape[1]}")
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")
    
    return X_train_pca, X_test_pca, pca


def make_stratified_splits(labels: List[float], seed: int = 42) -> List[str]:
    """Return split assignments (train/test) in 70/30 ratio, stratified by label."""
    rng = random.Random(seed)
    labels_arr = np.array(labels)
    
    n_bins = min(10, len(np.unique(labels_arr)))
    if n_bins > 1:
        _, bin_edges = np.histogram(labels_arr, bins=n_bins)
        binned_labels = np.digitize(labels_arr, bin_edges[1:-1])
    else:
        binned_labels = np.zeros(len(labels_arr), dtype=int)
    
    splits = [None] * len(labels)
    unique_bins = np.unique(binned_labels)
    for bin_val in unique_bins:
        idxs = np.where(binned_labels == bin_val)[0].tolist()
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(round(0.7 * n)))
        for i, idx in enumerate(idxs):
            splits[idx] = 'train' if i < n_train else 'test'
    
    return splits


def write_csv(csv_path: str, features: np.ndarray, labels: List[float], splits: List[str]) -> None:
    """Write features, labels, and splits to CSV."""
    D = features.shape[1]
    header = [f"feature_{i}" for i in range(D)] + ["label", "split"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for vec, lbl, sp in zip(features, labels, splits):
            row = list(map(str, vec.tolist())) + [str(lbl), sp]
            writer.writerow(row)


def build_complete_pipeline(
    shapefile_path: str,
    raster_path: str,
    output_dir: str = "complete_pipeline_dataset",
    n_clusters: int = 101,
    crop_size: int = 224,
    pca_variance: float = 0.95,
    train_ratio: float = 0.7,
    random_state: int = 42
) -> Dict[str, str]:
    """
    Build complete pipeline dataset.
    Returns dict with paths to saved files.
    """
    os.makedirs(output_dir, exist_ok=True)
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    print("="*60)
    print("COMPLETE PIPELINE: Cluster-Based Biomass Prediction")
    print("="*60)
    
    # Step 1: Group plots into clusters
    print("\n[Step 1/5] Grouping plots into clusters...")
    clusters = group_plots_into_clusters(shapefile_path, n_clusters=n_clusters)
    
    if len(clusters) == 0:
        raise ValueError("No clusters found!")
    
    # Step 2: Extract crops and calculate mean biomass
    print(f"\n[Step 2/5] Extracting {crop_size}x{crop_size} crops and calculating mean biomass...")
    cluster_data = []
    crop_paths = []
    labels = []
    cluster_info = []
    
    for cluster_id, cluster_plots in sorted(clusters.items()):
        # Get cluster centroid
        cluster_centroid = get_cluster_centroid(cluster_plots)
        
        # Get cluster biomass (mean of plots in cluster)
        cluster_biomass = get_cluster_biomass(cluster_plots)
        if cluster_biomass is None:
            continue
        
        # Extract crop
        crop_path = os.path.join(crops_dir, f"cluster_{cluster_id:03d}_biomass_{cluster_biomass:.2f}.png")
        crop_data = extract_cluster_crop(
            raster_path, cluster_centroid,
            crop_size=crop_size, output_path=crop_path
        )
        
        if crop_data is None:
            continue
        
        cluster_data.append({
            'cluster_id': cluster_id,
            'crop_path': crop_path,
            'biomass': cluster_biomass,
            'n_plots': len(cluster_plots),
            'plot_biomasses': [p['biomass'] for p in cluster_plots if p['biomass'] is not None]
        })
        crop_paths.append(crop_path)
        labels.append(cluster_biomass)
        cluster_info.append({
            'cluster_id': cluster_id,
            'n_plots': len(cluster_plots),
            'mean_biomass': cluster_biomass,
            'plot_biomasses': [p['biomass'] for p in cluster_plots if p['biomass'] is not None]
        })
    
    print(f"Processed {len(cluster_data)} valid clusters")
    print(f"  Biomass range: {min(labels):.2f} - {max(labels):.2f} ton/ha")
    print(f"  Biomass mean: {np.mean(labels):.2f}, std: {np.std(labels):.2f}")
    print(f"  Mean plots per cluster: {np.mean([c['n_plots'] for c in cluster_data]):.2f}")
    
    # Step 3: Extract DINO features
    print(f"\n[Step 3/5] Extracting DINO v2 features...")
    model = load_dino_v2_small(pretrained=True)
    features = extract_features(model, crop_paths, batch_size=32)
    
    print(f"  DINO features shape: {features.shape}")
    
    # Step 4: Create train/test split
    print(f"\n[Step 4/5] Creating train/test split...")
    splits = make_stratified_splits(labels, seed=random_state)
    
    # Separate train and test
    train_indices = [i for i, s in enumerate(splits) if s == 'train']
    test_indices = [i for i, s in enumerate(splits) if s == 'test']
    
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = np.array([labels[i] for i in train_indices])
    y_test = np.array([labels[i] for i in test_indices])
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Step 5: Apply PCA
    print(f"\n[Step 5/5] Applying PCA to preserve maximum information...")
    X_train_pca, X_test_pca, pca_model = apply_pca(X_train, X_test, variance_threshold=pca_variance)
    
    # Combine train and test for saving
    all_features_pca = np.vstack([X_train_pca, X_test_pca])
    all_splits = [splits[i] for i in train_indices] + [splits[i] for i in test_indices]
    all_labels_pca = np.concatenate([y_train, y_test])
    
    # Reorder to match original order
    reordered_indices = train_indices + test_indices
    original_order = sorted(range(len(reordered_indices)), key=lambda i: reordered_indices[i])
    all_features_pca = all_features_pca[original_order]
    all_labels_pca = all_labels_pca[original_order]
    all_splits = [all_splits[i] for i in original_order]
    
    # Save datasets
    print(f"\n[Saving] Writing datasets to files...")
    
    # Save raw DINO features
    raw_csv = os.path.join(output_dir, "dino_features_with_labels_and_split.csv")
    write_csv(raw_csv, features, labels, splits)
    print(f"  Raw DINO features: {raw_csv}")
    
    # Save PCA features
    pca_csv = os.path.join(output_dir, "dino_features_pca_with_labels_and_split.csv")
    write_csv(pca_csv, all_features_pca, all_labels_pca.tolist(), all_splits)
    print(f"  PCA features: {pca_csv}")
    
    # Save cluster info
    info_json = os.path.join(output_dir, "cluster_info.json")
    with open(info_json, 'w') as f:
        json.dump({
            'n_clusters': int(len(cluster_data)),
            'n_plots_total': int(sum(c['n_plots'] for c in cluster_data)),
            'mean_plots_per_cluster': float(np.mean([c['n_plots'] for c in cluster_data])),
            'biomass_stats': {
                'min': float(min(labels)),
                'max': float(max(labels)),
                'mean': float(np.mean(labels)),
                'std': float(np.std(labels))
            },
            'pca_info': {
                'n_components': int(pca_model.n_components_),
                'explained_variance_ratio': [float(x) for x in pca_model.explained_variance_ratio_.tolist()],
                'cumulative_variance': float(pca_model.explained_variance_ratio_.sum())
            },
            'clusters': [
                {
                    'cluster_id': int(c['cluster_id']),
                    'n_plots': int(c['n_plots']),
                    'mean_biomass': float(c['mean_biomass']),
                    'plot_biomasses': [float(b) for b in c['plot_biomasses']]
                }
                for c in cluster_info
            ]
        }, f, indent=2)
    print(f"  Cluster info: {info_json}")
    
    # Save PCA model
    import pickle
    pca_model_path = os.path.join(output_dir, "pca_model.pkl")
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca_model, f)
    print(f"  PCA model: {pca_model_path}")
    
    # Save summary
    summary_txt = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_txt, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPLETE PIPELINE DATASET SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total clusters: {len(cluster_data)}\n")
        f.write(f"Total plots: {sum(c['n_plots'] for c in cluster_data)}\n")
        f.write(f"Mean plots per cluster: {np.mean([c['n_plots'] for c in cluster_data]):.2f}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(f"Biomass Statistics:\n")
        f.write(f"  Range: {min(labels):.2f} - {max(labels):.2f} ton/ha\n")
        f.write(f"  Mean: {np.mean(labels):.2f} ton/ha\n")
        f.write(f"  Std: {np.std(labels):.2f} ton/ha\n\n")
        f.write(f"DINO Features:\n")
        f.write(f"  Original dimension: {features.shape[1]}\n\n")
        f.write(f"PCA Features:\n")
        f.write(f"  Reduced dimension: {X_train_pca.shape[1]}\n")
        f.write(f"  Explained variance: {pca_model.explained_variance_ratio_.sum()*100:.2f}%\n")
        f.write(f"  Variance threshold: {pca_variance*100:.1f}%\n")
    print(f"  Summary: {summary_txt}")
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Dataset saved to: {output_dir}")
    print(f"{'='*60}")
    
    return {
        'output_dir': output_dir,
        'raw_features_csv': raw_csv,
        'pca_features_csv': pca_csv,
        'cluster_info_json': info_json,
        'pca_model_pkl': pca_model_path,
        'summary_txt': summary_txt,
        'crops_dir': crops_dir
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build complete pipeline dataset")
    parser.add_argument("--shapefile", type=str, 
                       default="../german data/biomass_karlsruhe.shp",
                       help="Path to shapefile with plots")
    parser.add_argument("--raster", type=str, 
                       default="../german data/karlsruhe.tif",
                       help="Path to high-resolution raster")
    parser.add_argument("--output_dir", type=str,
                       default="complete_pipeline_dataset",
                       help="Output directory")
    parser.add_argument("--n_clusters", type=int, default=101,
                       help="Number of clusters (default: 101)")
    parser.add_argument("--crop_size", type=int, default=224,
                       help="Crop size (default: 224 for DINO)")
    parser.add_argument("--pca_variance", type=float, default=0.95,
                       help="PCA variance threshold (default: 0.95)")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                       help="Training set ratio")
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    base_dir = "/Users/assadabid/Desktop/Balakot data for Prof. Shafait/biomass"
    if args.shapefile.startswith("../"):
        shapefile_path = os.path.join(base_dir, args.shapefile.replace("../", ""))
    elif not os.path.isabs(args.shapefile):
        shapefile_path = os.path.join(base_dir, args.shapefile)
    else:
        shapefile_path = args.shapefile
    
    if args.raster.startswith("../"):
        raster_path = os.path.join(base_dir, args.raster.replace("../", ""))
    elif not os.path.isabs(args.raster):
        raster_path = os.path.join(base_dir, args.raster)
    else:
        raster_path = args.raster
    
    # Check files exist
    for path, name in [(shapefile_path, "shapefile"), (raster_path, "raster")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found: {path}")
    
    result = build_complete_pipeline(
        shapefile_path=shapefile_path,
        raster_path=raster_path,
        output_dir=args.output_dir,
        n_clusters=args.n_clusters,
        crop_size=args.crop_size,
        pca_variance=args.pca_variance,
        train_ratio=args.train_ratio
    )
    
    print(f"\nAll files saved in: {result['output_dir']}")


if __name__ == "__main__":
    main()

