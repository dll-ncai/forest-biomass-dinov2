#!/usr/bin/env python3
"""
Run cluster-separated training with multiple feature counts and create comparison table.
"""
import os
import subprocess
import pandas as pd
import json

base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(base_dir, "augmented_zero_biomass_analysis", "cluster_separated_distance_analysis")

# Feature counts from 20 to 300 in increments of 10
feature_counts = list(range(20, 301, 10))  # [20, 30, 40, ..., 300]
all_results = []

print("="*80)
print("RUNNING ANALYSIS FOR MULTIPLE FEATURE COUNTS")
print("="*80)
print(f"Feature counts to test: {feature_counts}")
print()

for n_features in feature_counts:
    print(f"\n{'='*80}")
    print(f"RUNNING WITH {n_features} FEATURES")
    print(f"{'='*80}")
    
    # Run the training script with 85-15 split and 5-fold CV
    cmd = [
        "python3",
        os.path.join(base_dir, "train_cluster_separated_top_features.py"),
        "--input_csv", "complete_pipeline_dataset/dino_features_with_labels_and_split_augmented.csv",
        "--output_dir", os.path.join("augmented_zero_biomass_analysis", "cluster_separated_distance_analysis"),
        "--n_top_features", str(n_features),
        "--test_size", "0.15",
        "--n_folds", "5"
    ]
    
    result = subprocess.run(
        cmd,
        cwd=base_dir,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"ERROR running with {n_features} features:")
        print(result.stderr)
        continue
    
    # Read the results
    results_file = os.path.join(output_dir, f'results_top{n_features}_features.json')
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Extract results for both clusters
        for cluster_id in [0, 1]:
            cluster_key = f'cluster_{cluster_id}'
            if cluster_key in results_data:
                for method in ['nnls', 'linear', 'xgb', 'mlp']:
                    if method in results_data[cluster_key]:
                        test_metrics = results_data[cluster_key][method]['test_metrics']
                        train_metrics = results_data[cluster_key][method].get('train_metrics', {})
                        
                        # Test metrics
                        all_results.append({
                            'n_features': n_features,
                            'cluster': cluster_id,
                            'method': method.upper(),
                            'split': 'test',
                            'percent_rmse': test_metrics['percent_rmse'],
                            'rmse': test_metrics['rmse'],
                            'mae': test_metrics['mae'],
                            'r2': test_metrics['r2']
                        })
                        
                        # Train metrics
                        if train_metrics:
                            all_results.append({
                                'n_features': n_features,
                                'cluster': cluster_id,
                                'method': method.upper(),
                                'split': 'train',
                                'percent_rmse': train_metrics['percent_rmse'],
                                'rmse': train_metrics['rmse'],
                                'mae': train_metrics['mae'],
                                'r2': train_metrics['r2']
                            })

# Create comparison DataFrame
df = pd.DataFrame(all_results)

# Save to CSV
output_csv = os.path.join(output_dir, 'comparison_multiple_feature_counts.csv')
df.to_csv(output_csv, index=False)
print(f"\n✅ Saved comparison to: {output_csv}")

# Print summary table
print("\n" + "="*80)
print("COMPARISON TABLE - ALL FEATURE COUNTS")
print("="*80)

for cluster_id in [0, 1]:
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*80}")
    
    cluster_data = df[df['cluster'] == cluster_id]
    
    for method in ['NNLS', 'LINEAR', 'XGB']:
        method_data = cluster_data[cluster_data['method'] == method]
        if not method_data.empty:
            print(f"\n{method}:")
            print(f"{'N Features':<12} {'%RMSE':<12} {'RMSE':<12} {'MAE':<12} {'R²':<12}")
            print("-" * 60)
            for n_feat in sorted(method_data['n_features'].unique()):
                row = method_data[method_data['n_features'] == n_feat].iloc[0]
                print(f"{int(row['n_features']):<12} {row['percent_rmse']:>10.2f}% "
                      f"{row['rmse']:>10.2f} {row['mae']:>10.2f} {row['r2']:>10.4f}")

print("\n" + "="*80)
print("BEST PERFORMANCE PER METHOD AND CLUSTER")
print("="*80)

for cluster_id in [0, 1]:
    print(f"\nCluster {cluster_id}:")
    cluster_data = df[df['cluster'] == cluster_id]
    
    for method in ['NNLS', 'LINEAR', 'XGB']:
        method_data = cluster_data[cluster_data['method'] == method]
        if not method_data.empty:
            best_idx = method_data['percent_rmse'].idxmin()
            best = method_data.loc[best_idx]
            print(f"  {method}: Best with {int(best['n_features'])} features - "
                  f"%RMSE={best['percent_rmse']:.2f}%, R²={best['r2']:.4f}")

