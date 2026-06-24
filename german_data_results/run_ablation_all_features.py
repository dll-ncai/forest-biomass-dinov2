#!/usr/bin/env python3
"""
Ablation Study 1: All Features (No Feature Selection) with Clustering
=====================================================================
Uses the same UMAP+KMeans clustering as the main pipeline but trains
models on ALL 384 DINO features instead of selecting top-N per cluster.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from scipy.optimize import nnls
from scipy.stats import pearsonr

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ── Import model helpers from existing script ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_cluster_separated_top_features import (
    compute_metrics, predict_nnls, predict_linear, predict_xgb,
    identify_umap_clusters
)

BASE_DIR = SCRIPT_DIR
INPUT_CSV = os.path.join(BASE_DIR, "complete_pipeline_dataset",
                         "dino_features_with_labels_and_split_augmented.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented_zero_biomass_analysis",
                          "ablation_all_features")


def tune_and_evaluate(X_train, y_train, X_test, y_test, cluster_name, n_folds=5):
    """Tune hyper-parameters via k-fold CV and evaluate on test set."""
    results = {}

    # ── NNLS ──
    print(f"\n  Tuning NNLS ({cluster_name})...")
    best_nnls, best_score = None, float('inf')
    k_values = [k for k in [3,4,5,6,7,8,10,12,15] if k < len(X_train)]
    for k in k_values:
        for cosine in [True, False]:
            for simplex in [True, False]:
                kf = KFold(n_splits=min(n_folds, len(X_train)), shuffle=True, random_state=42)
                scores = []
                for tr_idx, va_idx in kf.split(X_train):
                    try:
                        yp = predict_nnls(X_train[tr_idx], y_train[tr_idx],
                                          X_train[va_idx], neighbor_k=k,
                                          cosine=cosine, simplex=simplex)
                        scores.append(compute_metrics(y_train[va_idx], yp)['percent_rmse'])
                    except:
                        pass
                cv = np.mean(scores) if scores else float('inf')
                if cv < best_score:
                    best_score = cv
                    best_nnls = {'k': k, 'cosine': cosine, 'simplex': simplex,
                                 'cv_score': float(cv)}
    if best_nnls:
        yp_test = predict_nnls(X_train, y_train, X_test,
                               neighbor_k=best_nnls['k'],
                               cosine=best_nnls['cosine'],
                               simplex=best_nnls['simplex'])
        yp_train = predict_nnls(X_train, y_train, X_train,
                                neighbor_k=best_nnls['k'],
                                cosine=best_nnls['cosine'],
                                simplex=best_nnls['simplex'],
                                exclude_self=True)
        results['nnls'] = {
            'config': best_nnls,
            'test_metrics': compute_metrics(y_test, yp_test),
            'train_metrics': compute_metrics(y_train, yp_train),
            'predictions': yp_test, 'true': y_test
        }
        print(f"    Best NNLS: k={best_nnls['k']}, cosine={best_nnls['cosine']}, "
              f"simplex={best_nnls['simplex']}, CV %RMSE={best_nnls['cv_score']:.2f}%")
        print(f"    Test: %RMSE={results['nnls']['test_metrics']['percent_rmse']:.2f}%, "
              f"R²={results['nnls']['test_metrics']['r2']:.4f}")

    # ── Ridge ──
    print(f"\n  Tuning Ridge ({cluster_name})...")
    best_ridge, best_score = None, float('inf')
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]:
        for scale in [True, False]:
            kf = KFold(n_splits=min(n_folds, len(X_train)), shuffle=True, random_state=42)
            scores = []
            for tr_idx, va_idx in kf.split(X_train):
                try:
                    yp = predict_linear(X_train[tr_idx], y_train[tr_idx],
                                        X_train[va_idx], alpha=alpha, scale=scale)
                    scores.append(compute_metrics(y_train[va_idx], yp)['percent_rmse'])
                except:
                    pass
            cv = np.mean(scores) if scores else float('inf')
            if cv < best_score:
                best_score = cv
                best_ridge = {'alpha': alpha, 'scale': scale, 'cv_score': float(cv)}
    if best_ridge:
        yp_test = predict_linear(X_train, y_train, X_test,
                                 alpha=best_ridge['alpha'], scale=best_ridge['scale'])
        yp_train = predict_linear(X_train, y_train, X_train,
                                  alpha=best_ridge['alpha'], scale=best_ridge['scale'])
        results['linear'] = {
            'config': best_ridge,
            'test_metrics': compute_metrics(y_test, yp_test),
            'train_metrics': compute_metrics(y_train, yp_train),
            'predictions': yp_test, 'true': y_test
        }
        print(f"    Best Ridge: alpha={best_ridge['alpha']}, scale={best_ridge['scale']}, "
              f"CV %RMSE={best_ridge['cv_score']:.2f}%")
        print(f"    Test: %RMSE={results['linear']['test_metrics']['percent_rmse']:.2f}%, "
              f"R²={results['linear']['test_metrics']['r2']:.4f}")

    # ── XGBoost ──
    if HAS_XGB:
        print(f"\n  Tuning XGBoost ({cluster_name})...")
        import itertools
        best_xgb, best_score = None, float('inf')
        combos = list(itertools.product(
            [50, 100, 200], [0.01, 0.03, 0.05, 0.1], [3, 4, 5, 6], [0.8], [0.8], [0.1]
        ))
        combos += [(100,0.05,5,0.8,0.8,1.0), (100,0.1,6,1.0,1.0,10.0)]
        for n_est, lr, md, ss, cs, rl in combos:
            kf = KFold(n_splits=min(n_folds, len(X_train)), shuffle=True, random_state=42)
            scores = []
            for tr_idx, va_idx in kf.split(X_train):
                try:
                    yp = predict_xgb(X_train[tr_idx], y_train[tr_idx], X_train[va_idx],
                                     n_estimators=n_est, learning_rate=lr, max_depth=md,
                                     subsample=ss, colsample_bytree=cs, reg_lambda=rl)
                    scores.append(compute_metrics(y_train[va_idx], yp)['percent_rmse'])
                except:
                    pass
            cv = np.mean(scores) if scores else float('inf')
            if cv < best_score:
                best_score = cv
                best_xgb = dict(n_estimators=n_est, learning_rate=lr, max_depth=md,
                                subsample=ss, colsample_bytree=cs, reg_lambda=rl,
                                cv_score=float(cv))
        if best_xgb:
            yp_test = predict_xgb(X_train, y_train, X_test, **{k: v for k, v in best_xgb.items() if k != 'cv_score'})
            yp_train = predict_xgb(X_train, y_train, X_train, **{k: v for k, v in best_xgb.items() if k != 'cv_score'})
            results['xgb'] = {
                'config': best_xgb,
                'test_metrics': compute_metrics(y_test, yp_test),
                'train_metrics': compute_metrics(y_train, yp_train),
                'predictions': yp_test, 'true': y_test
            }
            print(f"    Best XGB: n_est={best_xgb['n_estimators']}, lr={best_xgb['learning_rate']}, "
                  f"md={best_xgb['max_depth']}, CV %RMSE={best_xgb['cv_score']:.2f}%")
            print(f"    Test: %RMSE={results['xgb']['test_metrics']['percent_rmse']:.2f}%, "
                  f"R²={results['xgb']['test_metrics']['r2']:.4f}")

    return results


def create_scatter_plots(results_c0, results_c1, output_dir):
    """Create combined scatter plots for both clusters."""
    methods = ['nnls', 'linear', 'xgb']
    labels = ['NNLS', 'Ridge', 'XGBoost']
    available = [m for m in methods if (results_c0 and m in results_c0) or (results_c1 and m in results_c1)]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(8*len(available), 7))
    if len(available) == 1:
        axes = [axes]
    fig.suptitle('Ablation: All 384 Features (With Clustering)', fontsize=16, fontweight='bold', y=1.02)

    for idx, method in enumerate(available):
        ax = axes[idx]
        label = labels[methods.index(method)]
        all_true, all_pred, cluster_ids = [], [], []
        for cid, res in [(0, results_c0), (1, results_c1)]:
            if res and method in res:
                all_true.extend(res[method]['true'].tolist())
                all_pred.extend(res[method]['predictions'].tolist())
                cluster_ids.extend([cid]*len(res[method]['true']))
        if all_true:
            at, ap, ci = np.array(all_true), np.array(all_pred), np.array(cluster_ids)
            combined = compute_metrics(at, ap)
            for cid, color, cname in [(0,'#2196F3','Cluster 0'),(1,'#FF9800','Cluster 1')]:
                mask = ci == cid
                if mask.any():
                    ax.scatter(at[mask], ap[mask], alpha=0.7, s=100, edgecolors='black',
                               linewidths=0.5, label=cname, color=color)
            mn, mx = min(at.min(), ap.min()), max(at.max(), ap.max())
            ax.plot([mn, mx], [mn, mx], 'k--', lw=2, alpha=0.5)
            ax.set_xlabel('True Biomass (ton/ha)', fontsize=12)
            ax.set_ylabel('Predicted Biomass (ton/ha)', fontsize=12)
            ax.set_title(f"{label}\nR²={combined['r2']:.3f}, %RMSE={combined['percent_rmse']:.1f}%",
                         fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_all_features_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter plot")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*70)
    print("ABLATION STUDY 1: ALL 384 FEATURES (WITH CLUSTERING)")
    print("="*70)

    # Load data
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features (ALL used)")

    # UMAP + KMeans clustering (same as main pipeline)
    X_umap, cluster_labels = identify_umap_clusters(X, n_clusters=2)

    # Process each cluster
    all_results = {}
    for cluster_id in [0, 1]:
        mask = cluster_labels == cluster_id
        X_c = X[mask]
        y_c = y[mask]
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster_id}: {len(X_c)} samples, ALL {X.shape[1]} features")
        print(f"  Biomass: {y_c.min():.1f} – {y_c.max():.1f} ton/ha (mean={y_c.mean():.1f})")
        print(f"{'='*60}")

        # Use unified split from CSV
        cluster_splits = df['split'].values[mask]
        train_mask_c = cluster_splits == 'train'
        test_mask_c = cluster_splits == 'test'
        
        X_train = X_c[train_mask_c]
        X_test = X_c[test_mask_c]
        y_train = y_c[train_mask_c]
        y_test = y_c[test_mask_c]
        
        print(f"  Train: {len(X_train)} (from CSV), Test: {len(X_test)} (from CSV)")

        results = tune_and_evaluate(X_train, y_train, X_test, y_test,
                                    f"Cluster {cluster_id}", n_folds=5)
        all_results[f'cluster_{cluster_id}'] = results

    # Combined metrics
    print("\n" + "="*70)
    print("COMBINED RESULTS (ALL 384 FEATURES)")
    print("="*70)
    combined_summary = {}
    for method in ['nnls', 'linear', 'xgb']:
        r0 = all_results.get('cluster_0', {}).get(method)
        r1 = all_results.get('cluster_1', {}).get(method)
        if r0 and r1:
            all_true = np.concatenate([r0['true'], r1['true']])
            all_pred = np.concatenate([r0['predictions'], r1['predictions']])
            combined = compute_metrics(all_true, all_pred)
            combined_summary[method] = combined
            method_name = {'nnls': 'NNLS', 'linear': 'Ridge', 'xgb': 'XGBoost'}[method]
            print(f"  {method_name}: %RMSE={combined['percent_rmse']:.2f}%, "
                  f"R²={combined['r2']:.4f}, RMSE={combined['rmse']:.2f}")

    # Create scatter plots
    create_scatter_plots(all_results.get('cluster_0'), all_results.get('cluster_1'), OUTPUT_DIR)

    # Save results (strip numpy arrays for JSON)
    save_results = {}
    for ck, cv in all_results.items():
        save_results[ck] = {}
        for mk, mv in cv.items():
            save_results[ck][mk] = {
                'config': mv['config'],
                'test_metrics': mv['test_metrics'],
                'train_metrics': mv['train_metrics'],
                'predictions': mv['predictions'].tolist(),
                'true': mv['true'].tolist()
            }
    save_results['combined'] = combined_summary
    save_results['n_features_used'] = len(feature_cols)
    save_results['ablation_type'] = 'all_features_with_clustering'

    with open(os.path.join(OUTPUT_DIR, 'ablation_all_features_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\n✅ Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
