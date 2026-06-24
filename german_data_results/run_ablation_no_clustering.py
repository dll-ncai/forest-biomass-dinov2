#!/usr/bin/env python3
"""
Ablation Study 2: No Clustering — Single Model on All Data
===========================================================
Trains a SINGLE model on ALL data (no UMAP+KMeans clustering).
Feature selection via Pearson correlation on the full training set.
Sweeps feature counts 20–300 in steps of 10.
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

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from scipy.optimize import nnls
from scipy.stats import pearsonr

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from train_cluster_separated_top_features import (
    compute_metrics, predict_nnls, predict_linear, predict_xgb,
    get_top_features_for_cluster
)

BASE_DIR = SCRIPT_DIR
INPUT_CSV = os.path.join(BASE_DIR, "complete_pipeline_dataset",
                         "dino_features_with_labels_and_split_augmented.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "augmented_zero_biomass_analysis",
                          "ablation_no_clustering")

FEATURE_COUNTS = list(range(20, 301, 10))


def tune_and_evaluate_single(X_train, y_train, X_test, y_test,
                              n_folds=5, label=""):
    """Tune all 3 models via k-fold CV and evaluate on test set."""
    results = {}

    # ── NNLS ──
    print(f"  Tuning NNLS {label}...")
    best, best_score = None, float('inf')
    k_values = [k for k in [3,4,5,6,7,8,10,12,15] if k < len(X_train)]
    for k in k_values:
        for cosine in [True, False]:
            for simplex in [True, False]:
                kf = KFold(n_splits=min(n_folds, len(X_train)),
                           shuffle=True, random_state=42)
                scores = []
                for tr_i, va_i in kf.split(X_train):
                    try:
                        yp = predict_nnls(X_train[tr_i], y_train[tr_i],
                                          X_train[va_i], neighbor_k=k,
                                          cosine=cosine, simplex=simplex)
                        scores.append(compute_metrics(y_train[va_i], yp)['percent_rmse'])
                    except:
                        pass
                cv = np.mean(scores) if scores else float('inf')
                if cv < best_score:
                    best_score = cv
                    best = dict(k=k, cosine=cosine, simplex=simplex, cv_score=float(cv))
    if best:
        yp_test = predict_nnls(X_train, y_train, X_test,
                               neighbor_k=best['k'],
                               cosine=best['cosine'],
                               simplex=best['simplex'])
        yp_train = predict_nnls(X_train, y_train, X_train,
                                neighbor_k=best['k'],
                                cosine=best['cosine'],
                                simplex=best['simplex'],
                                exclude_self=True)
        results['nnls'] = {
            'config': best,
            'test_metrics': compute_metrics(y_test, yp_test),
            'train_metrics': compute_metrics(y_train, yp_train),
            'predictions': yp_test, 'true': y_test
        }
        print(f"    -> CV %RMSE={best['cv_score']:.2f}%, "
              f"Test %RMSE={results['nnls']['test_metrics']['percent_rmse']:.2f}%")

    # ── Ridge ──
    print(f"  Tuning Ridge {label}...")
    best, best_score = None, float('inf')
    for alpha in [0.01,0.1,0.5,1,5,10,50,100,500,1000,5000]:
        for scale in [True, False]:
            kf = KFold(n_splits=min(n_folds, len(X_train)),
                       shuffle=True, random_state=42)
            scores = []
            for tr_i, va_i in kf.split(X_train):
                try:
                    yp = predict_linear(X_train[tr_i], y_train[tr_i],
                                        X_train[va_i], alpha=alpha, scale=scale)
                    scores.append(compute_metrics(y_train[va_i], yp)['percent_rmse'])
                except:
                    pass
            cv = np.mean(scores) if scores else float('inf')
            if cv < best_score:
                best_score = cv
                best = dict(alpha=alpha, scale=scale, cv_score=float(cv))
    if best:
        yp_test = predict_linear(X_train, y_train, X_test,
                                 alpha=best['alpha'], scale=best['scale'])
        yp_train = predict_linear(X_train, y_train, X_train,
                                  alpha=best['alpha'], scale=best['scale'])
        results['linear'] = {
            'config': best,
            'test_metrics': compute_metrics(y_test, yp_test),
            'train_metrics': compute_metrics(y_train, yp_train),
            'predictions': yp_test, 'true': y_test
        }
        print(f"    -> CV %RMSE={best['cv_score']:.2f}%, "
              f"Test %RMSE={results['linear']['test_metrics']['percent_rmse']:.2f}%")

    # ── XGBoost ──
    if HAS_XGB:
        import itertools
        print(f"  Tuning XGBoost {label}...")
        best, best_score = None, float('inf')
        combos = list(itertools.product(
            [50,100,200], [0.01,0.03,0.05,0.1], [3,4,5,6], [0.8], [0.8], [0.1]
        ))
        combos += [(100,0.05,5,0.8,0.8,1.0),(100,0.1,6,1.0,1.0,10.0)]
        for n_est,lr,md,ss,cs,rl in combos:
            kf = KFold(n_splits=min(n_folds, len(X_train)),
                       shuffle=True, random_state=42)
            scores = []
            for tr_i, va_i in kf.split(X_train):
                try:
                    yp = predict_xgb(X_train[tr_i], y_train[tr_i], X_train[va_i],
                                     n_estimators=n_est, learning_rate=lr,
                                     max_depth=md, subsample=ss,
                                     colsample_bytree=cs, reg_lambda=rl)
                    scores.append(compute_metrics(y_train[va_i], yp)['percent_rmse'])
                except:
                    pass
            cv = np.mean(scores) if scores else float('inf')
            if cv < best_score:
                best_score = cv
                best = dict(n_estimators=n_est, learning_rate=lr, max_depth=md,
                            subsample=ss, colsample_bytree=cs, reg_lambda=rl,
                            cv_score=float(cv))
        if best:
            cfg = {k:v for k,v in best.items() if k != 'cv_score'}
            yp_test = predict_xgb(X_train, y_train, X_test, **cfg)
            yp_train = predict_xgb(X_train, y_train, X_train, **cfg)
            results['xgb'] = {
                'config': best,
                'test_metrics': compute_metrics(y_test, yp_test),
                'train_metrics': compute_metrics(y_train, yp_train),
                'predictions': yp_test, 'true': y_test
            }
            print(f"    -> CV %RMSE={best['cv_score']:.2f}%, "
                  f"Test %RMSE={results['xgb']['test_metrics']['percent_rmse']:.2f}%")

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("="*70)
    print("ABLATION STUDY 2: NO CLUSTERING — SINGLE MODEL ON ALL DATA")
    print("="*70)

    # Load data
    df = pd.read_csv(INPUT_CSV)
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features")

    # Use unified split from CSV
    splits = df['split'].values
    train_mask = splits == 'train'
    test_mask = splits == 'test'
    
    X_train_full = X[train_mask]
    X_test_full = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"Train: {len(X_train_full)} (from CSV), Test: {len(X_test_full)} (from CSV)")

    # Feature ranking on training data only
    print("\nRanking features by Pearson correlation on training set...")
    ranked_features = get_top_features_for_cluster(X_train_full, y_train,
                                                    n_features=len(feature_cols))

    all_sweep = []
    best_per_method = {}

    for n_feat in FEATURE_COUNTS:
        top_idx = ranked_features[:n_feat]
        X_tr = X_train_full[:, top_idx]
        X_te = X_test_full[:, top_idx]

        print(f"\n{'='*60}")
        print(f"  TOP {n_feat} FEATURES")
        print(f"{'='*60}")

        results = tune_and_evaluate_single(X_tr, y_train, X_te, y_test,
                                            n_folds=5, label=f"(top {n_feat})")

        for method in ['nnls', 'linear', 'xgb']:
            if method in results:
                tm = results[method]['test_metrics']
                all_sweep.append({
                    'n_features': n_feat, 'method': method.upper(),
                    'split': 'test',
                    'percent_rmse': tm['percent_rmse'],
                    'rmse': tm['rmse'], 'mae': tm['mae'], 'r2': tm['r2']
                })
                trm = results[method]['train_metrics']
                all_sweep.append({
                    'n_features': n_feat, 'method': method.upper(),
                    'split': 'train',
                    'percent_rmse': trm['percent_rmse'],
                    'rmse': trm['rmse'], 'mae': trm['mae'], 'r2': trm['r2']
                })
                # Track best
                key = method.upper()
                if key not in best_per_method or tm['percent_rmse'] < best_per_method[key]['percent_rmse']:
                    best_per_method[key] = {**tm, 'n_features': n_feat,
                                             'config': results[method]['config']}

        # Save per-feature-count results (without arrays)
        save_res = {}
        for mk, mv in results.items():
            save_res[mk] = {
                'config': mv['config'],
                'test_metrics': mv['test_metrics'],
                'train_metrics': mv['train_metrics'],
                'predictions': mv['predictions'].tolist(),
                'true': mv['true'].tolist(),
            }
        with open(os.path.join(OUTPUT_DIR, f'results_top{n_feat}_features.json'), 'w') as f:
            json.dump(save_res, f, indent=2)

    # Save sweep CSV
    sweep_df = pd.DataFrame(all_sweep)
    sweep_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_no_clustering.csv'), index=False)

    # ── Summary plot ──
    print("\n" + "="*70)
    print("BEST RESULTS (NO CLUSTERING)")
    print("="*70)
    for method, info in best_per_method.items():
        print(f"  {method}: Best at {info['n_features']} features — "
              f"%RMSE={info['percent_rmse']:.2f}%, R²={info['r2']:.4f}")

    # Create performance-vs-features plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Ablation: No Clustering — Single Model on All Data',
                 fontsize=15, fontweight='bold')
    colors = {'NNLS': '#A23B72', 'LINEAR': '#2E86AB', 'XGB': '#F18F01'}
    names = {'NNLS': 'NNLS', 'LINEAR': 'Ridge', 'XGB': 'XGBoost'}
    test_df = sweep_df[sweep_df['split'] == 'test']
    for method in ['NNLS', 'LINEAR', 'XGB']:
        md = test_df[test_df['method'] == method].sort_values('n_features')
        if not md.empty:
            axes[0].plot(md['n_features'], md['percent_rmse'], '-o', markersize=4,
                         color=colors[method], label=names[method])
            axes[1].plot(md['n_features'], md['r2'], '-o', markersize=4,
                         color=colors[method], label=names[method])
    axes[0].set_xlabel('Number of Features'); axes[0].set_ylabel('%RMSE')
    axes[0].set_title('%RMSE vs Features'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Number of Features'); axes[1].set_ylabel('R²')
    axes[1].set_title('R² vs Features'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_no_clustering_performance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Create scatter plot for best configuration per method
    fig, axes = plt.subplots(1, len(best_per_method), figsize=(8*len(best_per_method), 7))
    if len(best_per_method) == 1:
        axes = [axes]
    fig.suptitle('Ablation: No Clustering — Best Configuration Per Model',
                 fontsize=15, fontweight='bold', y=1.02)
    for idx, (method, info) in enumerate(best_per_method.items()):
        ax = axes[idx]
        # Re-run best to get predictions
        n_feat = info['n_features']
        top_idx = ranked_features[:n_feat]
        X_tr = X_train_full[:, top_idx]
        X_te = X_test_full[:, top_idx]
        if method == 'NNLS':
            yp = predict_nnls(X_tr, y_train, X_te,
                              neighbor_k=info['config']['k'],
                              cosine=info['config']['cosine'],
                              simplex=info['config']['simplex'])
        elif method == 'LINEAR':
            yp = predict_linear(X_tr, y_train, X_te,
                                alpha=info['config']['alpha'],
                                scale=info['config']['scale'])
        elif method == 'XGB':
            cfg = {k:v for k,v in info['config'].items() if k != 'cv_score'}
            yp = predict_xgb(X_tr, y_train, X_te, **cfg)
        else:
            continue
        ax.scatter(y_test, yp, alpha=0.7, s=60, edgecolors='black', linewidths=0.5,
                   color=colors.get(method, 'gray'))
        mn, mx = min(y_test.min(), yp.min()), max(y_test.max(), yp.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=2, alpha=0.5)
        ax.set_xlabel('True Biomass (ton/ha)', fontsize=12)
        ax.set_ylabel('Predicted Biomass (ton/ha)', fontsize=12)
        ax.set_title(f"{names[method]} (top {n_feat})\n"
                     f"R²={info['r2']:.3f}, %RMSE={info['percent_rmse']:.1f}%",
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_no_clustering_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Save best summary
    best_summary = {k: {kk: vv for kk, vv in v.items()} for k, v in best_per_method.items()}
    with open(os.path.join(OUTPUT_DIR, 'best_no_clustering_summary.json'), 'w') as f:
        json.dump(best_summary, f, indent=2, default=str)

    print(f"\n✅ All results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
