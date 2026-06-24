#!/usr/bin/env python3
"""
Train three models (NNLS, Linear/Ridge, XGBoost) on two clusters separately,
using only the TOP 20 most correlated features with biomass for each cluster.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
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
except Exception as e:
    HAS_XGB = False
    print(f"DEBUG: XGBoost import failed with: {e}")

try:
    from sklearn.neural_network import MLPRegressor
    HAS_MLP = True
except Exception:
    HAS_MLP = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute RMSE, MAE, R², and %RMSE."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    percent_rmse = (rmse / np.mean(y_true)) * 100 if np.mean(y_true) > 0 else 0.0
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'percent_rmse': float(percent_rmse)
    }


def get_top_features_for_cluster(
    X_cluster: np.ndarray,
    y_cluster: np.ndarray,
    n_features: int = 20
) -> List[int]:
    """Get top N features most correlated with biomass for a cluster."""
    correlations = []
    for i in range(X_cluster.shape[1]):
        if np.std(X_cluster[:, i]) < 1e-10:
            continue
        pearson_r, _ = pearsonr(X_cluster[:, i], y_cluster)
        correlations.append((i, abs(pearson_r), pearson_r))
    
    # Sort by absolute correlation, get top N
    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [idx for idx, _, _ in correlations[:n_features]]
    return top_features


def predict_nnls(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    neighbor_k: int = 5,
    cosine: bool = True,
    simplex: bool = True,
    exclude_self: bool = True,
    self_threshold: float = 1e-6
) -> np.ndarray:
    """
    Predict using NNLS.
    
    Args:
        exclude_self: If True, exclude training samples that are too similar to the query point
                     (distance < self_threshold). This prevents self-reference issues when
                     predicting on training data.
        self_threshold: Distance threshold below which a training sample is considered
                       identical to the query point (default: 1e-6).
    """
    predictions = []
    
    for x_test in X_test:
        if cosine:
            x_test_norm = x_test / (np.linalg.norm(x_test) + 1e-10)
            X_train_norm = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-10)
            distances = 1 - np.dot(X_train_norm, x_test_norm)
        else:
            distances = np.linalg.norm(X_train - x_test, axis=1)
        
        # Exclude the query point itself (or very similar points) from neighbor search
        if exclude_self:
            # Mask out points that are too similar to the query point
            # Use a small threshold that works for both cosine and Euclidean distances
            min_dist = np.min(distances)
            if cosine:
                # For cosine distance, identical points have distance ~0
                threshold = max(self_threshold, 1e-10)
            else:
                # For Euclidean distance, identical points have distance exactly 0
                threshold = max(self_threshold, 1e-10)
            
            # Exclude points that are essentially identical (distance very close to minimum)
            valid_mask = distances > threshold
            if np.sum(valid_mask) == 0:
                # If all points are excluded (shouldn't happen), use the closest ones anyway
                valid_mask = np.ones(len(X_train), dtype=bool)
                warnings.warn(f"All training samples too similar to query point, using all samples")
            valid_indices = np.where(valid_mask)[0]
            valid_distances = distances[valid_mask]
        else:
            valid_indices = np.arange(len(X_train))
            valid_distances = distances
        
        # Ensure we have enough neighbors
        k = min(neighbor_k, len(valid_indices))
        if k == 0:
            # Fallback: use mean of y_train if no neighbors available
            predictions.append(np.mean(y_train))
            continue
        
        # Get k nearest neighbors from valid set
        nearest_local_indices = np.argsort(valid_distances)[:k]
        nearest_indices = valid_indices[nearest_local_indices]
        X_neighbors = X_train[nearest_indices]
        y_neighbors = y_train[nearest_indices]
        
        try:
            if simplex:
                weights, _ = nnls(X_neighbors.T, x_test, maxiter=50000)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
            else:
                weights, _ = nnls(X_neighbors.T, x_test, maxiter=50000)
        except RuntimeError:
            weights = np.ones(len(y_neighbors)) / len(y_neighbors)

        
        pred = np.dot(weights, y_neighbors)
        predictions.append(pred)
    
    return np.array(predictions)


def predict_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
    scale: bool = True
) -> np.ndarray:
    """Predict using Ridge Regression."""
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)


def predict_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    min_child_weight: float = 1.0
) -> np.ndarray:
    """Predict using XGBoost."""
    if not HAS_XGB:
        raise ImportError("xgboost not available")
    
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        min_child_weight=min_child_weight,
        random_state=42,
        n_jobs=1
    )
    
    model.fit(X_train, y_train)
    return model.predict(X_test)


def predict_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    hidden_layer_sizes: Union[Tuple[int, ...], Tuple[int, int]] = (4, 2),
    alpha: float = 0.01,
    learning_rate: str = 'adaptive',
    max_iter: int = 500,
    scale: bool = True
) -> np.ndarray:
    """Predict using MLP (Multi-Layer Perceptron) with architecture 4-2-1."""
    if not HAS_MLP:
        raise ImportError("sklearn.neural_network.MLPRegressor not available")
    
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # MLP with 4-2-1 architecture (4 neurons -> 2 neurons -> 1 output)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size='auto',
        learning_rate=learning_rate,
        learning_rate_init=0.001,
        max_iter=max_iter,
        shuffle=True,
        random_state=42,
        tol=1e-4,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled)


def identify_umap_clusters(X: np.ndarray, n_clusters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Apply UMAP and identify clusters. Returns (umap_embedding, cluster_labels)."""
    if not HAS_UMAP:
        raise ImportError("umap-learn required")
    
    print("Applying UMAP to identify clusters...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42,
        metric='euclidean'
    )
    X_umap = reducer.fit_transform(X)
    print(f"  UMAP shape: {X_umap.shape}")
    
    # Cluster in UMAP space
    print(f"Clustering UMAP space into {n_clusters} groups...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_umap)
    
    print(f"  Cluster sizes: {np.bincount(cluster_labels)}")
    for cluster_id in range(n_clusters):
        print(f"  Cluster {cluster_id}: {np.sum(cluster_labels == cluster_id)} samples")
    
    return X_umap, cluster_labels


def evaluate_on_cluster_with_top_features(
    X_full: np.ndarray,
    y: np.ndarray,
    cluster_mask: np.ndarray,
    cluster_name: str,
    top_features: List[int],
    splits: np.ndarray = None,
    test_size: float = 0.1,
    n_folds: int = 5,
    random_state: int = 42
):
    """Evaluate models on a specific cluster using only top features with 90-10 train/test split."""
    print(f"\n{'='*60}")
    print(f"Evaluating on {cluster_name} using TOP {len(top_features)} FEATURES")
    print(f"  Top features: {top_features[:10]}..." if len(top_features) > 10 else f"  Top features: {top_features}")
    print(f"{'='*60}")
    
    # Filter data for this cluster and select top features
    X_cluster = X_full[cluster_mask][:, top_features]
    y_cluster = y[cluster_mask]
    
    # Create train/test split
    if splits is not None:
        cluster_splits = splits[cluster_mask]
        train_mask = cluster_splits == 'train'
        test_mask = cluster_splits == 'test'
        
        X_train = X_cluster[train_mask]
        X_test = X_cluster[test_mask]
        y_train = y_cluster[train_mask]
        y_test = y_cluster[test_mask]
        
        test_size = len(X_test) / len(X_cluster) if len(X_cluster) > 0 else 0
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_cluster, y_cluster,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
    
    print(f"  Total samples: {len(X_cluster)}")
    print(f"  Train: {len(X_train)} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test: {len(X_test)} samples ({test_size*100:.0f}%)")
    print(f"  Features used: {len(top_features)}")
    print(f"  Biomass range: {y_cluster.min():.2f} - {y_cluster.max():.2f} ton/ha")
    print(f"  Biomass mean: {y_cluster.mean():.2f}, std: {y_cluster.std():.2f}")
    
    if len(X_train) < 3 or len(X_test) < 1:
        print(f"  ⚠️  Insufficient data for evaluation")
        return None
    
    results = {}
    
    # Choose CV method
    use_loo = (n_folds == 0)
    cv_method = "LOO (Leave-One-Out)" if use_loo else f"{n_folds}-Fold CV"
    print(f"\n  Hyperparameter tuning using: {cv_method}")
    
    # NNLS - Comprehensive hyperparameter tuning
    print(f"\n  Tuning NNLS with {cv_method}...")
    best_nnls = None
    best_nnls_score = float('inf')
    
    # Expanded search space
    k_values = [3, 4, 5, 6, 7, 8, 10, 12, 15]
    k_values = [k for k in k_values if k < len(X_train)]  # Don't exceed training size
    
    for k in k_values:
        for cosine in [True, False]:
            for simplex in [True, False]:
                if use_loo and len(X_train) > 1:
                    loo = LeaveOneOut()
                    scores = []
                    for train_idx, val_idx in loo.split(X_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                        if len(X_tr) < k:  # Skip if not enough neighbors
                            continue
                        try:
                            y_pred = predict_nnls(X_tr, y_tr, X_val, neighbor_k=k, cosine=cosine, simplex=simplex)
                            metrics = compute_metrics(y_val, y_pred)
                            scores.append(metrics['percent_rmse'])
                        except:
                            pass
                    cv_score = np.mean(scores) if scores else float('inf')
                else:
                    kf_n_splits = min(n_folds, len(X_train))
                    kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=42)
                    scores = []
                    for train_idx, val_idx in kf.split(X_train):
                        X_tr, X_val = X_train[train_idx], X_train[val_idx]
                        y_tr, y_val = y_train[train_idx], y_train[val_idx]
                        try:
                            y_pred = predict_nnls(X_tr, y_tr, X_val, neighbor_k=k, cosine=cosine, simplex=simplex)
                            metrics = compute_metrics(y_val, y_pred)
                            scores.append(metrics['percent_rmse'])
                        except:
                            pass
                    cv_score = np.mean(scores) if scores else float('inf')
                
                if cv_score < best_nnls_score:
                    best_nnls_score = cv_score
                    best_nnls = {'k': k, 'cosine': cosine, 'simplex': simplex, 'cv_score': cv_score}
    
    if best_nnls:
        print(f"    Best: k={best_nnls['k']}, cosine={best_nnls['cosine']}, simplex={best_nnls['simplex']}, CV %RMSE={best_nnls['cv_score']:.2f}%")
        # Test predictions
        y_pred_test_nnls = predict_nnls(X_train, y_train, X_test, 
                                        neighbor_k=best_nnls['k'], 
                                        cosine=best_nnls['cosine'], 
                                        simplex=best_nnls['simplex'])
        test_metrics_nnls = compute_metrics(y_test, y_pred_test_nnls)
        # Train predictions (exclude self to prevent perfect overfitting)
        y_pred_train_nnls = predict_nnls(X_train, y_train, X_train, 
                                         neighbor_k=best_nnls['k'], 
                                         cosine=best_nnls['cosine'], 
                                         simplex=best_nnls['simplex'],
                                         exclude_self=True)
        train_metrics_nnls = compute_metrics(y_train, y_pred_train_nnls)
        results['nnls'] = {
            'config': best_nnls,
            'test_metrics': test_metrics_nnls,
            'train_metrics': train_metrics_nnls,
            'predictions': y_pred_test_nnls,
            'true': y_test,
            'top_features': top_features
        }
        print(f"    Train: %RMSE={train_metrics_nnls['percent_rmse']:.2f}%, R²={train_metrics_nnls['r2']:.4f}")
        print(f"    Test: %RMSE={test_metrics_nnls['percent_rmse']:.2f}%, R²={test_metrics_nnls['r2']:.4f}")
    
    # Linear Regression - Comprehensive hyperparameter tuning
    print(f"\n  Tuning Linear Regression with {cv_method}...")
    best_linear = None
    best_linear_score = float('inf')
    
    # Expanded search space for regularization
    alpha_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    for alpha in alpha_values:
        for scale in [True, False]:
            if use_loo and len(X_train) > 1:
                loo = LeaveOneOut()
                scores = []
                for train_idx, val_idx in loo.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    try:
                        y_pred = predict_linear(X_tr, y_tr, X_val, alpha=alpha, scale=scale)
                        metrics = compute_metrics(y_val, y_pred)
                        scores.append(metrics['percent_rmse'])
                    except:
                        pass
                cv_score = np.mean(scores) if scores else float('inf')
            else:
                kf_n_splits = min(n_folds, len(X_train))
                kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=42)
                scores = []
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    try:
                        y_pred = predict_linear(X_tr, y_tr, X_val, alpha=alpha, scale=scale)
                        metrics = compute_metrics(y_val, y_pred)
                        scores.append(metrics['percent_rmse'])
                    except:
                        pass
                cv_score = np.mean(scores) if scores else float('inf')
            
            if cv_score < best_linear_score:
                best_linear_score = cv_score
                best_linear = {'alpha': alpha, 'scale': scale, 'cv_score': cv_score}
    
    if best_linear:
        print(f"    Best: alpha={best_linear['alpha']}, scale={best_linear['scale']}, CV %RMSE={best_linear['cv_score']:.2f}%")
        # Test predictions
        y_pred_test_linear = predict_linear(X_train, y_train, X_test,
                                           alpha=best_linear['alpha'],
                                           scale=best_linear['scale'])
        test_metrics_linear = compute_metrics(y_test, y_pred_test_linear)
        # Train predictions
        y_pred_train_linear = predict_linear(X_train, y_train, X_train,
                                            alpha=best_linear['alpha'],
                                            scale=best_linear['scale'])
        train_metrics_linear = compute_metrics(y_train, y_pred_train_linear)
        results['linear'] = {
            'config': best_linear,
            'test_metrics': test_metrics_linear,
            'train_metrics': train_metrics_linear,
            'predictions': y_pred_test_linear,
            'true': y_test,
            'top_features': top_features
        }
        print(f"    Train: %RMSE={train_metrics_linear['percent_rmse']:.2f}%, R²={train_metrics_linear['r2']:.4f}")
        print(f"    Test: %RMSE={test_metrics_linear['percent_rmse']:.2f}%, R²={test_metrics_linear['r2']:.4f}")
    
    # XGBoost - Comprehensive hyperparameter tuning
    if HAS_XGB:
        print(f"\n  Tuning XGBoost with {cv_method} (comprehensive grid search)...")
        best_xgb = None
        best_xgb_score = float('inf')
        
        # Expanded hyperparameter grid
        n_estimators_list = [50, 100, 200]
        learning_rates = [0.01, 0.03, 0.05, 0.1]
        max_depths = [3, 4, 5, 6]
        subsamples = [0.8, 1.0]
        colsample_bytrees = [0.8, 1.0]
        reg_lambdas = [0.1, 1.0, 10.0]
        
        # Grid search (sampled to avoid too many combinations)
        import itertools
        param_combinations = list(itertools.product(
            n_estimators_list,
            learning_rates,
            max_depths,
            subsamples[:1],  # Limit to first
            colsample_bytrees[:1],  # Limit to first
            reg_lambdas[:1]  # Limit to first
        ))
        
        # Add some additional promising combinations
        param_combinations.extend([
            (100, 0.01, 3, 1.0, 1.0, 1.0),
            (100, 0.05, 5, 0.8, 0.8, 1.0),
            (200, 0.03, 4, 1.0, 1.0, 0.1),
            (100, 0.1, 6, 1.0, 1.0, 10.0),
        ])
        
        print(f"    Testing {len(param_combinations)} XGBoost configurations...")
        
        for params in param_combinations:
            n_est, lr, md, ss, cs, rl = params
            if use_loo and len(X_train) > 1:
                loo = LeaveOneOut()
                scores = []
                for train_idx, val_idx in loo.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    try:
                        y_pred = predict_xgb(X_tr, y_tr, X_val, 
                                           n_estimators=n_est,
                                           learning_rate=lr,
                                           max_depth=md,
                                           subsample=ss,
                                           colsample_bytree=cs,
                                           reg_lambda=rl)
                        metrics = compute_metrics(y_val, y_pred)
                        scores.append(metrics['percent_rmse'])
                    except Exception as e:
                        pass
                cv_score = np.mean(scores) if scores else float('inf')
            else:
                kf_n_splits = min(n_folds, len(X_train))
                kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=42)
                scores = []
                for train_idx, val_idx in kf.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    try:
                        y_pred = predict_xgb(X_tr, y_tr, X_val, 
                                           n_estimators=n_est,
                                           learning_rate=lr,
                                           max_depth=md,
                                           subsample=ss,
                                           colsample_bytree=cs,
                                           reg_lambda=rl)
                        metrics = compute_metrics(y_val, y_pred)
                        scores.append(metrics['percent_rmse'])
                    except Exception as e:
                        pass
                cv_score = np.mean(scores) if scores else float('inf')
            
            if cv_score < best_xgb_score:
                best_xgb_score = cv_score
                best_xgb = {
                    'n_estimators': n_est,
                    'learning_rate': lr,
                    'max_depth': md,
                    'subsample': ss,
                    'colsample_bytree': cs,
                    'reg_lambda': rl,
                    'cv_score': cv_score
                }
        
        if best_xgb:
            print(f"    Best: n_est={best_xgb['n_estimators']}, lr={best_xgb['learning_rate']}, "
                  f"md={best_xgb['max_depth']}, subsample={best_xgb['subsample']}, "
                  f"colsample={best_xgb['colsample_bytree']}, reg_lambda={best_xgb['reg_lambda']}, "
                  f"CV %RMSE={best_xgb['cv_score']:.2f}%")
            # Test predictions
            y_pred_test_xgb = predict_xgb(X_train, y_train, X_test,
                                         n_estimators=best_xgb['n_estimators'],
                                         learning_rate=best_xgb['learning_rate'],
                                         max_depth=best_xgb['max_depth'],
                                         subsample=best_xgb['subsample'],
                                         colsample_bytree=best_xgb['colsample_bytree'],
                                         reg_lambda=best_xgb['reg_lambda'])
            test_metrics_xgb = compute_metrics(y_test, y_pred_test_xgb)
            # Train predictions
            y_pred_train_xgb = predict_xgb(X_train, y_train, X_train,
                                          n_estimators=best_xgb['n_estimators'],
                                          learning_rate=best_xgb['learning_rate'],
                                          max_depth=best_xgb['max_depth'],
                                          subsample=best_xgb['subsample'],
                                          colsample_bytree=best_xgb['colsample_bytree'],
                                          reg_lambda=best_xgb['reg_lambda'])
            train_metrics_xgb = compute_metrics(y_train, y_pred_train_xgb)
            results['xgb'] = {
                'config': best_xgb,
                'test_metrics': test_metrics_xgb,
                'train_metrics': train_metrics_xgb,
                'predictions': y_pred_test_xgb,
                'true': y_test,
                'top_features': top_features
            }
            print(f"    Train: %RMSE={train_metrics_xgb['percent_rmse']:.2f}%, R²={train_metrics_xgb['r2']:.4f}")
            print(f"    Test: %RMSE={test_metrics_xgb['percent_rmse']:.2f}%, R²={test_metrics_xgb['r2']:.4f}")
    else:
        print(f"\n  Skipping XGBoost (not available)")
    
    # MLP - Hyperparameter tuning
    if HAS_MLP:
        print(f"\n  Tuning MLP (4-2-1 architecture) with {cv_method}...")
        best_mlp = None
        best_mlp_score = float('inf')
        
        # Hyperparameter search space for MLP
        alpha_values = [0.001, 0.01, 0.1, 1.0]
        learning_rates = ['constant', 'adaptive']
        max_iters = [200, 500, 1000]
        
        # Architecture is fixed to 4-2-1 (hidden_layer_sizes=(4, 2))
        for alpha in alpha_values:
            for lr_type in learning_rates:
                for max_iter in max_iters:
                    if use_loo and len(X_train) > 1:
                        loo = LeaveOneOut()
                        scores = []
                        for train_idx, val_idx in loo.split(X_train):
                            X_tr, X_val = X_train[train_idx], X_train[val_idx]
                            y_tr, y_val = y_train[train_idx], y_train[val_idx]
                            try:
                                y_pred = predict_mlp(X_tr, y_tr, X_val, 
                                                   hidden_layer_sizes=(4, 2),
                                                   alpha=alpha,
                                                   learning_rate=lr_type,
                                                   max_iter=max_iter,
                                                   scale=True)
                                metrics = compute_metrics(y_val, y_pred)
                                scores.append(metrics['percent_rmse'])
                            except Exception as e:
                                pass
                        cv_score = np.mean(scores) if scores else float('inf')
                    else:
                        kf_n_splits = min(n_folds, len(X_train))
                        kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=42)
                        scores = []
                        for train_idx, val_idx in kf.split(X_train):
                            X_tr, X_val = X_train[train_idx], X_train[val_idx]
                            y_tr, y_val = y_train[train_idx], y_train[val_idx]
                            try:
                                y_pred = predict_mlp(X_tr, y_tr, X_val,
                                                   hidden_layer_sizes=(4, 2),
                                                   alpha=alpha,
                                                   learning_rate=lr_type,
                                                   max_iter=max_iter,
                                                   scale=True)
                                metrics = compute_metrics(y_val, y_pred)
                                scores.append(metrics['percent_rmse'])
                            except Exception as e:
                                pass
                        cv_score = np.mean(scores) if scores else float('inf')
                    
                    if cv_score < best_mlp_score:
                        best_mlp_score = cv_score
                        best_mlp = {
                            'hidden_layer_sizes': (4, 2),
                            'alpha': alpha,
                            'learning_rate': lr_type,
                            'max_iter': max_iter,
                            'scale': True,
                            'cv_score': cv_score
                        }
        
        if best_mlp:
            print(f"    Best: alpha={best_mlp['alpha']}, lr={best_mlp['learning_rate']}, "
                  f"max_iter={best_mlp['max_iter']}, CV %RMSE={best_mlp['cv_score']:.2f}%")
            # Test predictions
            y_pred_test_mlp = predict_mlp(X_train, y_train, X_test,
                                         hidden_layer_sizes=best_mlp['hidden_layer_sizes'],
                                         alpha=best_mlp['alpha'],
                                         learning_rate=best_mlp['learning_rate'],
                                         max_iter=best_mlp['max_iter'],
                                         scale=best_mlp['scale'])
            test_metrics_mlp = compute_metrics(y_test, y_pred_test_mlp)
            # Train predictions
            y_pred_train_mlp = predict_mlp(X_train, y_train, X_train,
                                          hidden_layer_sizes=best_mlp['hidden_layer_sizes'],
                                          alpha=best_mlp['alpha'],
                                          learning_rate=best_mlp['learning_rate'],
                                          max_iter=best_mlp['max_iter'],
                                          scale=best_mlp['scale'])
            train_metrics_mlp = compute_metrics(y_train, y_pred_train_mlp)
            results['mlp'] = {
                'config': best_mlp,
                'test_metrics': test_metrics_mlp,
                'train_metrics': train_metrics_mlp,
                'predictions': y_pred_test_mlp,
                'true': y_test,
                'top_features': top_features
            }
            print(f"    Train: %RMSE={train_metrics_mlp['percent_rmse']:.2f}%, R²={train_metrics_mlp['r2']:.4f}")
            print(f"    Test: %RMSE={test_metrics_mlp['percent_rmse']:.2f}%, R²={test_metrics_mlp['r2']:.4f}")
    else:
        print(f"\n  Skipping MLP (not available)")
    
    return results


def create_combined_scatter_plots(
    results_cluster_0: Dict[str, Any],
    results_cluster_1: Dict[str, Any],
    output_dir: str,
    n_features: int
):
    """Create combined scatter plots showing both clusters together for each method."""
    methods = ['nnls', 'linear', 'xgb', 'mlp']
    method_labels = ['NNLS', 'Linear/Ridge', 'XGBoost', 'MLP (4-2-1)']
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f'Combined Predictions (Both Clusters) - Using Top {n_features} Most Correlated Features', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (method, label) in enumerate(zip(methods, method_labels)):
        ax = axes[idx]
        
        # Combine data from both clusters
        all_true = []
        all_pred = []
        cluster_labels = []
        
        if method in results_cluster_0 and results_cluster_0[method]:
            r0 = results_cluster_0[method]
            all_true.extend(r0['true'].tolist())
            all_pred.extend(r0['predictions'].tolist())
            cluster_labels.extend([0] * len(r0['true']))
        
        if method in results_cluster_1 and results_cluster_1[method]:
            r1 = results_cluster_1[method]
            all_true.extend(r1['true'].tolist())
            all_pred.extend(r1['predictions'].tolist())
            cluster_labels.extend([1] * len(r1['true']))
        
        if all_true:
            all_true = np.array(all_true)
            all_pred = np.array(all_pred)
            cluster_labels = np.array(cluster_labels)
            
            # Plot with different colors for each cluster
            mask_0 = cluster_labels == 0
            mask_1 = cluster_labels == 1
            
            if np.sum(mask_0) > 0:
                ax.scatter(all_true[mask_0], all_pred[mask_0], alpha=0.7, s=100, 
                          edgecolors='black', linewidths=0.5, label='Cluster 0', color='blue')
            if np.sum(mask_1) > 0:
                ax.scatter(all_true[mask_1], all_pred[mask_1], alpha=0.7, s=100, 
                          edgecolors='black', linewidths=0.5, label='Cluster 1', color='orange')
            
            # Compute combined metrics
            combined_metrics = compute_metrics(all_true, all_pred)
            
            min_val = min(all_true.min(), all_pred.min())
            max_val = max(all_true.max(), all_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            ax.set_xlabel('True Biomass (ton/ha)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Predicted Biomass (ton/ha)', fontsize=11, fontweight='bold')
            ax.set_title(f'{label}\nR²={combined_metrics["r2"]:.4f}, RMSE={combined_metrics["rmse"]:.2f}, %RMSE={combined_metrics["percent_rmse"]:.2f}%', 
                         fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    combined_path = os.path.join(output_dir, f'combined_top{n_features}_scatter_plots.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined scatter plots: {combined_path}")
    plt.close()


def visualize_results(
    results_cluster_0: Dict[str, Any],
    results_cluster_1: Dict[str, Any],
    output_dir: str
):
    """Create scatter plots for all techniques (now including MLP)."""
    
    # Get number of features from results to use in title
    n_features_used = len(results_cluster_0.get('nnls', {}).get('top_features', [])) if results_cluster_0 and 'nnls' in results_cluster_0 else 20
    
    # Create figure for scatter plots - 2 rows x 4 columns (including MLP)
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    fig.suptitle(f'Model Predictions vs True Values - Cluster-Separated Training\nUsing Top {n_features_used} Most Correlated Features', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    methods = ['nnls', 'linear', 'xgb', 'mlp']
    method_labels = ['NNLS', 'Linear/Ridge', 'XGBoost', 'MLP (4-2-1)']
    
    for idx, (method, label) in enumerate(zip(methods, method_labels)):
        if idx >= 4:  # Safety check
            break
        ax0 = axes[0, idx]
        ax1 = axes[1, idx]
        
        # Cluster 0
        if method in results_cluster_0 and results_cluster_0[method]:
            r0 = results_cluster_0[method]
            y_true_0 = r0['true']
            y_pred_0 = r0['predictions']
            r2_0 = r0['test_metrics']['r2']
            rmse_0 = r0['test_metrics']['rmse']
            percent_rmse_0 = r0['test_metrics']['percent_rmse']
            
            ax0.scatter(y_true_0, y_pred_0, alpha=0.7, s=100, edgecolors='black', linewidths=0.5)
            min_val = min(y_true_0.min(), y_pred_0.min())
            max_val = max(y_true_0.max(), y_pred_0.max())
            ax0.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            ax0.set_xlabel('True Biomass (ton/ha)', fontsize=11, fontweight='bold')
            ax0.set_ylabel('Predicted Biomass (ton/ha)', fontsize=11, fontweight='bold')
            ax0.set_title(f'Cluster 0 - {label}\nR²={r2_0:.4f}, RMSE={rmse_0:.2f}, %RMSE={percent_rmse_0:.2f}%', 
                         fontsize=12, fontweight='bold')
            ax0.legend()
            ax0.grid(True, alpha=0.3)
        
        # Cluster 1
        if method in results_cluster_1 and results_cluster_1[method]:
            r1 = results_cluster_1[method]
            y_true_1 = r1['true']
            y_pred_1 = r1['predictions']
            r2_1 = r1['test_metrics']['r2']
            rmse_1 = r1['test_metrics']['rmse']
            percent_rmse_1 = r1['test_metrics']['percent_rmse']
            
            ax1.scatter(y_true_1, y_pred_1, alpha=0.7, s=100, edgecolors='black', linewidths=0.5, color='orange')
            min_val = min(y_true_1.min(), y_pred_1.min())
            max_val = max(y_true_1.max(), y_pred_1.max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            ax1.set_xlabel('True Biomass (ton/ha)', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Predicted Biomass (ton/ha)', fontsize=11, fontweight='bold')
            ax1.set_title(f'Cluster 1 - {label}\nR²={r2_1:.4f}, RMSE={rmse_1:.2f}, %RMSE={percent_rmse_1:.2f}%', 
                         fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Get number of features from results
    n_features = len(results_cluster_0.get('nnls', {}).get('top_features', [])) if results_cluster_0 else 20
    scatter_path = os.path.join(output_dir, f'cluster_separated_top{n_features}_scatter_plots.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved scatter plots: {scatter_path}")
    plt.close()
    
    # Create combined scatter plots (both clusters together)
    create_combined_scatter_plots(results_cluster_0, results_cluster_1, output_dir, n_features)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train models on clusters using top correlated features")
    parser.add_argument("--input_csv", type=str,
                       default="complete_pipeline_dataset/dino_features_with_labels_and_split.csv",
                       help="Path to input CSV with full features")
    parser.add_argument("--output_dir", type=str,
                       default="cluster_separated_distance_analysis",
                       help="Output directory")
    parser.add_argument("--n_top_features", type=int, default=20,
                       help="Number of top features to use per cluster")
    parser.add_argument("--test_size", type=float, default=0.15,
                       help="Test set size (default: 0.15 for 85-15 split)")
    parser.add_argument("--n_folds", type=int, default=5,
                       help="Number of folds for CV (default: 5, use 0 for LOO)")
    
    args = parser.parse_args()
    
    # Convert relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.input_csv):
        input_csv = os.path.join(base_dir, args.input_csv)
    else:
        input_csv = args.input_csv
    
    if not os.path.isabs(args.output_dir):
        output_dir = os.path.join(base_dir, args.output_dir)
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("CLUSTER-SEPARATED TRAINING WITH TOP CORRELATED FEATURES")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(input_csv)
    feature_cols = [c for c in df.columns if c.startswith('feature_')]
    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.float32)
    
    print(f"Dataset: {len(X)} samples, {len(feature_cols)} features (FULL VECTORS)")
    print(f"\nWill create {100*(1-args.test_size):.0f}-{100*args.test_size:.0f} train/test split")
    cv_method = "LOO (Leave-One-Out)" if args.n_folds == 0 else f"{args.n_folds}-Fold CV"
    print(f"Using {cv_method} for hyperparameter tuning")
    
    # Identify clusters in UMAP space
    if not HAS_UMAP:
        print("Installing umap-learn...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'umap-learn', '--quiet'])
        import umap
    
    X_umap, cluster_labels = identify_umap_clusters(X, n_clusters=2)
    
    # Get top features for each cluster
    cluster_0_mask = cluster_labels == 0
    cluster_1_mask = cluster_labels == 1
    
    X_cluster_0 = X[cluster_0_mask]
    y_cluster_0 = y[cluster_0_mask]
    X_cluster_1 = X[cluster_1_mask]
    y_cluster_1 = y[cluster_1_mask]
    
    print(f"\n{'='*60}")
    print("SELECTING TOP FEATURES FOR EACH CLUSTER")
    print(f"{'='*60}")
    
    print(f"\nCluster 0: {len(X_cluster_0)} samples")
    top_features_0 = get_top_features_for_cluster(X_cluster_0, y_cluster_0, n_features=args.n_top_features)
    print(f"  Selected top {len(top_features_0)} features: {top_features_0}")
    
    print(f"\nCluster 1: {len(X_cluster_1)} samples")
    top_features_1 = get_top_features_for_cluster(X_cluster_1, y_cluster_1, n_features=args.n_top_features)
    print(f"  Selected top {len(top_features_1)} features: {top_features_1}")
    
    # Check overlap
    overlap = set(top_features_0) & set(top_features_1)
    print(f"\n  Overlapping features: {len(overlap)} features")
    if len(overlap) > 0:
        print(f"    {sorted(list(overlap))}")
    
    # Get splits from dataframe if available
    splits = df['split'].values if 'split' in df.columns else None
    if splits is not None:
        print("Using splits provided in CSV file")

    # Evaluate on each cluster separately using top features
    results_cluster_0 = evaluate_on_cluster_with_top_features(
        X, y, cluster_0_mask, "Cluster 0", top_features_0,
        splits=splits,
        test_size=args.test_size, n_folds=args.n_folds, random_state=42
    )
    
    results_cluster_1 = evaluate_on_cluster_with_top_features(
        X, y, cluster_1_mask, "Cluster 1", top_features_1,
        splits=splits,
        test_size=args.test_size, n_folds=args.n_folds, random_state=42
    )
    
    # Visualize results
    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)
    visualize_results(results_cluster_0, results_cluster_1, output_dir)
    
    # Save results
    print("\n" + "="*60)
    print("Saving results...")
    print("="*60)
    
    # Save comparison summary
    comparison_data = []
    for cluster_idx, results in enumerate([results_cluster_0, results_cluster_1]):
        cluster_name = f'cluster_{cluster_idx}'
        top_features = top_features_0 if cluster_idx == 0 else top_features_1
        if results:
            for method in ['nnls', 'linear', 'xgb', 'mlp']:
                if method in results and results[method]:
                    metrics = results[method]['test_metrics']
                    comparison_data.append({
                        'cluster': cluster_idx,
                        'method': method,
                        'rmse': metrics['rmse'],
                        'mae': metrics['mae'],
                        'r2': metrics['r2'],
                        'percent_rmse': metrics['percent_rmse'],
                        'n_features': len(top_features)
                    })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = os.path.join(output_dir, f'comparison_top{args.n_top_features}_features.csv')
        comparison_df.to_csv(comparison_path, index=False)
        print(f"  Comparison summary: {comparison_path}")
    
    # Save detailed results (without prediction arrays)
    all_results = {
        'cluster_0': {
            method: {
                'config': results_cluster_0[method]['config'],
                'test_metrics': results_cluster_0[method]['test_metrics'],
                'train_metrics': results_cluster_0[method].get('train_metrics', {}),
                'top_features': results_cluster_0[method]['top_features'],
                'predictions': results_cluster_0[method]['predictions'].tolist() if hasattr(results_cluster_0[method]['predictions'], 'tolist') else results_cluster_0[method]['predictions'],
                'true': results_cluster_0[method]['true'].tolist() if hasattr(results_cluster_0[method]['true'], 'tolist') else results_cluster_0[method]['true']
            }
            for method in results_cluster_0 if results_cluster_0[method]
        },
        'cluster_1': {
            method: {
                'config': results_cluster_1[method]['config'],
                'test_metrics': results_cluster_1[method]['test_metrics'],
                'train_metrics': results_cluster_1[method].get('train_metrics', {}),
                'top_features': results_cluster_1[method]['top_features'],
                'predictions': results_cluster_1[method]['predictions'].tolist() if hasattr(results_cluster_1[method]['predictions'], 'tolist') else results_cluster_1[method]['predictions'],
                'true': results_cluster_1[method]['true'].tolist() if hasattr(results_cluster_1[method]['true'], 'tolist') else results_cluster_1[method]['true']
            }
            for method in results_cluster_1 if results_cluster_1[method]
        },
        'top_features': {
            'cluster_0': top_features_0,
            'cluster_1': top_features_1,
            'overlap': sorted(list(overlap))
        },
        'n_features_used': args.n_top_features
    }
    
    results_path = os.path.join(output_dir, f'results_top{args.n_top_features}_features.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Detailed results: {results_path}")
    
    print(f"\n✅ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()

