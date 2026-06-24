#!/usr/bin/env python3
"""
Find best models using top-ranked features (20 to 300 in steps of 10), same as German methodology.
Uses LOO validation. All three models (NNLS, Ridge, XGBoost) are hyperparameter-tuned via LOO
(best config per feature count). Writes comparison_multiple_feature_counts.png (LOO curves +
summary table) and other outputs.
"""
import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from scipy.optimize import nnls
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
FEATURE_COUNTS = list(range(20, 301, 10))  # 20, 30, ..., 300
METHODS = ["linear", "xgb", "nnls"]
METHOD_DISPLAY = {"linear": "Linear Regression", "xgb": "XGBoost", "nnls": "NNLS"}
METHOD_COLORS = {"linear": "#2E86AB", "xgb": "#F18F01", "nnls": "#A23B72"}


def get_feature_and_label_columns(df: pd.DataFrame):
    label_col = "label" if "label" in df.columns else ("biomass" if "biomass" in df.columns else None)
    if label_col is None:
        raise ValueError("CSV must have 'label' or 'biomass'.")
    exclude = {label_col, "split"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols, label_col


def load_ranked_feature_indices(correlations_csv: str, feature_cols: List[str]) -> List[int]:
    """Load ranking from feature_correlations.csv (already sorted by pearson_abs desc). Return list of column indices."""
    df = pd.read_csv(correlations_csv)
    if "feature_idx" in df.columns:
        return df["feature_idx"].astype(int).tolist()
    name_to_idx = {c: i for i, c in enumerate(feature_cols)}
    if "feature_name" in df.columns:
        return [name_to_idx[str(n)] for n in df["feature_name"] if str(n) in name_to_idx]
    return list(range(len(feature_cols)))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    mean_y = float(np.mean(y_true)) if len(y_true) else 0.0
    prmse = (rmse / max(abs(mean_y), 1e-12)) * 100.0
    return {"rmse": rmse, "%rmse": prmse, "mae": mae, "r2": r2}


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def predict_nnls(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray,
                 neighbor_k: int, cosine: bool, simplex: bool) -> np.ndarray:
    if cosine:
        X_train_norm = l2_normalize_rows(X_train)
        X_eval_norm = l2_normalize_rows(X_eval)
    else:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_norm = scaler.fit_transform(X_train)
        X_eval_norm = scaler.transform(X_eval)
    predictions = []
    for i in range(X_eval_norm.shape[0]):
        x = X_eval_norm[i]
        if cosine:
            sim = X_train_norm @ x
            idx = np.argsort(-sim)[: min(neighbor_k, len(X_train_norm))]
        else:
            dists = np.linalg.norm(X_train_norm - x[None, :], axis=1)
            idx = np.argsort(dists)[: min(neighbor_k, len(X_train_norm))]
        A = X_train_norm[idx].T
        w, _ = nnls(A, x)
        if simplex:
            s = float(np.sum(w))
            if s > 0:
                w = w / s
        w_full = np.zeros(len(X_train_norm), dtype=np.float64)
        w_full[idx] = w
        predictions.append(float(np.dot(w_full, y_train)))
    return np.array(predictions, dtype=np.float32)


def predict_linear(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray,
                   scale: bool, alpha: float) -> np.ndarray:
    if scale:
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_train_s = scaler.fit_transform(X_train)
        X_eval_s = scaler.transform(X_eval)
    else:
        X_train_s, X_eval_s = X_train, X_eval
    model = Ridge(alpha=alpha) if alpha > 0 else LinearRegression()
    model.fit(X_train_s, y_train)
    return model.predict(X_eval_s).astype(np.float32)


def predict_xgb(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, config: Dict) -> np.ndarray:
    if not HAS_XGB:
        return np.zeros(len(X_eval), dtype=np.float32)
    model = XGBRegressor(
        n_estimators=config.get("n_estimators", 200),
        learning_rate=config.get("learning_rate", 0.03),
        max_depth=config.get("max_depth", 5),
        subsample=config.get("subsample", 1.0),
        colsample_bytree=config.get("colsample_bytree", 0.9),
        reg_lambda=config.get("reg_lambda", 0.1),
        reg_alpha=config.get("reg_alpha", 0.0),
        min_child_weight=config.get("min_child_weight", 1.0),
        random_state=42, n_jobs=1, objective="reg:squarederror", eval_metric="rmse",
    )
    model.fit(X_train, y_train)
    return model.predict(X_eval).astype(np.float32)


def loo_evaluate(X: np.ndarray, y: np.ndarray, method: str, config: Dict) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    n = X.shape[0]
    preds, truths = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_tr, y_tr = X[mask], y[mask]
        X_te, y_te = X[~mask], y[~mask]
        if method == "nnls":
            p = predict_nnls(X_tr, y_tr, X_te, config.get("neighbor_k", 3),
                             config.get("cosine", True), config.get("simplex", True))
        elif method == "linear":
            p = predict_linear(X_tr, y_tr, X_te, config.get("scale", True), config.get("alpha", 10.0))
        elif method == "xgb":
            p = predict_xgb(X_tr, y_tr, X_te, config)
        else:
            raise ValueError(method)
        preds.append(float(p[0]))
        truths.append(float(y_te[0]))
    preds = np.array(preds)
    truths = np.array(truths)
    metrics = compute_metrics(truths, preds)
    return metrics, truths, preds


def tune_nnls_loo(X: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict[str, float], np.ndarray, np.ndarray]:
    """Find best NNLS config by LOO %RMSE over a grid (same grid as German dataset).
    Returns (best_config, metrics, y_true, y_pred)."""
    grid = [
        {"neighbor_k": k, "cosine": cos, "simplex": simp}
        for k in [3, 5, 8, 15]
        for cos in [True, False]
        for simp in [True]
    ]
    # 4 * 2 = 8 configs
    best_config = None
    best_metrics = None
    best_preds = best_truths = None
    best_prmse = np.inf
    for cfg in grid:
        try:
            metrics, truths, preds = loo_evaluate(X, y, "nnls", cfg)
            if metrics["%rmse"] < best_prmse:
                best_prmse = metrics["%rmse"]
                best_config = cfg.copy()
                best_metrics = metrics
                best_preds = preds
                best_truths = truths
        except Exception:
            continue
    if best_config is None:
        best_config = {"neighbor_k": 3, "cosine": True, "simplex": True}
        best_metrics, best_truths, best_preds = loo_evaluate(X, y, "nnls", best_config)
    return best_config, best_metrics, best_truths, best_preds


def tune_ridge_loo(X: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict[str, float], np.ndarray, np.ndarray]:
    """Find best Ridge config by LOO %RMSE over a grid (same grid as German dataset).
    Returns (best_config, metrics, y_true, y_pred)."""
    grid = [
        {"scale": sc, "alpha": a}
        for a in [0.1, 1.0, 10.0, 100.0, 1000.0]
        for sc in [True, False]
    ]
    # 5 * 2 = 10 configs
    best_config = None
    best_metrics = None
    best_preds = best_truths = None
    best_prmse = np.inf
    for cfg in grid:
        try:
            metrics, truths, preds = loo_evaluate(X, y, "linear", cfg)
            if metrics["%rmse"] < best_prmse:
                best_prmse = metrics["%rmse"]
                best_config = cfg.copy()
                best_metrics = metrics
                best_preds = preds
                best_truths = truths
        except Exception:
            continue
    if best_config is None:
        best_config = {"scale": True, "alpha": 10.0}
        best_metrics, best_truths, best_preds = loo_evaluate(X, y, "linear", best_config)
    return best_config, best_metrics, best_truths, best_preds


def tune_xgb_loo(X: np.ndarray, y: np.ndarray) -> Tuple[Dict, Dict[str, float], np.ndarray, np.ndarray]:
    """Find best XGB config by LOO %RMSE over a small grid. Returns (best_config, metrics, y_true, y_pred)."""
    if not HAS_XGB:
        return {}, {"rmse": 0, "%rmse": 1e9, "mae": 0, "r2": 0}, np.array([]), np.array([])
    grid = [
        {"n_estimators": n, "learning_rate": lr, "max_depth": d, "reg_lambda": reg, "min_child_weight": 2.0,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1}
        for lr in [0.01, 0.05]
        for d in [2, 3]
        for reg in [1.0, 5.0]
        for n in [30, 60]
    ]
    # 2*2*2*2 = 16 configs. Very manageable.
    best_config = None
    best_metrics = None
    best_preds = best_truths = None
    best_prmse = np.inf
    for cfg in grid:
        try:
            metrics, truths, preds = loo_evaluate(X, y, "xgb", cfg)
            if metrics["%rmse"] < best_prmse:
                best_prmse = metrics["%rmse"]
                best_config = cfg.copy()
                best_metrics = metrics
                best_preds = preds
                best_truths = truths
        except Exception:
            continue
    if best_config is None:
        best_config = {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 3, "reg_lambda": 1.0, "min_child_weight": 3.0}
        best_metrics, best_truths, best_preds = loo_evaluate(X, y, "xgb", best_config)
    return best_config, best_metrics, best_truths, best_preds


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Find best models by top-ranked features (20-250, step 10)")
    parser.add_argument("--input_csv", type=str, default=None, help="Full features CSV")
    parser.add_argument("--correlations_csv", type=str, default=None, help="feature_correlations.csv from correlation analysis")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: script dir)")
    args = parser.parse_args()
    output_dir = args.output_dir or SCRIPT_DIR
    input_csv = args.input_csv or os.path.join(BASE_DIR, "dino_features_with_labels_and_split.csv")
    correlations_csv = args.correlations_csv or os.path.join(SCRIPT_DIR, "feature_correlations.csv")
    if not os.path.isabs(input_csv):
        input_csv = os.path.normpath(os.path.join(SCRIPT_DIR, input_csv))
    if not os.path.isabs(correlations_csv):
        correlations_csv = os.path.normpath(os.path.join(SCRIPT_DIR, correlations_csv))
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(SCRIPT_DIR, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("BEST MODELS BY TOP-RANKED FEATURES (Pakistani Dataset)")
    print("=" * 60)
    print("\nLoading data...")
    df = pd.read_csv(input_csv)
    feature_cols, label_col = get_feature_and_label_columns(df)
    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df[label_col].values.astype(np.float32)
    n_total, n_features_total = X_full.shape
    print(f"  Samples: {n_total}, Full features: {n_features_total}")

    print("\nLoading feature ranking...")
    ranked_indices = load_ranked_feature_indices(correlations_csv, feature_cols)
    print(f"  Ranked {len(ranked_indices)} features")

    feature_counts = [n for n in FEATURE_COUNTS if n <= len(ranked_indices)]
    if not feature_counts:
        feature_counts = [min(20, len(ranked_indices)), min(50, len(ranked_indices)), len(ranked_indices)]
    print(f"\nFeature counts to evaluate: {feature_counts}")
    print("All models: hyperparameter tuning via LOO per feature count.")

    all_results = {}
    rows = []
    best_configs_per_feat = {}  # n_feat -> {method: best_config}

    for n_feat in feature_counts:
        top_idx = ranked_indices[:n_feat]
        X = X_full[:, top_idx]
        y = y_full.copy()
        print(f"\n  Top {n_feat} features ...")
        all_results[n_feat] = {}
        best_configs_per_feat[n_feat] = {}
        for method in METHODS:
            if method == "xgb" and not HAS_XGB:
                continue
            if method == "xgb":
                best_cfg, metrics, y_true, y_pred = tune_xgb_loo(X, y)
            elif method == "nnls":
                best_cfg, metrics, y_true, y_pred = tune_nnls_loo(X, y)
            elif method == "linear":
                best_cfg, metrics, y_true, y_pred = tune_ridge_loo(X, y)
            else:
                raise ValueError(method)
            best_configs_per_feat[n_feat][method] = best_cfg
            all_results[n_feat][method] = {
                "rmse": metrics["rmse"], "%rmse": metrics["%rmse"],
                "mae": metrics["mae"], "r2": metrics["r2"],
                "y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
                "best_config": best_cfg,
            }
            rows.append({
                "n_features": n_feat, "method": METHOD_DISPLAY[method],
                "rmse": metrics["rmse"], "percent_rmse": metrics["%rmse"],
                "mae": metrics["mae"], "r2": metrics["r2"],
            })
            print(f"    {METHOD_DISPLAY[method]}: %RMSE={metrics['%rmse']:.2f}, R²={metrics['r2']:.4f} (tuned: {best_cfg})")

    # Save JSON (without long y_true/y_pred for readability; optional full file)
    results_short = {str(k): {m: {mk: mv for mk, mv in v.items() if mk not in ("y_true", "y_pred")} for m, v in val.items()} for k, val in all_results.items()}
    with open(os.path.join(output_dir, "results_top_features.json"), "w") as f:
        json.dump(results_short, f, indent=2)
    with open(os.path.join(output_dir, "results_top_features_full.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "results_top_features.csv"), index=False)
    print(f"\nSaved results to {output_dir}")

    # Best config per method (lowest %RMSE); include tuned hyperparams for all methods
    best_configs = []
    for method in METHODS:
        sub = results_df[results_df["method"] == METHOD_DISPLAY[method]]
        if len(sub) == 0:
            continue
        best_row = sub.loc[sub["percent_rmse"].idxmin()]
        n_best = int(best_row["n_features"])
        rec = {
            "method": best_row["method"],
            "n_features": n_best,
            "percent_rmse": float(best_row["percent_rmse"]),
            "r2": float(best_row["r2"]),
        }
        if n_best in best_configs_per_feat and method in best_configs_per_feat[n_best]:
            rec["hyperparams"] = best_configs_per_feat[n_best][method]
        best_configs.append(rec)
    best_df = pd.DataFrame([{k: v for k, v in c.items() if k != "xgb_hyperparams"} for c in best_configs])
    best_df.to_csv(os.path.join(output_dir, "best_configurations.csv"), index=False)
    with open(os.path.join(output_dir, "best_configurations.json"), "w") as f:
        json.dump(best_configs, f, indent=2)

    # ----- Visualizations -----
    print("\nCreating visualizations...")

    # 1) Performance vs n_features (line plot: %RMSE and R2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for method in METHODS:
        sub = results_df[results_df["method"] == METHOD_DISPLAY[method]]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("n_features")
        axes[0].plot(sub["n_features"], sub["percent_rmse"], marker="o", label=METHOD_DISPLAY[method],
                    color=METHOD_COLORS.get(method, "gray"), linewidth=2, markersize=5)
        axes[1].plot(sub["n_features"], sub["r2"], marker="o", label=METHOD_DISPLAY[method],
                     color=METHOD_COLORS.get(method, "gray"), linewidth=2, markersize=5)
    axes[0].set_xlabel("Number of top features", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("% RMSE", fontsize=12, fontweight="bold")
    axes[0].set_title("Leave-One-Out % RMSE vs Top Features", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Number of top features", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("R²", fontsize=12, fontweight="bold")
    axes[1].set_title("Leave-One-Out R² vs Top Features", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Pakistani Dataset: Model Performance vs Number of Top-Ranked Features", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_vs_n_features.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved performance_vs_n_features.png")

    # 2) Heatmap: n_features x method, color = %RMSE
    pivot_prmse = results_df.pivot(index="n_features", columns="method", values="percent_rmse")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot_prmse.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot_prmse.columns)))
    ax.set_xticklabels(pivot_prmse.columns)
    ax.set_yticks(range(len(pivot_prmse.index)))
    ax.set_yticklabels(pivot_prmse.index)
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of top features", fontsize=12, fontweight="bold")
    ax.set_title("Leave-One-Out % RMSE Heatmap\n(Pakistani Dataset)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="% RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_percent_rmse.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved heatmap_percent_rmse.png")

    # 3) Heatmap R2
    pivot_r2 = results_df.pivot(index="n_features", columns="method", values="r2")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pivot_r2.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot_r2.columns)))
    ax.set_xticklabels(pivot_r2.columns)
    ax.set_yticks(range(len(pivot_r2.index)))
    ax.set_yticklabels(pivot_r2.index)
    ax.set_xlabel("Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of top features", fontsize=12, fontweight="bold")
    ax.set_title("Leave-One-Out R² Heatmap\n(Pakistani Dataset)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="R²")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap_r2.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved heatmap_r2.png")

    # 4) Best configurations table (figure)
    if len(best_df) > 0:
        fig, ax = plt.subplots(figsize=(10, max(3, len(best_df) * 0.5)))
        ax.axis("off")
        table = ax.table(
            cellText=best_df.values,
            colLabels=best_df.columns,
            loc="center",
            cellLoc="center",
            colColours=["#e0e0e0"] * len(best_df.columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        plt.title("Best Configuration per Method (lowest % RMSE)\nPakistani Dataset", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "best_configurations_table.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved best_configurations_table.png")

    # 5) comparison_multiple_feature_counts.png (LOO version: one row, 3 methods, LOO %RMSE curve + summary table)
    methods_plot = [METHOD_DISPLAY[m] for m in METHODS if METHOD_DISPLAY[m] in results_df["method"].unique()]
    display_to_key = {"Linear Regression": "linear", "XGBoost": "xgb", "NNLS": "nnls"}
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, len(methods_plot), hspace=0.35, wspace=0.3)
    fig.suptitle(
        "Model Performance Comparison: Pakistani Dataset (Leave-One-Out)\n"
        "XGBoost: best config per feature count from LOO hyperparameter tuning",
        fontsize=14, fontweight="bold", y=0.98,
    )
    overall_best = results_df.loc[results_df["percent_rmse"].idxmin()]
    for idx, method in enumerate(methods_plot):
        ax = fig.add_subplot(gs[0, idx])
        sub = results_df[results_df["method"] == method].sort_values("n_features")
        if sub.empty:
            ax.set_title(f"{method}\n(no data)")
            continue
        color = METHOD_COLORS.get(display_to_key.get(method, ""), "gray")
        ax.plot(sub["n_features"], sub["percent_rmse"], color=color,
                marker="o", linewidth=2.5, markersize=6, label="LOO %RMSE")
        best_idx = sub["percent_rmse"].idxmin()
        best_row = sub.loc[best_idx]
        is_overall = best_row["method"] == overall_best["method"] and best_row["n_features"] == overall_best["n_features"]
        ax.scatter([best_row["n_features"]], [best_row["percent_rmse"]],
                   color="gold" if is_overall else "red", s=400 if is_overall else 250, zorder=10,
                   edgecolors="black", linewidths=2, marker="*", label=f"Best: {int(best_row['n_features'])} feat")
        ax.annotate(f"%RMSE: {best_row['percent_rmse']:.1f}%\nR²: {best_row['r2']:.3f}",
                    xy=(best_row["n_features"], best_row["percent_rmse"]), xytext=(10, 12), textcoords="offset points",
                    fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", edgecolor="black", alpha=0.9))
        ax.set_xlabel("Number of features", fontsize=11, fontweight="bold")
        ax.set_ylabel("LOO % RMSE", fontsize=11, fontweight="bold")
        ax.set_title(f"{method}\nBest: {int(best_row['n_features'])} features → {best_row['percent_rmse']:.2f}%")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")
    summary_data = []
    for method in methods_plot:
        sub = results_df[results_df["method"] == method]
        if sub.empty:
            continue
        best_row = sub.loc[sub["percent_rmse"].idxmin()]
        summary_data.append([
            method,
            int(best_row["n_features"]),
            f"{best_row['percent_rmse']:.2f}%",
            f"{best_row['r2']:.4f}",
        ])
    cols = ["Method", "Best n_features", "LOO % RMSE", "LOO R²"]
    table = ax_tab.table(cellText=summary_data, colLabels=cols, cellLoc="center", loc="center",
                         colWidths=[0.2, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    for i in range(len(cols)):
        table[(0, i)].set_facecolor("#4A90E2")
        table[(0, i)].set_text_props(weight="bold", color="white")
    best_method = summary_data[np.argmin([float(r[2].rstrip("%")) for r in summary_data])][0]
    for row in range(1, len(summary_data) + 1):
        if summary_data[row - 1][0] == best_method:
            table[(row, 2)].set_facecolor("#FFD700")
            table[(row, 2)].set_text_props(weight="bold")
    ax_tab.set_title(
        f"Best performance summary (Leave-One-Out) | Best: {overall_best['method']} @ {int(overall_best['n_features'])} features = {overall_best['percent_rmse']:.2f}%",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.savefig(os.path.join(output_dir, "comparison_multiple_feature_counts.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved comparison_multiple_feature_counts.png (LOO)")

    # Save LOO comparison CSV for compatibility
    results_df.to_csv(os.path.join(output_dir, "comparison_multiple_feature_counts.csv"), index=False)
    print("  Saved comparison_multiple_feature_counts.csv (LOO)")

    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS (lowest % RMSE)")
    print("=" * 60)
    print(best_df.to_string(index=False))
    print("\n✅ Done. All outputs in:", output_dir)


if __name__ == "__main__":
    main()
