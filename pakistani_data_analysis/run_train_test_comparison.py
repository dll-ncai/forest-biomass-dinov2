#!/usr/bin/env python3
"""
Train/test comparison with hyperparameter tuning on TRAIN only.
Feature counts 20 to 300 in steps of 10. Compare methods on TEST set.
Produces comparison_multiple_feature_counts.png (train vs test curves + summary table)
to diagnose why XGBoost underperforms (overfitting: low train %RMSE, high test %RMSE).
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
from sklearn.model_selection import train_test_split, KFold
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
METHODS = ["nnls", "linear", "xgb"]
METHOD_DISPLAY = {"nnls": "NNLS", "linear": "LINEAR", "xgb": "XGB"}
METHOD_COLORS = {"nnls": "#2E86AB", "linear": "#A23B72", "xgb": "#F18F01"}
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 80% train, 20% test
TUNE_CV_FOLDS = 3  # CV on train for tuning


def get_feature_and_label_columns(df: pd.DataFrame):
    label_col = "label" if "label" in df.columns else ("biomass" if "biomass" in df.columns else None)
    if label_col is None:
        raise ValueError("CSV must have 'label' or 'biomass'.")
    exclude = {label_col, "split"}
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return feature_cols, label_col


def load_ranked_feature_indices(correlations_csv: str, feature_cols: List[str]) -> List[int]:
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
    return {"rmse": rmse, "mae": mae, "r2": r2, "percent_rmse": prmse}


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
        if simplex and np.sum(w) > 0:
            w = w / np.sum(w)
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
    model = Ridge(alpha=alpha, random_state=RANDOM_STATE) if alpha > 0 else LinearRegression()
    model.fit(X_train_s, y_train)
    return model.predict(X_eval_s).astype(np.float32)


def predict_xgb(X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray, config: Dict) -> np.ndarray:
    if not HAS_XGB:
        return np.zeros(len(X_eval), dtype=np.float32)
    model = XGBRegressor(
        n_estimators=config.get("n_estimators", 100),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 3),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        reg_lambda=config.get("reg_lambda", 1.0),
        reg_alpha=config.get("reg_alpha", 0.1),
        min_child_weight=config.get("min_child_weight", 3.0),
        random_state=RANDOM_STATE, n_jobs=1, objective="reg:squarederror", eval_metric="rmse",
    )
    model.fit(X_train, y_train)
    return model.predict(X_eval).astype(np.float32)


def tune_linear_on_train(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    best_score, best_alpha, best_scale = np.inf, 1.0, True
    kf = KFold(n_splits=min(TUNE_CV_FOLDS, len(X_train) - 1), shuffle=True, random_state=RANDOM_STATE)
    for scale in [True]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            scores = []
            for tr_idx, val_idx in kf.split(X_train):
                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                pred = predict_linear(X_tr, y_tr, X_val, scale=scale, alpha=alpha)
                m = compute_metrics(y_val, pred)
                scores.append(m["percent_rmse"])
            mean_prmse = np.mean(scores)
            if mean_prmse < best_score:
                best_score, best_alpha, best_scale = mean_prmse, alpha, scale
    return {"scale": best_scale, "alpha": best_alpha}


def tune_xgb_on_train(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    if not HAS_XGB:
        return {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}
    best_score, best_cfg = np.inf, {}
    grid = [
        {"max_depth": d, "n_estimators": n, "learning_rate": lr, "reg_lambda": reg, "min_child_weight": mw}
        for d in [2, 3, 4]
        for n in [50, 100]
        for lr in [0.03, 0.05, 0.1]
        for reg in [0.1, 1.0, 10.0]
        for mw in [1.0, 3.0]
    ]
    kf = KFold(n_splits=min(TUNE_CV_FOLDS, len(X_train) - 1), shuffle=True, random_state=RANDOM_STATE)
    for cfg in grid[: min(36, len(grid))]:  # limit combos
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            try:
                pred = predict_xgb(X_tr, y_tr, X_val, cfg)
                m = compute_metrics(y_val, pred)
                scores.append(m["percent_rmse"])
            except Exception:
                scores.append(1e9)
        mean_prmse = np.mean(scores)
        if mean_prmse < best_score:
            best_score, best_cfg = mean_prmse, cfg.copy()
    return best_cfg if best_cfg else {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "reg_lambda": 1.0, "min_child_weight": 3.0}


def tune_nnls_on_train(X_train: np.ndarray, y_train: np.ndarray) -> Dict:
    best_score, best_k = np.inf, 5
    kf = KFold(n_splits=min(TUNE_CV_FOLDS, len(X_train) - 1), shuffle=True, random_state=RANDOM_STATE)
    for k in [3, 5, 7]:
        scores = []
        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            pred = predict_nnls(X_tr, y_tr, X_val, neighbor_k=k, cosine=True, simplex=True)
            m = compute_metrics(y_val, pred)
            scores.append(m["percent_rmse"])
        mean_prmse = np.mean(scores)
        if mean_prmse < best_score:
            best_score, best_k = mean_prmse, k
    return {"neighbor_k": best_k, "cosine": True, "simplex": True}


def run_experiment(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    ranked_indices: List[int],
    output_dir: str,
) -> pd.DataFrame:
    rows = []
    feature_counts = [n for n in FEATURE_COUNTS if n <= len(ranked_indices)]
    if not feature_counts:
        feature_counts = [min(20, len(ranked_indices)), min(100, len(ranked_indices)), len(ranked_indices)]

    for n_feat in feature_counts:
        top_idx = ranked_indices[:n_feat]
        X_tr = X_train[:, top_idx]
        X_te = X_test[:, top_idx]
        print(f"  n_features={n_feat} ...")

        for method in METHODS:
            if method == "xgb" and not HAS_XGB:
                continue
            if method == "linear":
                config = tune_linear_on_train(X_tr, y_train)
                pred_train = predict_linear(X_tr, y_train, X_tr, config["scale"], config["alpha"])
                pred_test = predict_linear(X_tr, y_train, X_te, config["scale"], config["alpha"])
            elif method == "xgb":
                config = tune_xgb_on_train(X_tr, y_train)
                pred_train = predict_xgb(X_tr, y_train, X_tr, config)
                pred_test = predict_xgb(X_tr, y_train, X_te, config)
            else:
                config = tune_nnls_on_train(X_tr, y_train)
                pred_train = predict_nnls(X_tr, y_train, X_tr, config["neighbor_k"], config["cosine"], config["simplex"])
                pred_test = predict_nnls(X_tr, y_train, X_te, config["neighbor_k"], config["cosine"], config["simplex"])

            m_train = compute_metrics(y_train, pred_train)
            m_test = compute_metrics(y_test, pred_test)
            rows.append({
                "n_features": n_feat, "method": METHOD_DISPLAY[method], "split": "train",
                "percent_rmse": m_train["percent_rmse"], "rmse": m_train["rmse"], "mae": m_train["mae"], "r2": m_train["r2"],
            })
            rows.append({
                "n_features": n_feat, "method": METHOD_DISPLAY[method], "split": "test",
                "percent_rmse": m_test["percent_rmse"], "rmse": m_test["rmse"], "mae": m_test["mae"], "r2": m_test["r2"],
            })
    return pd.DataFrame(rows)


def create_comparison_figure(df: pd.DataFrame, output_path: str):
    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()
    methods = [METHOD_DISPLAY[m] for m in METHODS if METHOD_DISPLAY[m] in df_test["method"].unique()]

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, len(methods) + 1, hspace=0.35, wspace=0.3)
    fig.suptitle(
        "Model Performance Comparison: Pakistani Dataset (Train/Test Split)\n"
        "Hyperparameters tuned on train set; metrics compared on test set (train vs test shows overfitting)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    overall_best_rmse = df_test.loc[df_test["percent_rmse"].idxmin()]
    overall_best_r2 = df_test.loc[df_test["r2"].idxmax()]

    for idx, method in enumerate(methods):
        ax = fig.add_subplot(gs[0, idx])
        mt = df_test[df_test["method"] == method].sort_values("n_features")
        mr = df_train[df_train["method"] == method].sort_values("n_features")
        if mt.empty:
            ax.set_title(f"{method}\n(no data)")
            continue

        ax.plot(mt["n_features"], mt["percent_rmse"], color=METHOD_COLORS.get(method.lower(), "gray"),
                marker="o", linewidth=2.5, markersize=6, label="Test %RMSE", linestyle="-", zorder=3)
        if not mr.empty:
            ax.plot(mr["n_features"], mr["percent_rmse"], color=METHOD_COLORS.get(method.lower(), "gray"),
                    marker="s", linewidth=2, markersize=5, label="Train %RMSE", linestyle="--", alpha=0.8, zorder=2)
            merged = pd.merge(mt[["n_features", "percent_rmse"]], mr[["n_features", "percent_rmse"]],
                             on="n_features", suffixes=("_test", "_train"), how="outer").sort_values("n_features")
            ax.fill_between(merged["n_features"],
                            merged["percent_rmse_train"].fillna(mr["percent_rmse"].max()),
                            merged["percent_rmse_test"].fillna(mt["percent_rmse"].max()),
                            alpha=0.2, color=METHOD_COLORS.get(method.lower(), "gray"), zorder=1, label="Train–Test gap")

        best_test_idx = mt["percent_rmse"].idxmin()
        best_row = mt.loc[best_test_idx]
        best_r2_row = mt.loc[mt["r2"].idxmax()]
        is_best_rmse = (best_row["method"] == overall_best_rmse["method"] and best_row["n_features"] == overall_best_rmse["n_features"])
        is_best_r2 = (best_r2_row["method"] == overall_best_r2["method"] and best_r2_row["n_features"] == overall_best_r2["n_features"])

        ax.scatter([best_row["n_features"]], [best_row["percent_rmse"]],
                   color="gold" if is_best_rmse else "red", s=400 if is_best_rmse else 250, zorder=10,
                   edgecolors="black", linewidths=2, marker="*", label=f"Best test: {int(best_row['n_features'])} feat")
        ax.annotate(f"%RMSE: {best_row['percent_rmse']:.1f}%",
                    xy=(best_row["n_features"], best_row["percent_rmse"]), xytext=(10, 12), textcoords="offset points",
                    fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", edgecolor="black", alpha=0.9))

        if best_r2_row["n_features"] != best_row["n_features"]:
            ax.scatter([best_r2_row["n_features"]], [best_r2_row["percent_rmse"]],
                       color="lime" if is_best_r2 else "cyan", s=200, zorder=8, edgecolors="darkgreen", marker="D",
                       label=f"Best R²: {int(best_r2_row['n_features'])} feat")
            ax.annotate(f"R²: {best_r2_row['r2']:.3f}",
                        xy=(best_r2_row["n_features"], best_r2_row["percent_rmse"]), xytext=(10, -18), textcoords="offset points",
                        fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor="cyan", alpha=0.8))

        ax.set_xlabel("Number of features", fontsize=11, fontweight="bold")
        ax.set_ylabel("%RMSE", fontsize=11, fontweight="bold")
        train_at_best_str = ""
        if not mr.empty:
            tab = mr[mr["n_features"] == best_row["n_features"]]
            if len(tab) > 0:
                train_at_best_str = f" | Train %RMSE at best: {tab.iloc[0]['percent_rmse']:.2f}%"
        ax.set_title(f"{method}\nBest test: {int(best_row['n_features'])} feat → {best_row['percent_rmse']:.2f}%{train_at_best_str}")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    # Summary table
    ax_tab = fig.add_subplot(gs[1, :])
    ax_tab.axis("off")
    summary_data = []
    for method in methods:
        mt = df_test[df_test["method"] == method]
        mr = df_train[df_train["method"] == method]
        if mt.empty:
            continue
        best_idx = mt["percent_rmse"].idxmin()
        best = mt.loc[best_idx]
        n_feat = int(best["n_features"])
        train_at_best = mr[mr["n_features"] == n_feat]
        train_rmse = train_at_best.iloc[0]["percent_rmse"] if not train_at_best.empty else np.nan
        train_r2 = train_at_best.iloc[0]["r2"] if not train_at_best.empty else np.nan
        summary_data.append([
            method,
            str(n_feat),
            f"{best['percent_rmse']:.2f}%",
            f"{train_rmse:.2f}%" if not np.isnan(train_rmse) else "N/A",
            f"{best['r2']:.4f}",
            f"{train_r2:.4f}" if not np.isnan(train_r2) else "N/A",
            "← Overfitting" if (not np.isnan(train_rmse) and train_rmse < best["percent_rmse"] * 0.5) else "",
        ])
    cols = ["Method", "Best features", "Test %RMSE", "Train %RMSE\n(at best)", "Test R²", "Train R²\n(at best)", "Note"]
    table = ax_tab.table(cellText=summary_data, colLabels=cols, cellLoc="center", loc="center",
                         colWidths=[0.12, 0.1, 0.1, 0.12, 0.1, 0.12, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
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
        f"Best performance summary (test set) | Best: {overall_best_rmse['method']} @ {int(overall_best_rmse['n_features'])} features = {overall_best_rmse['percent_rmse']:.2f}% | "
        "Large train–test gap → overfitting (especially XGB)",
        fontsize=11, fontweight="bold", pad=12,
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train/test comparison with HP tuning on train, 20–300 features")
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--correlations_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    output_dir = args.output_dir or SCRIPT_DIR
    input_csv = args.input_csv or os.path.join(BASE_DIR, "dino_features_with_labels_and_split.csv")
    correlations_csv = args.correlations_csv or os.path.join(SCRIPT_DIR, "feature_correlations.csv")
    for p in [input_csv, correlations_csv]:
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(SCRIPT_DIR, p))
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    if not os.path.isabs(input_csv):
        input_csv = os.path.normpath(os.path.join(SCRIPT_DIR, input_csv))
    if not os.path.isabs(correlations_csv):
        correlations_csv = os.path.normpath(os.path.join(SCRIPT_DIR, correlations_csv))
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(SCRIPT_DIR, output_dir))
    os.makedirs(output_dir, exist_ok=True)

    global TEST_SIZE
    TEST_SIZE = args.test_size

    print("=" * 60)
    print("TRAIN/TEST COMPARISON (tune on train, compare on test)")
    print("=" * 60)
    df = pd.read_csv(input_csv)
    feature_cols, label_col = get_feature_and_label_columns(df)
    X_full = df[feature_cols].values.astype(np.float32)
    y_full = df[label_col].values.astype(np.float32)

    if "split" in df.columns and df["split"].nunique() > 1:
        train_mask = df["split"].astype(str).str.lower().isin(["train", "train_only"])
        test_mask = ~train_mask
        X_train = X_full[train_mask]
        y_train = y_full[train_mask]
        X_test = X_full[test_mask]
        y_test = y_full[test_mask]
        print(f"Using 'split' column: train={len(y_train)}, test={len(y_test)}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        print(f"Random 80-20 split: train={len(y_train)}, test={len(y_test)}")

    ranked_indices = load_ranked_feature_indices(correlations_csv, feature_cols)
    print(f"Feature counts: {[n for n in FEATURE_COUNTS if n <= len(ranked_indices)][:5]}...{FEATURE_COUNTS[-1]}")
    print("Running experiment (tune on train, evaluate train + test)...")
    results_df = run_experiment(X_train, y_train, X_test, y_test, ranked_indices, output_dir)

    csv_path = os.path.join(output_dir, "comparison_multiple_feature_counts.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    create_comparison_figure(results_df, os.path.join(output_dir, "comparison_multiple_feature_counts.png"))
    print("Done. Check train vs test curves: large gap → overfitting (e.g. XGB).")


if __name__ == "__main__":
    main()
