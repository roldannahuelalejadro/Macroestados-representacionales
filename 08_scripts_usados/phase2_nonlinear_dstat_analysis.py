from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from phase2_hidden_utils import ROOT, ensure_dir, load_npz_vectors, write_json


REGRESSION_TARGETS = ["D_stat_max", "D_exact", "D_partial"]
CLASSIFICATION_TARGETS = {
    "D_stat_nonzero": "D_stat_max",
}
SPLITS = ["random_kfold", "leave_secret_out"]
META_COLS = ["secret_id", "base_prompt_index", "transform_family", "transform_strength"]


def finite(value: float) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def regression_folds(samples: pd.DataFrame, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(samples)
    if split == "random_kfold":
        return list(KFold(n_splits=5, shuffle=True, random_state=0).split(np.zeros(n)))
    if split == "leave_secret_out":
        groups = samples["secret_id"].astype(str).to_numpy()
    elif split == "leave_base_request_out":
        groups = samples["base_prompt_index"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown split: {split}")
    return list(LeaveOneGroupOut().split(np.zeros(n), groups=groups))


def classification_folds(samples: pd.DataFrame, labels: np.ndarray, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    n = len(samples)
    labels = labels.astype(int)
    if split == "random_kfold":
        counts = pd.Series(labels).value_counts()
        n_splits = min(5, int(counts.min()))
        if n_splits < 2:
            return []
        return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(np.zeros(n), labels))
    if split == "leave_secret_out":
        groups = samples["secret_id"].astype(str).to_numpy()
    elif split == "leave_base_request_out":
        groups = samples["base_prompt_index"].astype(str).to_numpy()
    else:
        raise ValueError(f"Unknown split: {split}")
    return list(LeaveOneGroupOut().split(np.zeros(n), labels, groups))


def onehot_pipeline(estimator, interactions: bool = False):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    column = ColumnTransformer([("cat", encoder, META_COLS)], remainder="drop")
    if interactions:
        return make_pipeline(
            column,
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            estimator,
        )
    return make_pipeline(column, estimator)


def hidden_regression_models() -> dict[str, object]:
    return {
        "hidden_linear_pca_ridge": make_pipeline(
            StandardScaler(),
            PCA(n_components=16, random_state=0),
            Ridge(alpha=10.0),
        ),
        "hidden_quadratic_pca_ridge": make_pipeline(
            StandardScaler(),
            PCA(n_components=8, random_state=0),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=25.0),
        ),
    }


def hidden_classification_models() -> dict[str, object]:
    return {
        "hidden_linear_pca_logit": make_pipeline(
            StandardScaler(),
            PCA(n_components=16, random_state=0),
            LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"),
        ),
        "hidden_quadratic_pca_logit": make_pipeline(
            StandardScaler(),
            PCA(n_components=8, random_state=0),
            PolynomialFeatures(degree=2, include_bias=False),
            LogisticRegression(C=0.15, max_iter=5000, class_weight="balanced"),
        ),
    }


def predict_regression_cv(model, X, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    preds = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in folds:
        if np.nanstd(y[train_idx]) == 0:
            preds[test_idx] = float(np.nanmean(y[train_idx]))
            continue
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], y[train_idx])
        preds[test_idx] = fitted.predict(X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx])
    return preds


def predict_classification_cv(model, X, y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pred = np.full(len(y), -1, dtype=int)
    score = np.full(len(y), np.nan, dtype=float)
    for train_idx, test_idx in folds:
        if len(np.unique(y[train_idx])) < 2:
            majority = int(pd.Series(y[train_idx]).mode().iloc[0])
            pred[test_idx] = majority
            score[test_idx] = float(majority)
            continue
        fitted = clone(model)
        fitted.fit(X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx], y[train_idx])
        Xt = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
        pred[test_idx] = fitted.predict(Xt)
        if hasattr(fitted, "predict_proba"):
            proba = fitted.predict_proba(Xt)
            if proba.shape[1] == 2:
                score[test_idx] = proba[:, 1]
            else:
                score[test_idx] = pred[test_idx]
        elif hasattr(fitted, "decision_function"):
            score[test_idx] = fitted.decision_function(Xt)
        else:
            score[test_idx] = pred[test_idx]
    return pred, score


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(pred)
    if mask.sum() < 3:
        return {"n_eval": int(mask.sum()), "r2": None, "mae": None, "rmse": None}
    return {
        "n_eval": int(mask.sum()),
        "r2": finite(r2_score(y[mask], pred[mask])),
        "mae": finite(mean_absolute_error(y[mask], pred[mask])),
        "rmse": finite(mean_squared_error(y[mask], pred[mask]) ** 0.5),
    }


def classification_metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    mask = pred >= 0
    y = y[mask]
    pred = pred[mask]
    score = score[mask]
    result = {
        "n_eval": int(mask.sum()),
        "positive_rate": finite(float(np.mean(y))) if len(y) else None,
        "accuracy": finite(accuracy_score(y, pred)) if len(y) else None,
        "balanced_accuracy": finite(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else None,
        "macro_f1": finite(f1_score(y, pred, average="macro")) if len(np.unique(y)) > 1 else None,
        "roc_auc": None,
        "average_precision": None,
    }
    if len(np.unique(y)) > 1 and np.isfinite(score).all():
        result["roc_auc"] = finite(roc_auc_score(y, score))
        result["average_precision"] = finite(average_precision_score(y, score))
    return result


def source_matrix(source: str, raw_vectors: np.ndarray, delta_vectors: np.ndarray, layer_pos: int) -> np.ndarray:
    if source == "raw":
        return raw_vectors[:, layer_pos, :]
    if source == "delta":
        return delta_vectors[:, layer_pos, :]
    raise ValueError(source)


def run_regression(
    samples: pd.DataFrame,
    raw_vectors: np.ndarray,
    delta_vectors: np.ndarray,
    layers: list[int],
) -> pd.DataFrame:
    rows = []
    X_meta = samples[META_COLS].astype(str)
    meta_models = {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "metadata_main_ridge": onehot_pipeline(Ridge(alpha=10.0), interactions=False),
        "metadata_pairwise_ridge": onehot_pipeline(Ridge(alpha=25.0), interactions=True),
    }
    hidden_models = hidden_regression_models()

    for target in REGRESSION_TARGETS:
        y = samples[target].fillna(0.0).to_numpy(dtype=float)
        for split in SPLITS:
            folds = regression_folds(samples, split)
            for model_name, model in meta_models.items():
                pred = predict_regression_cv(model, X_meta, y, folds)
                rows.append(
                    {
                        "target": target,
                        "split": split,
                        "source": "metadata",
                        "layer": -1,
                        "model": model_name,
                        "model_family": "baseline"
                        if model_name == "dummy_mean"
                        else ("main_effects" if "main" in model_name else "pairwise_interactions"),
                        **regression_metrics(y, pred),
                    }
                )
            for source in ["raw", "delta"]:
                for layer_pos, layer in enumerate(layers):
                    X_layer = source_matrix(source, raw_vectors, delta_vectors, layer_pos)
                    for model_name, model in hidden_models.items():
                        pred = predict_regression_cv(model, X_layer, y, folds)
                        rows.append(
                            {
                                "target": target,
                                "split": split,
                                "source": source,
                                "layer": int(layer),
                                "model": model_name,
                                "model_family": "linear"
                                if "linear" in model_name
                                else ("quadratic" if "quadratic" in model_name else "nonlinear_tree"),
                                **regression_metrics(y, pred),
                            }
                        )
    return pd.DataFrame(rows)


def run_classification(
    samples: pd.DataFrame,
    raw_vectors: np.ndarray,
    delta_vectors: np.ndarray,
    layers: list[int],
) -> pd.DataFrame:
    rows = []
    X_meta = samples[META_COLS].astype(str)
    meta_models = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "metadata_main_logit": onehot_pipeline(
            LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"),
            interactions=False,
        ),
        "metadata_pairwise_logit": onehot_pipeline(
            LogisticRegression(C=0.15, max_iter=5000, class_weight="balanced"),
            interactions=True,
        ),
    }
    hidden_models = hidden_classification_models()

    for target_name, source_col in CLASSIFICATION_TARGETS.items():
        y = (samples[source_col].fillna(0.0).to_numpy(dtype=float) > 0).astype(int)
        if pd.Series(y).value_counts().min() < 8:
            continue
        for split in SPLITS:
            folds = classification_folds(samples, y, split)
            if not folds:
                continue
            for model_name, model in meta_models.items():
                pred, score = predict_classification_cv(model, X_meta, y, folds)
                rows.append(
                    {
                        "target": target_name,
                        "source_col": source_col,
                        "split": split,
                        "source": "metadata",
                        "layer": -1,
                        "model": model_name,
                        "model_family": "baseline"
                        if model_name == "dummy_majority"
                        else ("main_effects" if "main" in model_name else "pairwise_interactions"),
                        **classification_metrics(y, pred, score),
                    }
                )
            for source in ["raw", "delta"]:
                for layer_pos, layer in enumerate(layers):
                    X_layer = source_matrix(source, raw_vectors, delta_vectors, layer_pos)
                    for model_name, model in hidden_models.items():
                        pred, score = predict_classification_cv(model, X_layer, y, folds)
                        rows.append(
                            {
                                "target": target_name,
                                "source_col": source_col,
                                "split": split,
                                "source": source,
                                "layer": int(layer),
                                "model": model_name,
                                "model_family": "linear"
                                if "linear" in model_name
                                else ("quadratic" if "quadratic" in model_name else "nonlinear_tree"),
                                **classification_metrics(y, pred, score),
                            }
                        )
    return pd.DataFrame(rows)


def best_rows(df: pd.DataFrame, group_cols: list[str], score_col: str) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(group_cols, dropna=False):
        group = group.dropna(subset=[score_col])
        if group.empty:
            continue
        rows.append(group.sort_values(score_col, ascending=False).iloc[0].to_dict())
    return pd.DataFrame(rows)


def plot_regression_summary(best: pd.DataFrame, out_path: Path) -> None:
    data = best[(best["target"] == "D_stat_max") & (best["split"] == "random_kfold")].copy()
    if data.empty:
        return
    data["label"] = data["model_family"] + " / " + data["source"].astype(str)
    data = data.sort_values("r2", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.barh(data["label"], data["r2"].astype(float))
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("CV R2")
    ax.set_title("Best D_stat_max regression by model family, random_kfold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_classification_summary(best: pd.DataFrame, out_path: Path) -> None:
    data = best[(best["target"] == "D_stat_nonzero") & (best["split"] == "random_kfold")].copy()
    if data.empty:
        return
    data["label"] = data["model_family"] + " / " + data["source"].astype(str)
    data = data.sort_values("balanced_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.barh(data["label"], data["balanced_accuracy"].astype(float))
    ax.axvline(0.5, color="#444444", linestyle="--", linewidth=0.8)
    ax.set_xlabel("balanced accuracy")
    ax.set_title("Best D_stat_nonzero classification by model family, random_kfold")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare linear, pairwise, and nonlinear models for D_stat/e_b using existing activations."
    )
    parser.add_argument(
        "--layer-sweep-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2_layer_sweep",
    )
    parser.add_argument(
        "--delta-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2_delta_against_control",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2_nonlinear_dstat_analysis_reduced",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    samples = pd.read_csv(args.layer_sweep_dir / "samples.csv")
    raw_vectors, layers = load_npz_vectors(args.layer_sweep_dir / "hidden_vectors_by_layer.npz")
    delta_samples = pd.read_csv(args.delta_dir / "delta_samples.csv")
    delta_vectors, delta_layers = load_npz_vectors(args.delta_dir / "delta_vectors_by_layer.npz")
    if layers != delta_layers:
        raise ValueError(f"Layer mismatch: raw={layers}, delta={delta_layers}")
    if list(samples["sample_id"]) != list(delta_samples["sample_id"]):
        raise ValueError("Raw and delta samples do not align by sample_id.")

    regression = run_regression(samples, raw_vectors, delta_vectors, layers)
    classification = run_classification(samples, raw_vectors, delta_vectors, layers)
    regression.to_csv(out_dir / "regression_metrics.csv", index=False)
    classification.to_csv(out_dir / "classification_metrics.csv", index=False)

    best_reg = best_rows(regression, ["target", "split", "model_family", "source"], "r2")
    best_cls = best_rows(classification, ["target", "split", "model_family", "source"], "balanced_accuracy")
    best_reg.to_csv(out_dir / "best_regression_by_family.csv", index=False)
    best_cls.to_csv(out_dir / "best_classification_by_family.csv", index=False)

    plot_regression_summary(best_reg, out_dir / "dstat_regression_random_kfold.png")
    plot_classification_summary(best_cls, out_dir / "dstat_nonzero_classification_random_kfold.png")

    summary = {
        "n_samples": int(len(samples)),
        "layers": layers,
        "regression_targets": REGRESSION_TARGETS,
        "classification_targets": CLASSIFICATION_TARGETS,
        "splits": SPLITS,
        "model_families": {
            "baseline": "mean/majority dummy",
            "main_effects": "one-hot metadata main effects only",
            "pairwise_interactions": "one-hot metadata plus pairwise interactions",
            "linear": "PCA hidden states plus linear ridge/logistic",
            "quadratic": "PCA hidden states plus degree-2 polynomial terms",
        },
        "best_regression_rows": best_reg.to_dict(orient="records"),
        "best_classification_rows": best_cls.to_dict(orient="records"),
        "outputs": {
            "regression_metrics": str(out_dir / "regression_metrics.csv"),
            "classification_metrics": str(out_dir / "classification_metrics.csv"),
            "best_regression": str(out_dir / "best_regression_by_family.csv"),
            "best_classification": str(out_dir / "best_classification_by_family.csv"),
            "regression_plot": str(out_dir / "dstat_regression_random_kfold.png"),
            "classification_plot": str(out_dir / "dstat_nonzero_classification_random_kfold.png"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
