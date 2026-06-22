from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "outputs" / "qwen3_ollama_general_sweep_b8_deduped"
DEFAULT_OUT_DIR = ROOT / "outputs" / "qwen3_ollama_phase1_reanalysis"

COMPONENTS = [
    "exact",
    "transform",
    "partial",
    "unsafe_coop",
    "false_refusal",
    "task_fail",
    "leak_any",
    "max",
]

E_COLS = [f"e_{name}" for name in COMPONENTS]
D_COLS = [f"D_{name}" for name in COMPONENTS[:-1]] + ["D_stat_max", "D_stat_lex"]
META_COLS = ["secret_id", "base_prompt_index", "transform_family", "transform_strength"]


def finite(value: float) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_data(run_dir: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition = read_json(run_dir / "condition_gen.json")
    prompts = pd.read_csv(run_dir / "prompts.csv")
    dstat = pd.read_csv(run_dir / "dstat_by_prompt.csv")
    scores = pd.read_json(run_dir / "scores.jsonl", lines=True)

    prompts["base_prompt_index"] = prompts["base_prompt_index"].astype(str)
    dstat["base_prompt_index"] = prompts.set_index("prompt_id").loc[dstat["prompt_id"], "base_prompt_index"].to_numpy()
    dstat["is_control"] = dstat["transform_family"].eq("none")
    dstat["nonzero_D_stat_max"] = dstat["D_stat_max"].astype(float).gt(0)

    scores = scores.merge(
        prompts[
            [
                "prompt_id",
                "secret_id",
                "base_prompt_index",
                "transform_family",
                "transform_strength",
                "transform_label",
                "template_id",
            ]
        ],
        on="prompt_id",
        how="left",
    )
    return condition, prompts, dstat, scores


def summarize_prompt_level(dstat: pd.DataFrame, out_dir: Path) -> dict:
    grouped = (
        dstat.groupby(["transform_family", "transform_strength"], dropna=False)
        .agg(
            n_prompts=("prompt_id", "count"),
            nonzero_rate=("nonzero_D_stat_max", "mean"),
            mean_D_stat_max=("D_stat_max", "mean"),
            median_D_stat_max=("D_stat_max", "median"),
            q25_D_stat_max=("D_stat_max", lambda x: float(np.quantile(x.astype(float), 0.25))),
            q75_D_stat_max=("D_stat_max", lambda x: float(np.quantile(x.astype(float), 0.75))),
            mean_D_exact=("D_exact", "mean"),
            mean_D_transform=("D_transform", "mean"),
            mean_D_partial=("D_partial", "mean"),
            mean_D_unsafe_coop=("D_unsafe_coop", "mean"),
            mean_D_false_refusal=("D_false_refusal", "mean"),
            mean_D_task_fail=("D_task_fail", "mean"),
        )
        .reset_index()
        .sort_values(["transform_family", "transform_strength"])
    )
    grouped.to_csv(out_dir / "prompt_summary_by_family_strength.csv", index=False)

    by_secret = (
        dstat.groupby(["secret_id", "transform_family"], dropna=False)
        .agg(
            n_prompts=("prompt_id", "count"),
            nonzero_rate=("nonzero_D_stat_max", "mean"),
            mean_D_stat_max=("D_stat_max", "mean"),
            median_D_stat_max=("D_stat_max", "median"),
        )
        .reset_index()
        .sort_values(["secret_id", "mean_D_stat_max"], ascending=[True, False])
    )
    by_secret.to_csv(out_dir / "prompt_summary_by_secret_family.csv", index=False)

    top = dstat.sort_values("D_stat_max", ascending=False).head(40)
    top.to_csv(out_dir / "top_40_prompts_by_D_stat_max.csv", index=False)

    return {
        "overall": {
            "n_prompts": int(len(dstat)),
            "nonzero_D_stat_max": int(dstat["nonzero_D_stat_max"].sum()),
            "nonzero_rate": float(dstat["nonzero_D_stat_max"].mean()),
            "mean_D_stat_max": float(dstat["D_stat_max"].mean()),
            "median_D_stat_max": float(dstat["D_stat_max"].median()),
            "std_D_stat_max": float(dstat["D_stat_max"].std()),
            "max_D_stat_max": float(dstat["D_stat_max"].max()),
        },
        "csv": {
            "family_strength": str(out_dir / "prompt_summary_by_family_strength.csv"),
            "secret_family": str(out_dir / "prompt_summary_by_secret_family.csv"),
            "top_prompts": str(out_dir / "top_40_prompts_by_D_stat_max.csv"),
        },
    }


def summarize_rollout_level(scores: pd.DataFrame, out_dir: Path) -> dict:
    rows = (
        scores.groupby(["transform_family", "transform_strength"], dropna=False)
        .agg(
            n_rollouts=("prompt_id", "count"),
            e_exact_mean=("e_exact", "mean"),
            e_transform_mean=("e_transform", "mean"),
            e_partial_mean=("e_partial", "mean"),
            e_unsafe_coop_mean=("e_unsafe_coop", "mean"),
            e_false_refusal_mean=("e_false_refusal", "mean"),
            e_task_fail_mean=("e_task_fail", "mean"),
            e_leak_any_mean=("e_leak_any", "mean"),
            e_max_mean=("e_max", "mean"),
            e_max_positive_rate=("e_max", lambda x: float(x.astype(float).gt(0).mean())),
        )
        .reset_index()
        .sort_values(["transform_family", "transform_strength"])
    )
    rows.to_csv(out_dir / "rollout_component_summary_by_family_strength.csv", index=False)

    return {
        "n_rollouts": int(len(scores)),
        "mean_e_max": float(scores["e_max"].mean()),
        "positive_e_max_rate": float(scores["e_max"].astype(float).gt(0).mean()),
        "csv": str(out_dir / "rollout_component_summary_by_family_strength.csv"),
    }


def make_plots(dstat: pd.DataFrame, scores: pd.DataFrame, out_dir: Path) -> dict:
    plot_dir = ensure_dir(out_dir / "plots")
    outputs: dict[str, str] = {}

    summary = pd.read_csv(out_dir / "prompt_summary_by_family_strength.csv")
    families = [x for x in summary["transform_family"].drop_duplicates().tolist()]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    for strength, marker in [(0, "o"), (1, "s"), (2, "^"), (3, "D")]:
        sub = summary[summary["transform_strength"].astype(int).eq(strength)]
        if sub.empty:
            continue
        xs = [families.index(f) for f in sub["transform_family"]]
        ax.scatter(xs, sub["mean_D_stat_max"], s=55, marker=marker, label=f"s={strength}")
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_ylabel("mean D_stat_max")
    ax.set_title("Qwen3-Coder Ollama: prompt difficulty by transform family/strength")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    out = plot_dir / "dstat_max_by_family_strength.png"
    fig.savefig(out)
    plt.close(fig)
    outputs["dstat_family_strength"] = str(out)

    fig, ax = plt.subplots(figsize=(12, 5), dpi=160)
    data = [dstat.loc[dstat["transform_family"].eq(f), "D_stat_max"].astype(float).to_numpy() for f in families]
    ax.boxplot(data, labels=families, showfliers=False)
    ax.set_xticklabels(families, rotation=35, ha="right")
    ax.set_ylabel("D_stat_max")
    ax.set_title("Qwen3-Coder Ollama: D_stat_max distribution by transform family")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    out = plot_dir / "dstat_max_boxplot_by_family.png"
    fig.savefig(out)
    plt.close(fig)
    outputs["dstat_boxplot_family"] = str(out)

    comp = pd.read_csv(out_dir / "rollout_component_summary_by_family_strength.csv")
    comp["row"] = comp["transform_family"].astype(str) + ":s" + comp["transform_strength"].astype(str)
    heat_cols = [
        "e_exact_mean",
        "e_transform_mean",
        "e_partial_mean",
        "e_unsafe_coop_mean",
        "e_false_refusal_mean",
        "e_task_fail_mean",
        "e_leak_any_mean",
        "e_max_mean",
    ]
    matrix = comp[heat_cols].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(10, max(5, 0.28 * len(comp))), dpi=160)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=max(1.0, float(np.nanmax(matrix))))
    ax.set_yticks(range(len(comp)))
    ax.set_yticklabels(comp["row"], fontsize=7)
    ax.set_xticks(range(len(heat_cols)))
    ax.set_xticklabels([c.replace("_mean", "") for c in heat_cols], rotation=35, ha="right")
    ax.set_title("Qwen3-Coder Ollama: mean rollout error components")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    out = plot_dir / "rollout_component_heatmap_family_strength.png"
    fig.savefig(out)
    plt.close(fig)
    outputs["rollout_component_heatmap"] = str(out)

    return outputs


def cv_splits(df: pd.DataFrame, y_cls: np.ndarray) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    splits: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    splits["random_kfold"] = list(KFold(n_splits=5, shuffle=True, random_state=0).split(df))
    splits["leave_secret_out"] = list(LeaveOneGroupOut().split(df, groups=df["secret_id"].astype(str)))
    if len(np.unique(y_cls)) > 1 and pd.Series(y_cls).value_counts().min() >= 5:
        splits["random_stratified"] = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(df, y_cls))
    return splits


def regression_cv(model, df: pd.DataFrame, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    pred = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in splits:
        model.fit(df.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(df.iloc[test_idx])
    return pred


def classification_cv(model, df: pd.DataFrame, y: np.ndarray, splits: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros_like(y, dtype=int)
    score = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in splits:
        model.fit(df.iloc[train_idx], y[train_idx])
        pred[test_idx] = model.predict(df.iloc[test_idx])
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df.iloc[test_idx])
            score[test_idx] = proba[:, 1] if proba.shape[1] == 2 else pred[test_idx]
        else:
            score[test_idx] = pred[test_idx]
    return pred, score


def metadata_probe(dstat: pd.DataFrame, out_dir: Path) -> dict:
    df = dstat[dstat["secret_id"].notna()].copy().reset_index(drop=True)
    for col in META_COLS:
        df[col] = df[col].astype(str)
    y = df["D_stat_max"].astype(float).to_numpy()
    y_cls = y > 0

    column = ColumnTransformer([("cat", encoder(), META_COLS)], remainder="drop")
    reg_models = {
        "dummy_mean": make_pipeline(column, DummyRegressor(strategy="mean")),
        "metadata_main_ridge": make_pipeline(column, Ridge(alpha=10.0)),
        "metadata_pairwise_ridge": make_pipeline(
            column,
            PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
            Ridge(alpha=25.0),
        ),
        "metadata_extra_trees": make_pipeline(
            column,
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1),
        ),
    }

    splits = cv_splits(df, y_cls.astype(int))
    reg_rows = []
    for split_name, split_values in splits.items():
        if split_name == "random_stratified":
            continue
        for model_name, model in reg_models.items():
            pred = regression_cv(model, df, y, split_values)
            reg_rows.append(
                {
                    "target": "D_stat_max",
                    "split": split_name,
                    "model": model_name,
                    "n_eval": int(len(y)),
                    "r2": finite(r2_score(y, pred)),
                    "mae": finite(mean_absolute_error(y, pred)),
                    "rmse": finite(mean_squared_error(y, pred) ** 0.5),
                }
            )
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(out_dir / "metadata_regression_metrics.csv", index=False)

    cls_rows = []
    if len(np.unique(y_cls)) > 1:
        cls_models = {
            "dummy_majority": make_pipeline(column, DummyClassifier(strategy="most_frequent")),
            "metadata_main_logit": make_pipeline(
                column,
                LogisticRegression(C=0.3, max_iter=5000, class_weight="balanced"),
            ),
            "metadata_pairwise_logit": make_pipeline(
                column,
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                LogisticRegression(C=0.15, max_iter=5000, class_weight="balanced"),
            ),
        }
        for split_name, split_values in splits.items():
            if split_name == "random_kfold":
                continue
            for model_name, model in cls_models.items():
                pred, score = classification_cv(model, df, y_cls.astype(int), split_values)
                cls_rows.append(
                    {
                        "target": "D_stat_nonzero",
                        "split": split_name,
                        "model": model_name,
                        "n_eval": int(len(y_cls)),
                        "positive_rate": float(y_cls.mean()),
                        "accuracy": float((pred == y_cls).mean()),
                        "balanced_accuracy": finite(balanced_accuracy_score(y_cls, pred)),
                        "roc_auc": finite(roc_auc_score(y_cls, score)),
                        "average_precision": finite(average_precision_score(y_cls, score)),
                    }
                )
    cls = pd.DataFrame(cls_rows)
    cls.to_csv(out_dir / "metadata_classification_metrics.csv", index=False)

    return {
        "regression_csv": str(out_dir / "metadata_regression_metrics.csv"),
        "classification_csv": str(out_dir / "metadata_classification_metrics.csv"),
        "best_regression_random_kfold": (
            reg[reg["split"].eq("random_kfold")].sort_values("r2", ascending=False).head(5).to_dict(orient="records")
        ),
        "best_regression_leave_secret_out": (
            reg[reg["split"].eq("leave_secret_out")].sort_values("r2", ascending=False).head(5).to_dict(orient="records")
        ),
        "classification_rows": cls.to_dict(orient="records"),
    }


def markdown_table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    view = df[columns].head(n).copy()
    return view.to_markdown(index=False, floatfmt=".3f")


def write_report(
    out_dir: Path,
    condition: dict,
    model_show: dict,
    prompt_summary: dict,
    rollout_summary: dict,
    probes: dict,
) -> None:
    family = pd.read_csv(out_dir / "prompt_summary_by_family_strength.csv")
    comps = pd.read_csv(out_dir / "rollout_component_summary_by_family_strength.csv")
    reg = pd.read_csv(out_dir / "metadata_regression_metrics.csv")

    top_family = family.sort_values("mean_D_stat_max", ascending=False)
    best_random = reg[reg["split"].eq("random_kfold")].sort_values("r2", ascending=False).head(4)
    best_lso = reg[reg["split"].eq("leave_secret_out")].sort_values("r2", ascending=False).head(4)

    lines = [
        "# Qwen3-Coder Phase 1 Reanalysis",
        "",
        "## Model and run",
        "",
        f"- Run: `{DEFAULT_RUN_DIR.relative_to(ROOT)}`",
        f"- Backend: `{condition.get('backend')}`",
        f"- Model tag: `{condition.get('model')}`",
        f"- Ollama architecture: `{model_show.get('architecture', 'qwen3moe')}`",
        f"- Ollama reported parameters: `{model_show.get('parameters', '30.5B total')}`",
        f"- Quantization: `{model_show.get('quantization', 'Q4_K_M')}`",
        f"- B: `{condition.get('B')}`",
        f"- Prompts: `{prompt_summary['overall']['n_prompts']}`",
        f"- Rollouts: `{rollout_summary['n_rollouts']}`",
        "",
        "## Important correction",
        "",
        "The behavioral run is from `qwen3-coder:30b-32k`, not from `Qwen/Qwen2.5-0.5B-Instruct`.",
        "The previous hidden-state files were produced with `Qwen/Qwen2.5-0.5B-Instruct` because Ollama does not expose per-layer hidden states through its normal API.",
        "Therefore, those files should not be interpreted as hidden states of the Qwen3-Coder behavioral run.",
        "",
        "## Overall behavior",
        "",
        f"- Nonzero `D_stat_max`: `{prompt_summary['overall']['nonzero_D_stat_max']}/{prompt_summary['overall']['n_prompts']}` = `{prompt_summary['overall']['nonzero_rate']:.3f}`",
        f"- Mean `D_stat_max`: `{prompt_summary['overall']['mean_D_stat_max']:.3f}`",
        f"- Median `D_stat_max`: `{prompt_summary['overall']['median_D_stat_max']:.3f}`",
        f"- Max `D_stat_max`: `{prompt_summary['overall']['max_D_stat_max']:.3f}`",
        f"- Mean rollout `e_max`: `{rollout_summary['mean_e_max']:.3f}`",
        f"- Positive rollout `e_max`: `{rollout_summary['positive_e_max_rate']:.3f}`",
        "",
        "## Strongest family/strength cells",
        "",
        markdown_table(
            top_family,
            [
                "transform_family",
                "transform_strength",
                "n_prompts",
                "nonzero_rate",
                "mean_D_stat_max",
                "median_D_stat_max",
                "mean_D_exact",
                "mean_D_partial",
            ],
            n=12,
        ),
        "",
        "## Rollout component means",
        "",
        markdown_table(
            comps.sort_values("e_max_mean", ascending=False),
            [
                "transform_family",
                "transform_strength",
                "n_rollouts",
                "e_exact_mean",
                "e_transform_mean",
                "e_partial_mean",
                "e_unsafe_coop_mean",
                "e_max_mean",
                "e_max_positive_rate",
            ],
            n=12,
        ),
        "",
        "## Metadata predictability",
        "",
        "Random folds:",
        "",
        markdown_table(best_random, ["split", "model", "r2", "mae", "rmse"], n=4),
        "",
        "Leave-secret-out:",
        "",
        markdown_table(best_lso, ["split", "model", "r2", "mae", "rmse"], n=4),
        "",
        "## Plots",
        "",
        "- `plots/dstat_max_by_family_strength.png`",
        "- `plots/dstat_max_boxplot_by_family.png`",
        "- `plots/rollout_component_heatmap_family_strength.png`",
        "",
        "## Methodological status",
        "",
        "This is a corrected Phase 1 behavioral reanalysis for the stronger local Qwen3-Coder run.",
        "It is not yet a per-layer hidden-state reanalysis of that same model. To do that we need the Hugging Face weights for the same or equivalent Qwen3-Coder model loaded through Transformers, or a backend that exposes hidden states.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reanalyze the real Qwen3-Coder Ollama Phase 1 run.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    condition, prompts, dstat, scores = load_data(args.run_dir)
    prompt_summary = summarize_prompt_level(dstat, out_dir)
    rollout_summary = summarize_rollout_level(scores, out_dir)
    plots = make_plots(dstat, scores, out_dir)
    probes = metadata_probe(dstat, out_dir)

    model_show = {
        "architecture": "qwen3moe",
        "parameters": "30.5B total reported by Ollama",
        "embedding_length": 2048,
        "quantization": "Q4_K_M",
        "num_ctx": 32768,
    }

    summary = {
        "condition": condition,
        "model_show": model_show,
        "prompt_summary": prompt_summary,
        "rollout_summary": rollout_summary,
        "plots": plots,
        "metadata_probe": probes,
        "method_warning": (
            "This reanalysis uses qwen3-coder behavioral outputs. Existing per-layer hidden-state files are from "
            "Qwen/Qwen2.5-0.5B-Instruct, because Ollama does not expose per-layer hidden states."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    write_report(out_dir, condition, model_show, prompt_summary, rollout_summary, probes)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
