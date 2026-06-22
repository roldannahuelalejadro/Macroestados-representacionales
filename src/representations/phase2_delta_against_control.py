from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .phase2_hidden_utils import (
        ROOT,
        ensure_dir,
        layer_metrics,
        load_npz_vectors,
        pca_coords,
        plot_pca,
        write_json,
    )
except ImportError:
    from phase2_hidden_utils import (
        ROOT,
        ensure_dir,
        layer_metrics,
        load_npz_vectors,
        pca_coords,
        plot_pca,
        write_json,
    )


def compute_delta_vectors(samples: pd.DataFrame, vectors: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, dict]:
    key_cols = ["secret_id", "base_prompt_index"]
    control_mask = (samples["transform_family"] == "none") & (samples["transform_strength"].astype(int) == 0)
    controls = samples[control_mask].copy()
    if controls.empty:
        raise ValueError("No control prompts found: expected transform_family=none and transform_strength=0.")

    control_index = {}
    for idx, row in controls.iterrows():
        key = tuple(row[col] for col in key_cols)
        control_index[key] = idx

    keep_indices = []
    control_indices = []
    missing = []
    for idx, row in samples.iterrows():
        key = tuple(row[col] for col in key_cols)
        if key not in control_index:
            missing.append((idx, key))
            continue
        keep_indices.append(idx)
        control_indices.append(control_index[key])

    if missing:
        print(f"warning: dropped {len(missing)} samples without matched none-control")

    keep_indices_np = np.array(keep_indices, dtype=int)
    control_indices_np = np.array(control_indices, dtype=int)

    delta = vectors[keep_indices_np] - vectors[control_indices_np]
    delta_samples = samples.iloc[keep_indices_np].copy().reset_index(drop=True)
    delta_samples["control_prompt_id"] = samples.iloc[control_indices_np]["prompt_id"].to_numpy()
    delta_samples["delta_norm_mean_layers"] = np.linalg.norm(delta, axis=2).mean(axis=1)

    meta = {
        "n_input_samples": int(len(samples)),
        "n_delta_samples": int(len(delta_samples)),
        "n_missing_controls": int(len(missing)),
        "delta_definition": "a(q_alpha) - a(q_secret_base_none_0)",
        "control_key": key_cols,
    }
    return delta_samples, delta, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 2: subtract matched none-control activations.")
    parser.add_argument(
        "--layer-sweep-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2_layer_sweep",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2_delta_against_control",
    )
    args = parser.parse_args()

    layer_sweep_dir = args.layer_sweep_dir
    out_dir = ensure_dir(args.out_dir)

    samples = pd.read_csv(layer_sweep_dir / "samples.csv")
    vectors, layers = load_npz_vectors(layer_sweep_dir / "hidden_vectors_by_layer.npz")

    delta_samples, delta_vectors, meta = compute_delta_vectors(samples, vectors)
    delta_samples.to_csv(out_dir / "delta_samples.csv", index=False)
    np.savez_compressed(
        out_dir / "delta_vectors_by_layer.npz",
        vectors=delta_vectors,
        layers=np.array(layers, dtype=np.int32),
        sample_id=delta_samples["sample_id"].to_numpy(),
        prompt_id=delta_samples["prompt_id"].to_numpy(),
        control_prompt_id=delta_samples["control_prompt_id"].to_numpy(),
    )

    metrics_family = layer_metrics(delta_samples, delta_vectors, layers, label_col="transform_family")
    metrics_family.to_csv(out_dir / "delta_layer_metrics_transform_family.csv", index=False)

    noncontrol = delta_samples["transform_family"] != "none"
    if noncontrol.sum() >= 8:
        metrics_noncontrol = layer_metrics(
            delta_samples[noncontrol].reset_index(drop=True),
            delta_vectors[noncontrol.to_numpy()],
            layers,
            label_col="transform_family",
        )
        metrics_noncontrol.to_csv(out_dir / "delta_layer_metrics_transform_family_noncontrol.csv", index=False)
        ranking = metrics_noncontrol
    else:
        ranking = metrics_family

    best_layer = int(ranking.sort_values("silhouette_pca2", ascending=False).iloc[0]["layer"])
    best_pos = layers.index(best_layer)

    coords, explained = pca_coords(delta_vectors[:, best_pos, :])
    pca_df = delta_samples.copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df["layer"] = best_layer
    pca_df.to_csv(out_dir / "delta_pca_best_layer_transform_family_points.csv", index=False)
    plot_pca(
        samples=delta_samples,
        coords=coords,
        color_col="transform_family",
        title=f"Delta hidden-state PCA by transform family (layer {best_layer})",
        out_path=out_dir / "delta_pca_best_layer_transform_family.png",
        explained=explained,
    )

    if noncontrol.sum() >= 8:
        coords_nc, explained_nc = pca_coords(delta_vectors[noncontrol.to_numpy(), best_pos, :])
        plot_pca(
            samples=delta_samples[noncontrol].reset_index(drop=True),
            coords=coords_nc,
            color_col="transform_family",
            title=f"Delta PCA by transform family, controls excluded (layer {best_layer})",
            out_path=out_dir / "delta_pca_best_layer_transform_family_noncontrol.png",
            explained=explained_nc,
        )

    summary = {
        **meta,
        "layers": layers,
        "best_layer_by_noncontrol_silhouette_pca2": best_layer,
        "outputs": {
            "samples": str(out_dir / "delta_samples.csv"),
            "vectors": str(out_dir / "delta_vectors_by_layer.npz"),
            "metrics": str(out_dir / "delta_layer_metrics_transform_family.csv"),
            "pca_plot": str(out_dir / "delta_pca_best_layer_transform_family.png"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
