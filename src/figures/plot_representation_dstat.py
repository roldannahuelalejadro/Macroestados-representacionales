from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]


def load_vectors(path: Path) -> tuple[np.ndarray, list[int]]:
    data = np.load(path, allow_pickle=True)
    return data["vectors"], [int(x) for x in data["layers"].tolist()]


def pca2(X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(Xs)
    return coords, [float(x) for x in pca.explained_variance_ratio_]


def scatter_dstat(samples: pd.DataFrame, coords: np.ndarray, explained: list[float], title: str, out: Path) -> None:
    values = samples["D_stat_max"].astype(float).to_numpy()
    families = samples["transform_family"].astype(str).to_numpy()
    markers = {
        "none": "o",
        "context_injection": "s",
        "partial_extraction": "^",
        "transform_extraction": "D",
    }

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    for family in sorted(pd.unique(families)):
        idx = families == family
        sc = ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=values[idx],
            cmap="magma",
            vmin=0,
            vmax=max(4.0, float(np.nanmax(values))),
            marker=markers.get(family, "o"),
            s=52,
            alpha=0.86,
            edgecolors="white",
            linewidths=0.45,
            label=f"{family} (n={idx.sum()})",
        )
    ax.axhline(0, color="#777777", linewidth=0.7, alpha=0.4)
    ax.axvline(0, color="#777777", linewidth=0.7, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% var.)")
    ax.grid(alpha=0.16)
    ax.legend(fontsize=7, loc="best")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("D_stat_max")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dir", type=Path, required=True)
    parser.add_argument("--delta-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-layer", type=int, default=8)
    parser.add_argument("--delta-layer", type=int, default=20)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    raw_samples = pd.read_csv(args.hidden_dir / "samples.csv")
    raw_vectors, raw_layers = load_vectors(args.hidden_dir / "hidden_vectors_by_layer.npz")
    raw_pos = raw_layers.index(args.raw_layer)
    raw_coords, raw_explained = pca2(raw_vectors[:, raw_pos, :])
    raw_points = raw_samples.copy()
    raw_points["PC1"] = raw_coords[:, 0]
    raw_points["PC2"] = raw_coords[:, 1]
    raw_points["layer"] = args.raw_layer
    raw_points.to_csv(args.out_dir / "raw_pca_dstat_points.csv", index=False)
    scatter_dstat(
        raw_samples,
        raw_coords,
        raw_explained,
        f"Raw proxy hidden PCA colored by D_stat_max (layer {args.raw_layer})",
        args.out_dir / "raw_pca_dstat.png",
    )

    delta_samples = pd.read_csv(args.delta_dir / "delta_samples.csv")
    delta_vectors, delta_layers = load_vectors(args.delta_dir / "delta_vectors_by_layer.npz")
    noncontrol = delta_samples["transform_family"].ne("none").to_numpy()
    delta_pos = delta_layers.index(args.delta_layer)
    delta_coords, delta_explained = pca2(delta_vectors[noncontrol, delta_pos, :])
    delta_points = delta_samples[noncontrol].copy().reset_index(drop=True)
    delta_points["PC1"] = delta_coords[:, 0]
    delta_points["PC2"] = delta_coords[:, 1]
    delta_points["layer"] = args.delta_layer
    delta_points.to_csv(args.out_dir / "delta_noncontrol_pca_dstat_points.csv", index=False)
    scatter_dstat(
        delta_points,
        delta_coords,
        delta_explained,
        f"Delta proxy hidden PCA colored by D_stat_max, controls excluded (layer {args.delta_layer})",
        args.out_dir / "delta_noncontrol_pca_dstat.png",
    )

    print(args.out_dir)


if __name__ == "__main__":
    main()
