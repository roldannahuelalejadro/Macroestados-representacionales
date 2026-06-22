from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from phase2_hidden_utils import (
    DEFAULT_LAYERS,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_ID,
    ROOT,
    build_samples,
    ensure_dir,
    layer_metrics,
    pca_coords,
    plot_pca,
    write_json,
)


def parse_layers(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def extract_layer_vectors(
    samples,
    model_id: str,
    layers: list[int],
    max_length: int,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    max_hidden_index = int(model.config.num_hidden_layers)
    bad_layers = [layer for layer in layers if layer < 0 or layer > max_hidden_index]
    if bad_layers:
        raise ValueError(f"Invalid hidden-state indices {bad_layers}; valid range is 0..{max_hidden_index}")

    all_vectors: list[np.ndarray] = []
    texts = samples["text"].astype(str).tolist()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            mask = encoded["attention_mask"].bool()
            denom = mask.sum(dim=1).clamp(min=1).view(-1, 1)

            batch_layers = []
            for layer in layers:
                hs = outputs.hidden_states[layer]
                pooled = (hs * mask.unsqueeze(-1)).sum(dim=1) / denom
                batch_layers.append(pooled.detach().float().cpu().numpy())
            # (batch, n_layers, hidden)
            all_vectors.append(np.stack(batch_layers, axis=1))

            print(f"processed {min(start + batch_size, len(texts))}/{len(texts)}")

    vectors = np.concatenate(all_vectors, axis=0)
    meta = {
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "hidden_size": int(model.config.hidden_size),
        "layers": layers,
        "representation": "mean_pool_nonpad_tokens",
        "n_samples": int(len(samples)),
        "max_length": int(max_length),
        "batch_size": int(batch_size),
    }
    return vectors, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Step 1: raw hidden-state layer sweep.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-per-family", type=int, default=None)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory containing prompts.csv and dstat_by_prompt.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "phase2_layer_sweep",
    )
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    layers = parse_layers(args.layers)

    samples = build_samples(run_dir=args.run_dir, max_per_family=args.max_per_family) if args.run_dir else build_samples(max_per_family=args.max_per_family)
    samples.to_csv(out_dir / "samples.csv", index=False)

    vectors, meta = extract_layer_vectors(
        samples=samples,
        model_id=args.model_id,
        layers=layers,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    np.savez_compressed(
        out_dir / "hidden_vectors_by_layer.npz",
        vectors=vectors,
        layers=np.array(layers, dtype=np.int32),
        sample_id=samples["sample_id"].to_numpy(),
        prompt_id=samples["prompt_id"].to_numpy(),
    )

    metrics = layer_metrics(samples, vectors, layers, label_col="transform_family")
    metrics.to_csv(out_dir / "layer_metrics_transform_family.csv", index=False)

    best_layer = int(metrics.sort_values("silhouette_pca2", ascending=False).iloc[0]["layer"])
    best_pos = layers.index(best_layer)
    coords, explained = pca_coords(vectors[:, best_pos, :])
    pca_df = samples.copy()
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df["layer"] = best_layer
    pca_df.to_csv(out_dir / "pca_best_layer_transform_family_points.csv", index=False)
    plot_pca(
        samples=samples,
        coords=coords,
        color_col="transform_family",
        title=f"Raw hidden-state PCA by transform family (layer {best_layer})",
        out_path=out_dir / "pca_best_layer_transform_family.png",
        explained=explained,
    )

    summary = {
        **meta,
        "best_layer_by_silhouette_pca2": best_layer,
        "best_silhouette_pca2": float(metrics["silhouette_pca2"].max()),
        "outputs": {
            "samples": str(out_dir / "samples.csv"),
            "vectors": str(out_dir / "hidden_vectors_by_layer.npz"),
            "metrics": str(out_dir / "layer_metrics_transform_family.csv"),
            "pca_plot": str(out_dir / "pca_best_layer_transform_family.png"),
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
