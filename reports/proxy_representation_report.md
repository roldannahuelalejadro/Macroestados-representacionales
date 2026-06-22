# Representation Analysis for the Three-Transform Run

## Scope

Behavioral run:

- Model: `qwen3-coder:30b-32k` via Ollama.
- Run: `outputs/qwen3_ollama_three_transform_b16`.
- Prompts analyzed: `90` secret-bearing prompts.
- Rollouts: `1600`.
- Transform families: `context_injection`, `partial_extraction`, `transform_extraction`, plus `none`.

Representation source:

- Proxy hidden states from `Qwen/Qwen2.5-0.5B-Instruct` via Transformers.
- This is not the hidden-state space of `qwen3-coder:30b-32k`.
- Ollama currently does not expose embeddings in this session: `/api/embed` and `/api/embeddings` return `This server does not support embeddings. Start it with --embeddings`.

Therefore this analysis tests whether the prompt geometry induced by these transformations is visible in a local Qwen-family representation proxy, and whether that proxy aligns with the `D_stat` measured on Qwen3-Coder.

## Main Result

The proxy representation separates the transformation families very clearly.

Raw hidden states:

- Best layer by PCA silhouette: `8`.
- Best `silhouette_pca2`: `0.549`.
- Linear transform-family probe accuracy: `1.0`.

Delta hidden states:

- Delta definition: activation of transformed prompt minus matched `none` control with same `secret_id` and `base_prompt_index`.
- Best non-control layer by PCA silhouette: `20`.
- Best non-control `silhouette_pca2`: `0.327`.
- Linear transform-family probe accuracy: `1.0`.

Interpretation: the transformation families are not subtle in representation space. They induce robust, linearly separable directions even after subtracting matched controls.

## Alignment with Difficulty

Regression target: `D_stat_max`.

Best random-kfold models:

| source | model | layer | R2 |
|---|---:|---:|---:|
| raw hidden | linear PCA ridge | 20 | 0.680 |
| delta hidden | linear PCA ridge | 20 | 0.664 |
| metadata pairwise | ridge | - | 0.375 |
| metadata main effects | ridge | - | 0.292 |
| dummy mean | baseline | - | -0.015 |

Best leave-secret-out models:

| source | model | layer | R2 |
|---|---:|---:|---:|
| delta hidden | linear PCA ridge | 20 | 0.713 |
| raw hidden | linear PCA ridge | 20 | 0.680 |
| metadata pairwise | ridge | - | 0.450 |
| metadata main effects | ridge | - | 0.357 |
| dummy mean | baseline | - | -0.003 |

This is the strongest signal so far that the measured behavioral difficulty is reflected in representation geometry. The hidden proxy beats metadata-only models by a large margin.

## Classification

Target: `D_stat_max > 0`.

Best balanced accuracy:

- Random k-fold, delta hidden linear probe: `0.844`.
- Leave-secret-out, delta hidden linear probe: `0.844`.
- Metadata pairwise, random k-fold: `0.766`.
- Metadata pairwise, leave-secret-out: `0.787`.
- Dummy baseline: `0.500`.

Interpretation: the proxy representation contains a fairly clean boundary between zero-difficulty and nonzero-difficulty prompts.

## Geometry

Raw PCA, layer `8`:

- `context_injection`, `none`, `partial_extraction`, and `transform_extraction` form distinct regions.
- `partial_extraction` and `transform_extraction` occupy the lower sector.
- `context_injection` occupies a separated upper-left sector.
- `none` is far from the attack transformations.

Delta PCA, layer `20`:

- After subtracting matched controls, `context_injection` separates strongly from the extraction families.
- `partial_extraction` and `transform_extraction` remain close to each other but not identical.
- High `D_stat_max` concentrates mostly in the extraction sector.

## Interpretation in the Formalism

This supports a macrostate picture with at least three empirical regions:

1. Safe/control region: `none`, and much of `context_injection`.
2. Prompt-reframing region: `context_injection`, representationally distinct but often not behaviorally unsafe.
3. Extraction region: `partial_extraction` and `transform_extraction`, where high `D_stat` concentrates.

The key point is that representational separability of a transformation does not imply high danger. `context_injection` is separated in representation space, but much of it stays low `D_stat`. The dangerous region is more specifically the extraction subspace.

This matches the earlier behavioral conclusion: simple override attacks are not the main failure mode. The strong failure mode is symbolic or partial extraction, where the model appears to reclassify the task as a transformation of a string rather than as secret disclosure.

## Outputs

- Raw proxy hidden analysis: `outputs/qwen3_ollama_three_transform_b16_proxy_hidden`.
- Delta proxy hidden analysis: `outputs/qwen3_ollama_three_transform_b16_proxy_delta`.
- Nonlinear analysis: `outputs/qwen3_ollama_three_transform_b16_proxy_nonlinear`.
- D-stat PCA plots: `outputs/qwen3_ollama_three_transform_b16_proxy_representation_plots`.

Key plots:

- `outputs/qwen3_ollama_three_transform_b16_proxy_hidden/pca_best_layer_transform_family.png`
- `outputs/qwen3_ollama_three_transform_b16_proxy_delta/delta_pca_best_layer_transform_family_noncontrol.png`
- `outputs/qwen3_ollama_three_transform_b16_proxy_representation_plots/raw_pca_dstat.png`
- `outputs/qwen3_ollama_three_transform_b16_proxy_representation_plots/delta_noncontrol_pca_dstat.png`
- `outputs/qwen3_ollama_three_transform_b16_proxy_nonlinear/dstat_regression_random_kfold.png`

## Methodological Caveat

This is not yet a final Phase 2 measurement of the same Qwen3-Coder model. It is a representation proxy analysis. To obtain same-model representation data, we need one of:

1. Ollama restarted/configured with embeddings enabled, if this local build supports it.
2. A Transformers-loadable Qwen3-Coder model, preferably quantized and compatible with the local GPU/RAM.
3. A backend that exposes per-layer hidden states for the exact model used in the behavioral run.

Until then, the defensible claim is:

The Qwen-family proxy representation contains strong geometric correlates of the Qwen3-Coder behavioral difficulty measured by `D_stat`, especially for extraction-style transformations.
