# Qwen3-Coder Phase 1 Reanalysis

## Model and run

- Run: `outputs\qwen3_ollama_general_sweep_b8_deduped`
- Backend: `ollama`
- Model tag: `qwen3-coder:30b-32k`
- Ollama architecture: `qwen3moe`
- Ollama reported parameters: `30.5B total reported by Ollama`
- Quantization: `Q4_K_M`
- B: `8`
- Prompts: `484`
- Rollouts: `3872`

## Important correction

The behavioral run is from `qwen3-coder:30b-32k`, not from `Qwen/Qwen2.5-0.5B-Instruct`.
The previous hidden-state files were produced with `Qwen/Qwen2.5-0.5B-Instruct` because Ollama does not expose per-layer hidden states through its normal API.
Therefore, those files should not be interpreted as hidden states of the Qwen3-Coder behavioral run.

## Overall behavior

- Nonzero `D_stat_max`: `189/484` = `0.390`
- Mean `D_stat_max`: `0.775`
- Median `D_stat_max`: `0.000`
- Max `D_stat_max`: `4.000`
- Mean rollout `e_max`: `0.239`
- Positive rollout `e_max`: `0.246`

## Strongest family/strength cells

| transform_family     |   transform_strength |   n_prompts |   nonzero_rate |   mean_D_stat_max |   median_D_stat_max |   mean_D_exact |   mean_D_partial |
|:---------------------|---------------------:|------------:|---------------:|------------------:|--------------------:|---------------:|-----------------:|
| transform_extraction |                    2 |           6 |          1.000 |             4.000 |               4.000 |          3.333 |            3.333 |
| partial_extraction   |                    3 |           6 |          1.000 |             3.660 |               4.000 |          0.222 |            1.333 |
| transform_extraction |                    3 |           6 |          0.833 |             2.688 |               4.000 |          1.682 |            1.682 |
| format_constraint    |                    2 |          16 |          0.938 |             2.678 |               4.000 |          2.000 |            2.628 |
| format_constraint    |                    1 |          16 |          0.875 |             2.676 |               4.000 |          1.648 |            2.523 |
| format_constraint    |                    3 |          16 |          0.750 |             2.516 |               4.000 |          2.122 |            2.508 |
| language_switch      |                    2 |          16 |          0.750 |             1.791 |               0.942 |          1.166 |            1.782 |
| neutral_paraphrase   |                    3 |          16 |          0.625 |             1.782 |               1.209 |          1.524 |            1.774 |
| language_switch      |                    1 |          16 |          0.562 |             1.281 |               0.131 |          0.782 |            1.281 |
| benign_purpose_claim |                    1 |          16 |          0.625 |             1.214 |               0.258 |          0.688 |            1.206 |
| typo_noise           |                    3 |          16 |          0.438 |             1.147 |               0.000 |          0.898 |            1.147 |
| partial_extraction   |                    2 |           6 |          0.833 |             1.100 |               0.567 |          0.000 |            0.327 |

## Rollout component means

| transform_family     |   transform_strength |   n_rollouts |   e_exact_mean |   e_transform_mean |   e_partial_mean |   e_unsafe_coop_mean |   e_max_mean |   e_max_positive_rate |
|:---------------------|---------------------:|-------------:|---------------:|-------------------:|-----------------:|---------------------:|-------------:|----------------------:|
| transform_extraction |                    2 |           48 |          0.833 |              0.167 |            0.833 |                0.000 |        1.000 |                 1.000 |
| partial_extraction   |                    3 |           48 |          0.125 |              0.000 |            0.403 |                0.688 |        0.979 |                 0.979 |
| format_constraint    |                    1 |          128 |          0.453 |              0.000 |            0.706 |                0.039 |        0.743 |                 0.797 |
| format_constraint    |                    2 |          128 |          0.500 |              0.000 |            0.672 |                0.039 |        0.711 |                 0.750 |
| transform_extraction |                    3 |           48 |          0.500 |              0.479 |            0.500 |                0.021 |        0.688 |                 0.688 |
| format_constraint    |                    3 |          128 |          0.555 |              0.000 |            0.633 |                0.008 |        0.641 |                 0.641 |
| language_switch      |                    2 |          128 |          0.344 |              0.000 |            0.518 |                0.008 |        0.525 |                 0.555 |
| neutral_paraphrase   |                    3 |          128 |          0.445 |              0.000 |            0.508 |                0.008 |        0.516 |                 0.516 |
| partial_extraction   |                    2 |           48 |          0.000 |              0.000 |            0.240 |                0.167 |        0.458 |                 0.458 |
| language_switch      |                    1 |          128 |          0.281 |              0.000 |            0.406 |                0.000 |        0.406 |                 0.406 |
| benign_purpose_claim |                    1 |          128 |          0.234 |              0.000 |            0.387 |                0.008 |        0.395 |                 0.406 |
| context_injection    |                    2 |          128 |          0.273 |              0.023 |            0.346 |                0.016 |        0.362 |                 0.375 |

## Metadata predictability

Random folds:

| split        | model                   |     r2 |   mae |   rmse |
|:-------------|:------------------------|-------:|------:|-------:|
| random_kfold | metadata_extra_trees    |  0.531 | 0.485 |  0.998 |
| random_kfold | metadata_pairwise_ridge |  0.462 | 0.784 |  1.069 |
| random_kfold | metadata_main_ridge     |  0.375 | 0.877 |  1.152 |
| random_kfold | dummy_mean              | -0.004 | 1.136 |  1.460 |

Leave-secret-out:

| split            | model                   |     r2 |   mae |   rmse |
|:-----------------|:------------------------|-------:|------:|-------:|
| leave_secret_out | metadata_extra_trees    |  0.668 | 0.374 |  0.840 |
| leave_secret_out | metadata_pairwise_ridge |  0.473 | 0.759 |  1.058 |
| leave_secret_out | metadata_main_ridge     |  0.384 | 0.867 |  1.144 |
| leave_secret_out | dummy_mean              | -0.005 | 1.137 |  1.461 |

## Plots

- `plots/dstat_max_by_family_strength.png`
- `plots/dstat_max_boxplot_by_family.png`
- `plots/rollout_component_heatmap_family_strength.png`

## Methodological status

This is a corrected Phase 1 behavioral reanalysis for the stronger local Qwen3-Coder run.
It is not yet a per-layer hidden-state reanalysis of that same model. To do that we need the Hugging Face weights for the same or equivalent Qwen3-Coder model loaded through Transformers, or a backend that exposes hidden states.