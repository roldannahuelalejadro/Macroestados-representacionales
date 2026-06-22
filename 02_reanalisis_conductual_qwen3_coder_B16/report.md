# Qwen3-Coder Phase 1 Reanalysis

## Model and run

- Run: `outputs\qwen3_ollama_general_sweep_b8_deduped`
- Backend: `ollama`
- Model tag: `qwen3-coder:30b-32k`
- Ollama architecture: `qwen3moe`
- Ollama reported parameters: `30.5B total reported by Ollama`
- Quantization: `Q4_K_M`
- B: `16`
- Prompts: `100`
- Rollouts: `1600`

## Important correction

The behavioral run is from `qwen3-coder:30b-32k`, not from `Qwen/Qwen2.5-0.5B-Instruct`.
The previous hidden-state files were produced with `Qwen/Qwen2.5-0.5B-Instruct` because Ollama does not expose per-layer hidden states through its normal API.
Therefore, those files should not be interpreted as hidden states of the Qwen3-Coder behavioral run.

## Overall behavior

- Nonzero `D_stat_max`: `48/100` = `0.480`
- Mean `D_stat_max`: `0.878`
- Median `D_stat_max`: `0.000`
- Max `D_stat_max`: `4.000`
- Mean rollout `e_max`: `0.274`
- Positive rollout `e_max`: `0.277`

## Strongest family/strength cells

| transform_family     |   transform_strength |   n_prompts |   nonzero_rate |   mean_D_stat_max |   median_D_stat_max |   mean_D_exact |   mean_D_partial |
|:---------------------|---------------------:|------------:|---------------:|------------------:|--------------------:|---------------:|-----------------:|
| transform_extraction |                    2 |           6 |          1.000 |             4.000 |               4.000 |          3.333 |            3.333 |
| partial_extraction   |                    3 |           6 |          1.000 |             3.755 |               4.000 |          0.326 |            1.438 |
| transform_extraction |                    3 |           6 |          1.000 |             2.454 |               3.265 |          1.692 |            1.692 |
| context_injection    |                    2 |          16 |          0.750 |             0.917 |               0.122 |          0.631 |            0.900 |
| partial_extraction   |                    2 |           6 |          1.000 |             0.896 |               0.675 |          0.011 |            0.329 |
| transform_extraction |                    1 |           6 |          0.500 |             0.771 |               0.032 |          0.104 |            0.104 |
| context_injection    |                    1 |          16 |          0.312 |             0.093 |               0.000 |          0.042 |            0.093 |
| context_injection    |                    3 |          16 |          0.125 |             0.012 |               0.000 |          0.012 |            0.012 |
| none                 |                    0 |          16 |          0.125 |             0.012 |               0.000 |          0.000 |            0.012 |
| partial_extraction   |                    1 |           6 |          0.000 |             0.000 |               0.000 |          0.000 |            0.000 |

## Rollout component means

| transform_family     |   transform_strength |   n_rollouts |   e_exact_mean |   e_transform_mean |   e_partial_mean |   e_unsafe_coop_mean |   e_max_mean |   e_max_positive_rate |
|:---------------------|---------------------:|-------------:|---------------:|-------------------:|-----------------:|---------------------:|-------------:|----------------------:|
| transform_extraction |                    2 |           96 |          0.833 |              0.156 |            0.833 |                0.000 |        1.000 |                 1.000 |
| partial_extraction   |                    3 |           96 |          0.146 |              0.000 |            0.424 |                0.677 |        0.990 |                 0.990 |
| transform_extraction |                    3 |           96 |          0.510 |              0.479 |            0.510 |                0.010 |        0.688 |                 0.688 |
| partial_extraction   |                    2 |           96 |          0.010 |              0.000 |            0.243 |                0.156 |        0.472 |                 0.479 |
| context_injection    |                    2 |          256 |          0.254 |              0.008 |            0.330 |                0.020 |        0.346 |                 0.352 |
| transform_extraction |                    1 |           96 |          0.083 |              0.000 |            0.083 |                0.000 |        0.250 |                 0.250 |
| context_injection    |                    1 |          256 |          0.039 |              0.000 |            0.065 |                0.004 |        0.067 |                 0.078 |
| context_injection    |                    3 |          256 |          0.012 |              0.000 |            0.012 |                0.000 |        0.012 |                 0.012 |
| none                 |                    0 |          256 |          0.000 |              0.000 |            0.010 |                0.000 |        0.010 |                 0.012 |
| partial_extraction   |                    1 |           96 |          0.000 |              0.000 |            0.000 |                0.000 |        0.000 |                 0.000 |

## Metadata predictability

Random folds:

| split        | model                   |     r2 |   mae |   rmse |
|:-------------|:------------------------|-------:|------:|-------:|
| random_kfold | metadata_extra_trees    |  0.716 | 0.359 |  0.760 |
| random_kfold | metadata_pairwise_ridge |  0.348 | 0.841 |  1.151 |
| random_kfold | metadata_main_ridge     |  0.250 | 0.929 |  1.235 |
| random_kfold | dummy_mean              | -0.038 | 1.123 |  1.453 |

Leave-secret-out:

| split            | model                   |     r2 |   mae |   rmse |
|:-----------------|:------------------------|-------:|------:|-------:|
| leave_secret_out | metadata_extra_trees    |  0.729 | 0.354 |  0.742 |
| leave_secret_out | metadata_pairwise_ridge |  0.450 | 0.784 |  1.057 |
| leave_secret_out | metadata_main_ridge     |  0.357 | 0.858 |  1.143 |
| leave_secret_out | dummy_mean              | -0.003 | 1.108 |  1.428 |

## Plots

- `plots/dstat_max_by_family_strength.png`
- `plots/dstat_max_boxplot_by_family.png`
- `plots/rollout_component_heatmap_family_strength.png`

## Methodological status

This is a corrected Phase 1 behavioral reanalysis for the stronger local Qwen3-Coder run.
It is not yet a per-layer hidden-state reanalysis of that same model. To do that we need the Hugging Face weights for the same or equivalent Qwen3-Coder model loaded through Transformers, or a backend that exposes hidden states.