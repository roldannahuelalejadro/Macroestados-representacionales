# Data

`raw/rollout_reports/qwen3_coder_b16_three_transforms/` contains the original rollout-level artifacts from the B=16 Qwen3-Coder run: generated responses, scoring JSONL, generation metrics, run logs, and the run configuration.

`raw/html_reports/` contains the original HTML report exported from that run.

`processed/behavioral_scores.csv` is the prompt-level D-stat score table.

`processed/behavioral_reanalysis_qwen3_coder_b16/` contains summary tables for the B=16 behavioral reanalysis.

`processed/proxy_hidden_states_qwen2p5_0p5b/` contains proxy hidden-state samples, per-layer vectors, PCA coordinates, and layer-separability metrics.

`processed/proxy_deltas_vs_control/` contains matched-control delta representations and their layer/PCA metrics.

`processed/proxy_nonlinear_dstat_prediction/` contains regression and classification outputs for predicting D-stat from metadata and proxy representations.

`processed/large_run_reanalysis_b8/` contains the larger B=8 comparison reanalysis.

No raw model weights are stored here.