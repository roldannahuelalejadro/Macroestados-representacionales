# LLM Safety Macrostates

This repository organizes the latest D-stat analyses for safety-prompt experiments.

In short:

- We measured safety difficulty as a population-level property of rollouts.
- We built proxy hidden-state coordinates.
- We fit effective models to test whether difficulty is organized by directions and couplings.
- Compositions are not additive: AB does not behave like A+B.
- This supports triage: prioritize compositions that deserve more rollouts or larger models.

## Repository Map

- `paper/` contains the English LaTeX manuscript, bibliography, figures, and build artifacts.
- `data/raw/` contains rollout logs, scored rollouts, generation metrics, run configuration, and raw HTML reports.
- `data/processed/` contains behavioral scores, reanalysis tables, proxy hidden-state tables, regression/classification metrics, and PCA coordinates.
- `prompts/` contains the composed prompt table plus derived base-prompt and transformation tables.
- `src/` contains the scripts used for scoring, representation extraction, modeling, and figure generation.
- `reports/` contains Markdown reports, PDFs, legacy source material, and generated figures.
- `docs/` keeps planning notes and the original Spanish summary as an archive.

## Main Run

The main behavioral run used `qwen3-coder:30b-32k` through Ollama with `B=16`, temperature `0.7`, top-p `0.9`, and the transform families `none`, `context_injection`, `partial_extraction`, and `transform_extraction`.

The representation analysis uses `Qwen/Qwen2.5-0.5B-Instruct` as a proxy hidden-state model. This means the behavioral difficulty was measured on Qwen3-Coder, while the representation geometry is a proxy analysis from a smaller Qwen-family model.

## Quick Start

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

The central tables are:

- `data/processed/behavioral_scores.csv`
- `data/processed/regression_metrics.csv`
- `data/processed/classification_metrics.csv`
- `prompts/composed_prompts.csv`

The current bundle does not include a finalized `scattering_pairs.csv`; composition and scattering notes are kept in the paper and reports.