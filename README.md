# AutoTail-BSFGM Scholarly Classification

This repository contains the code, configurations, paper source, and lightweight audit artifacts for:

**AutoTail-BSFGM: Class-Balance-Aware Fine-Tuning for Chinese Scholarly Text Classification**

The project studies robust and class-balance-aware fine-tuning for Chinese scholarly text classification. The current paper evaluates AutoTail-BSFGM on two CSL-derived tasks:

- Abstract-to-discipline classification with 67 discipline labels.
- CSL title-to-category classification with 13 broad categories.

The repository is intentionally lightweight. It does **not** include model checkpoints, cached pretrained models, raw training logs, or large prediction parquet files.

## Current Claim Boundary

Supported claim:

> AutoTail-BSFGM improves same-encoder Chinese scholarly abstract classification across two base-size encoders and improves class-balance-sensitive metrics on a second CSL title-classification task, without changing inference-time architecture.

Not claimed:

- Official hidden-test leaderboard state of the art.
- Universal improvement on every split or every metric.
- A new pretrained language model.
- A general long-tail learning theory.

## Repository Contents

| Path | Purpose |
|---|---|
| `src/newalg/` | Training runner, method implementation, judging, reporting, and research-loop utilities. |
| `configs/` | Experiment specifications used during the study. |
| `data/` | Small local JSONL conversions of public CSL-derived tasks and label maps. |
| `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/` | arXiv/JASIST-style manuscript source, figures, tables, and artifact manifest. |
| `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/*.csv` | Source tables for the paper results. |
| `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.svg` | Editable figure sources. |
| `tests/` | Lightweight repository health checks for data loading, reporting, dashboards, proposal utilities, and traditional baselines. These tests do not rerun the full paper experiments. |
| `REPRODUCIBILITY.md` | Practical guide for data splits, rerun commands, and reproduction boundaries. |
| `DATA_AVAILABILITY.md` | Mapping from the paper's data availability statement to repository files. |
| `docs/repository_publication_plan.md` | What is included, excluded, and still needed for a public GitHub release. |
| `docs/code_scope_for_paper.md` | Separation between paper-reproduction files, broader platform code, and legacy experiments. |

## What Is Excluded

The following local artifacts are intentionally excluded from GitHub:

- `outputs/artifacts/`: model checkpoints, prediction parquet files, confusion matrices, and error samples.
- `outputs/pretrained/` and `outputs/cache/`: downloaded model/data caches.
- `outputs/logs/`: raw local training logs.
- `outputs/runs.duckdb`: local run registry.
- `outputs/submissions/`: official-test prediction files.
- `paper/**/arxiv_source.zip` and review packages: generated upload archives.
- `references/**/*.pdf`: locally downloaded reference papers used for reading/template comparison.

Large artifacts can be regenerated from the recorded configs or shared separately through a release, institutional storage, or Zenodo if needed.

## Quick Start

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

The test suite is a fast repository-level check. It covers toy fixtures, dataset handling, reporting/judging utilities, proposal generation, offline literature discovery, dashboard rendering, optimizer bookkeeping, and a traditional baseline smoke test. It is not a substitute for rerunning the CSL fine-tuning configs listed below.

Compile the paper:

```bash
cd paper/arxiv_scholarly_autotail_bsaux_fgm_20260528
tectonic main.tex
```

## Reproducing Experiments

For a practical reproduction guide, see `REPRODUCIBILITY.md`.

The paper-level run identifiers and source tables are listed in:

- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/artifact_manifest.md`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/per_seed_runs.csv`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/main_results_journal.csv`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/significance_and_cost.csv`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/category_delta_analysis.csv`

Representative configs for the paper are under `configs/`, including:

- `jasist_csl_abstract_screen_seed13.yaml`
- `jasist_csl_abstract_screen_seed42.yaml`
- `jasist_csl_abstract_confirm_seed3407.yaml`
- `jasist_csl_abstract_macbert_seed13.yaml`
- `jasist_csl_abstract_macbert_seed42.yaml`
- `jasist_csl_abstract_macbert_seed3407.yaml`
- `jasist_csl_cls_ctg_macbert_3seed.yaml`

Earlier TNEWS, IFLYTEK, CLUE scouting, calibration, TAPT, R-Drop, and agent-search configs were part of the research history but are not part of the first paper-reproduction scope. See `docs/code_scope_for_paper.md`.

The full local experiment registry and checkpoint-level artifacts are not part of the public repository.

## Data

The included data files are small CSL-derived JSONL conversions for reproducibility and auditing. See:

- `data/csl_instruction_discipline_abstract/README.md`
- `data/csl_cls_ctg/README.md`
- `DATA_AVAILABILITY.md`

## Citation

The arXiv citation will be added after the preprint is public.
