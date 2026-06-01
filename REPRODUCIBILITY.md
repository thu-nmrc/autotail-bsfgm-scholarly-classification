# Reproducibility Guide

This repository is the public companion repository for the arXiv paper:

**AutoTail-BSFGM: Class-Balance-Aware Fine-Tuning for Chinese Scholarly Text Classification**

It is designed to make the paper auditable and rerunnable without publishing local checkpoints, raw logs, or the broader exploratory research history.

## What Is Included

The repository includes:

| Item | Location |
|---|---|
| Paper source | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/main.tex` |
| Bibliography | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/references.bib` |
| Source result tables | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/*.csv` |
| Editable figures | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.svg` |
| Experiment configs | `configs/jasist_csl_*.yaml` |
| Local CSL-derived data splits | `data/csl_instruction_discipline_abstract/`, `data/csl_cls_ctg/` |
| Core implementation | `src/newalg/training.py`, `src/newalg/config.py`, `src/newalg/datasets.py` |
| Run and artifact map | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/artifact_manifest.md` |

## Dataset Splits

The public repository contains the exact lightweight JSONL splits used by the paper package.

| Task | Input | Labels | Train | Validation | Lockbox | Test |
|---|---|---:|---:|---:|---:|---:|
| CSL abstract-to-discipline | Scientific abstract | 67 | 8,640 | 1,200 | 960 | 1,200 |
| CSL title-to-category | Scientific title | 13 | 7,200 | 1,000 | 800 | 1,000 |

The physical JSONL files contain the full local train split before the deterministic lockbox cut:

| Directory | Train JSONL | Validation JSONL | Test JSONL |
|---|---:|---:|---:|
| `data/csl_instruction_discipline_abstract/` | 9,600 | 1,200 | 1,200 |
| `data/csl_cls_ctg/` | 8,000 | 1,000 | 1,000 |

The runner creates the lockbox split from the train file using `lockbox_fraction: 0.1` and `lockbox_seed: 3407` in `configs/research_task.yaml`.

## Environment

Use Python 3.11 through `uv`:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Run the test suite:

```bash
uv run pytest
```

The tests use toy fixtures and small local artifacts to check that the repository code paths are healthy. They cover dataset loading, lockbox/method-card handling, judging and report generation, traditional-baseline execution, offline literature discovery, proposal/agent utilities, optimizer bookkeeping, and dashboard rendering. They do not train the reported Transformer models and do not reproduce the paper tables by themselves; paper-level reproduction requires running the selected configs below.

Compile the paper:

```bash
cd paper/arxiv_scholarly_autotail_bsaux_fgm_20260528
tectonic main.tex
```

## Rerunning Paper Experiments

The paper compares same-encoder label-smoothing baselines against AutoTail-BSFGM candidates. The main configs are:

| Purpose | Config |
|---|---|
| RoBERTa abstract seed 13 | `configs/jasist_csl_abstract_screen_seed13.yaml` |
| RoBERTa abstract seed 42 | `configs/jasist_csl_abstract_screen_seed42.yaml` |
| RoBERTa abstract seed 3407 | `configs/jasist_csl_abstract_confirm_seed3407.yaml` |
| MacBERT abstract seed 13 | `configs/jasist_csl_abstract_macbert_seed13.yaml` |
| MacBERT abstract seed 42 | `configs/jasist_csl_abstract_macbert_seed42.yaml` |
| MacBERT abstract seed 3407 | `configs/jasist_csl_abstract_macbert_seed3407.yaml` |
| MacBERT title-to-category three-seed check | `configs/jasist_csl_cls_ctg_macbert_3seed.yaml` |

Example command:

```bash
uv run newalg run-loop \
  --task configs/research_task.yaml \
  --budget screen \
  --proposal-file configs/jasist_csl_abstract_screen_seed13.yaml \
  --top-k 2
```

This command runs the label-smoothing baseline and the AutoTail-BSFGM candidate in that config. Full three-seed reproduction requires running the listed configs and can be time-consuming on CPU or Apple MPS.

## What Is Not Included

The repository intentionally excludes:

| Excluded item | Reason |
|---|---|
| `outputs/artifacts/<run-id>/model.pt` | Large model checkpoints. |
| `outputs/artifacts/<run-id>/*.parquet` | Prediction-level artifacts; useful but too large/noisy for the main repo. |
| `outputs/runs.duckdb` | Local run registry contains broader exploratory history. |
| `outputs/pretrained/` | Downloaded model cache. |
| `outputs/logs/` | Raw training logs. |

The paper tables are included as CSV/TEX artifacts. Checkpoint-level verification can be regenerated from the listed configs or shared separately through institutional storage if required.

## Expected Reproduction Boundary

This repository supports:

- auditing the exact paper source, figures, tables, and references;
- inspecting the AutoTail-BSFGM implementation;
- rerunning the selected CSL configs;
- rebuilding the paper PDF;
- checking the claim boundary and run IDs.

It does not claim:

- official hidden-test leaderboard state of the art;
- universal long-tail improvement on all datasets;
- full preservation of every exploratory run conducted before the paper package was curated.
