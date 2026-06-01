# Data Availability and Reproducibility Map

This file maps the paper's Data Availability Statement to concrete repository files.

## Public Data Used

The experiments use CSL-derived public data.

| Task | Local path | Source note |
|---|---|---|
| Abstract-to-discipline | `data/csl_instruction_discipline_abstract/` | CSL-derived abstract records converted to local JSONL splits. |
| Title-to-category | `data/csl_cls_ctg/` | Local JSONL conversion of the public CSL `cls_ctg` benchmark. |

Dataset notes and label maps are included in each dataset directory.

## Paper Tables and Figures

The paper includes generated tables and figures. Their source files are included here:

| Artifact | Path |
|---|---|
| Dataset statistics | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/dataset_statistics.csv` |
| Main result table | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/main_results_journal.csv` |
| Significance and cost table | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/significance_and_cost.csv` |
| Category-band table | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/category_delta_analysis.csv` |
| Per-seed run table | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/per_seed_runs.csv` |
| Editable figure sources | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.svg` |
| PDF figures used by LaTeX | `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.pdf` |

## Run Identifiers

Run identifiers are recorded in:

- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/artifact_manifest.md`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/per_seed_runs.csv`

The run IDs map the reported metrics to the local experiment registry.

## Local-Only Artifacts

The following files exist locally but are intentionally not included in the public GitHub repository:

| Local artifact | Reason excluded |
|---|---|
| `outputs/runs.duckdb` | Local run registry; can contain many non-paper exploratory runs. |
| `outputs/artifacts/<run-id>/model.pt` | Large model checkpoint files. |
| `outputs/artifacts/<run-id>/*.parquet` | Prediction-level artifacts; useful for audit but too large/noisy for the main repo. |
| `outputs/pretrained/` | Downloaded pretrained model cache. |
| `outputs/cache/` | Dataset/model cache. |
| `outputs/logs/` | Raw local training logs. |
| `outputs/submissions/` | Official-test prediction files; not for public release by default. |

If checkpoint-level verification is required, these artifacts can be regenerated from the recorded configurations or shared through a separate release/storage channel.

## Recommended Data Availability Statement

For the arXiv version:

> The experiments use CSL-derived public data and locally generated train, validation, lockbox, and test splits. The repository provides the paper source, generated CSV tables, figure sources, run identifiers, experiment configurations, and an artifact manifest. Model checkpoints and large prediction-level artifacts are not included in the source repository but can be regenerated from the recorded configurations.

