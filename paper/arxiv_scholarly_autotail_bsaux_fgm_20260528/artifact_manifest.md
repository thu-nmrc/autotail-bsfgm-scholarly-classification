# Artifact Manifest

Generated: 2026-05-29

This manifest records the files needed to audit the JASIST-style arXiv paper `AutoTail-BSFGM: Class-Balance-Aware Fine-Tuning for Chinese Scholarly Text Classification`.

## Paper Package

| Path | Purpose |
|---|---|
| `main.tex` | Main LaTeX manuscript |
| `references.bib` | Bibliography |
| `figures/task_overview.pdf` | Dataset/task overview figure |
| `figures/method_framework.pdf` | Method framework figure |
| `figures/main_results_journal.pdf` | Journal-style main result delta figure |
| `figures/category_delta_analysis.pdf` | Category-band analysis figure |
| `figures/task_overview.svg` | Editable source for dataset/task overview figure |
| `figures/method_framework.svg` | Editable source for method framework figure |
| `figures/main_results_journal.svg` | Editable source for main result figure with value labels |
| `figures/category_delta_analysis.svg` | Editable source for category-band figure with value labels |
| `tables/dataset_statistics.csv` | Dataset and split statistics |
| `tables/main_results_journal.csv` | Main result source table |
| `tables/significance_and_cost.csv` | Paired significance and cost source table |
| `tables/category_delta_analysis.csv` | Category-band source table |
| `tables/summary_results.csv` | Three-seed mean metrics |
| `tables/delta_results.csv` | Deltas over same-encoder label-smoothing baselines |
| `tables/per_seed_runs.csv` | Per-seed metrics, run IDs, and artifact directories |

## Source Experiment Files

| Path | Purpose |
|---|---|
| `outputs/runs.duckdb` | Local-only canonical run registry used to generate all paper tables |
| `outputs/reports/jasist_csl_abstract_screen_20260526.md` | Local-only detailed CSL abstract and cross-task report |
| `outputs/reports/jasist_csl_algorithm_branch_20260525.md` | Local-only branch-level research log and JASIST readiness assessment |
| `configs/jasist_csl_abstract_screen_seed13.yaml` | RoBERTa abstract baseline/candidate seed 13 config |
| `configs/jasist_csl_abstract_screen_seed42.yaml` | RoBERTa abstract baseline/candidate seed 42 config |
| `configs/jasist_csl_abstract_confirm_seed3407.yaml` | RoBERTa abstract baseline/candidate seed 3407 config |
| `configs/jasist_csl_abstract_macbert_seed13.yaml` | MacBERT abstract seed 13 config |
| `configs/jasist_csl_abstract_macbert_seed42.yaml` | MacBERT abstract seed 42 config |
| `configs/jasist_csl_abstract_macbert_seed3407.yaml` | MacBERT abstract seed 3407 config |
| `configs/jasist_csl_cls_ctg_macbert_3seed.yaml` | CSL title-to-category cross-task config |
| `src/newalg/` | Training runner and method implementation |
| `data/csl_instruction_discipline_abstract/` | Local abstract-to-discipline dataset files |
| `data/csl_cls_ctg/` | Local CSL title-to-category dataset files |

## Core Run IDs

| Task | Encoder | Method | Run IDs |
|---|---|---|---|
| Abstract-to-discipline | RoBERTa-WWM | LS | `run-6e65714263c8`, `run-b510c588c001`, `run-59db91087ad3` |
| Abstract-to-discipline | RoBERTa-WWM | AutoTail-BSFGM | `run-91832d1d147d`, `run-795ae42ab0cb`, `run-a49d721857a8` |
| Abstract-to-discipline | MacBERT | LS | `run-e3305599bce0`, `run-76419d93eb10`, `run-ef7de1fd8b1a` |
| Abstract-to-discipline | MacBERT | AutoTail-BSFGM | `run-6a5da81eed6e`, `run-38693b58eb5e`, `run-c39fff9d0b22` |
| Title-to-category | MacBERT | LS | `run-3345b5a1233c`, `run-79ceebc7b1f3`, `run-767cdf998ce4` |
| Title-to-category | MacBERT | AutoTail-BSFGM | `run-4c6356acd34a`, `run-6a1bcee5852e`, `run-28b6c97a2068` |

## Large Artifacts

The current local `outputs/artifacts/` directory is approximately 22 GiB. It contains model weights, prediction files, confusion matrices, and error samples. It is not required for reading the paper, but it is required for checkpoint-level verification and regenerating prediction-level significance checks.

For arXiv source upload, do not include full model checkpoints. For internal review or teacher audit, provide:

- `outputs/runs.duckdb`
- `tables/*.csv`
- relevant `outputs/artifacts/<run-id>/validation_predictions.parquet`
- relevant `outputs/artifacts/<run-id>/lockbox_predictions.parquet`
- relevant `outputs/artifacts/<run-id>/experiment_spec.json`

## Known Claim Boundary

The paper should not claim:

- official CSL hidden-test leaderboard superiority;
- universal long-tail classification improvement;
- consistent raw lockbox accuracy improvement on all tasks.

The supported claim is:

- same-encoder three-seed improvements on CSL abstract-to-discipline classification;
- validation and balanced-metric transfer on CSL title-to-category classification;
- improved class-balance-sensitive behavior with no inference-time architecture change.
