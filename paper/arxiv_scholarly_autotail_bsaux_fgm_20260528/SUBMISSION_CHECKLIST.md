# arXiv Submission Checklist

## Required Before Upload

- [ ] Confirm whether the placeholder author `THU-NMRC` should be kept or replaced.
- [ ] Decide whether to list affiliation, email, or acknowledgements.
- [ ] Confirm companion repository URL is public: `https://github.com/thu-nmrc/autotail-bsfgm-scholarly-classification`.
- [ ] Recompile after author edits: `tectonic main.tex`.
- [ ] Review `main.pdf` visually once before upload.

## Suggested arXiv Metadata

- Title: `AutoTail-BSFGM: Class-Balance-Aware Fine-Tuning for Chinese Scholarly Text Classification`
- Primary category: `cs.CL`
- Secondary category: optional `cs.IR`
- Comment: `14 pages, 4 figures`

## Upload Files

Use `arxiv_source.zip` for arXiv source upload. It contains:

- `main.tex`
- `references.bib`
- `figures/task_overview.pdf`
- `figures/method_framework.pdf`
- `figures/main_results_journal.pdf`
- `figures/category_delta_analysis.pdf`
- `tables/dataset_statistics.tex`
- `tables/main_results_journal.tex`
- `tables/significance_and_cost.tex`
- `tables/category_delta_analysis.tex`

Do not upload model checkpoints to arXiv.

The manuscript includes the public companion repository as the code and data link.

Editable SVG sources are included in the teacher review package, but the arXiv source package uses PDF figures for safer LaTeX compilation.

## Optional Supplementary/Internal Files

Keep these for teacher/auditor review:

- `main.pdf`
- `tables/*.csv`
- `artifact_manifest.md`
- `README.md`
- `outputs/runs.duckdb`
- relevant `outputs/artifacts/<run-id>/validation_predictions.parquet`
- relevant `outputs/artifacts/<run-id>/lockbox_predictions.parquet`

## Claim Boundary To Preserve

Safe claim:

> AutoTail-BSFGM improves Chinese scholarly abstract classification across two base-size encoders and improves class-balance-sensitive metrics on a second CSL title-classification task.

Do not claim:

- official CSL leaderboard SOTA;
- universal accuracy improvement;
- lockbox accuracy improvement on every task;
- a new pretrained model.
