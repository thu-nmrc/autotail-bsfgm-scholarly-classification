# AutoTail-BSFGM arXiv Submission Package

Generated: 2026-05-29

This directory contains the JASIST-style arXiv paper package for the current CSL scholarly text classification result.

## Main Claim

AutoTail-BSFGM is a single-model fine-tuning method for Chinese scholarly text classification. It improves class-balance-sensitive behavior and gives consistent gains on CSL abstract-to-discipline classification under RoBERTa-WWM and MacBERT-base.

The claim is intentionally bounded:

- Strongest evidence: CSL abstract-to-67-discipline classification.
- Cross-task evidence: official CSL title-to-category classification improves validation and balanced metrics.
- Limitation: title-to-category lockbox accuracy is slightly negative, so the paper should not claim universal accuracy improvement.

## Files

- `main.tex`: 14-page JASIST-style arXiv manuscript.
- `references.bib`: bibliography with information science, scholarly NLP, long-tail learning, and robust fine-tuning references.
- `figures/task_overview.pdf`: dataset and split overview.
- `figures/method_framework.pdf`: AutoTail-BSFGM framework.
- `figures/main_results_journal.pdf`: journal-style main result deltas.
- `figures/category_delta_analysis.pdf`: category-band analysis figure.
- `figures/*.svg`: editable vector sources for all four paper figures.
- `tables/dataset_statistics.csv`: dataset and split statistics.
- `tables/main_results_journal.csv`: journal-paper result table source.
- `tables/significance_and_cost.csv`: paired check and cost source table.
- `tables/category_delta_analysis.csv`: prediction-derived category-band analysis.
- `tables/summary_results.csv`: three-seed mean results.
- `tables/delta_results.csv`: deltas over same-encoder LS baselines.
- `tables/per_seed_runs.csv`: per-seed run IDs and metrics.
- `artifact_manifest.md`: reproducibility and audit files.
- Public companion repository: https://github.com/thu-nmrc/autotail-bsfgm-scholarly-classification

## Compile

This package was prepared for `tectonic`:

```bash
cd paper/arxiv_scholarly_autotail_bsaux_fgm_20260528
tectonic main.tex
```

If using Overleaf or a standard LaTeX installation, compile `main.tex` with BibTeX support.

## Before Uploading to arXiv

1. Confirm whether the placeholder author `THU-NMRC` should be kept or replaced.
2. Review the claim boundary in the abstract and conclusion.
3. Confirm the public companion repository remains available.
4. Upload `main.tex`, `references.bib`, the four referenced PDF figures, and the generated `tables/*.tex` files to arXiv.
5. Include CSV tables if submitting a source package with supplementary reproducibility files.
