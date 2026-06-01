# Code Scope for the AutoTail-BSFGM Paper

This repository was developed through a broader automated algorithm-research workflow. Not every local file supports the current paper directly. This document separates the paper-reproduction scope from earlier exploratory or platform code.

## Paper-Reproduction Scope

These files are needed to inspect, rerun, or audit the paper experiments.

| Path | Role in paper |
|---|---|
| `src/newalg/training.py` | Core Transformer training runner, label-smoothing baseline, AutoTail prior adjustment, Balanced-Softmax auxiliary loss, FGM adversarial training, metrics export. |
| `src/newalg/config.py` | Pydantic experiment schema used by the reported YAML configs. |
| `src/newalg/datasets.py` | Local JSONL dataset loading, validation, label-map handling, and split access. |
| `src/newalg/registry.py` | Run metadata registry used to recover run IDs, metrics, durations, and artifact paths. |
| `src/newalg/judge.py` | Metric comparison and result-judging helpers. |
| `src/newalg/reporting.py` | Report/table generation utilities used during paper packaging. |
| `src/newalg/pipeline.py` | CLI-level orchestration for running experiments and generating reports. |
| `src/newalg/utils.py` | Shared filesystem and serialization helpers. |
| `src/newalg/cli.py` | Public command entrypoint; includes both paper-relevant and broader research-platform commands. |
| `configs/jasist_csl_abstract_screen_seed13.yaml` | Abstract-to-discipline RoBERTa seed 13 baseline/candidate run. |
| `configs/jasist_csl_abstract_screen_seed42.yaml` | Abstract-to-discipline RoBERTa seed 42 baseline/candidate run. |
| `configs/jasist_csl_abstract_confirm_seed3407.yaml` | Abstract-to-discipline RoBERTa seed 3407 confirmation run. |
| `configs/jasist_csl_abstract_macbert_seed13.yaml` | Abstract-to-discipline MacBERT seed 13 run. |
| `configs/jasist_csl_abstract_macbert_seed42.yaml` | Abstract-to-discipline MacBERT seed 42 run. |
| `configs/jasist_csl_abstract_macbert_seed3407.yaml` | Abstract-to-discipline MacBERT seed 3407 run. |
| `configs/jasist_csl_cls_ctg_macbert_3seed.yaml` | CSL title-to-category MacBERT three-seed transfer run. |
| `configs/jasist_csl_abstract_ablation_seed13.yaml` | Paper ablation support. |
| `configs/jasist_csl_abstract_fgm_ablation_seed13.yaml` | FGM ablation support. |
| `configs/jasist_csl_abstract_fgm_ablation_seed42.yaml` | FGM ablation support. |
| `configs/jasist_csl_abstract_tfidf_teacher_ablation_seed13.yaml` | TF-IDF teacher ablation support. |
| `data/csl_instruction_discipline_abstract/` | Abstract-to-discipline CSL-derived local split. |
| `data/csl_cls_ctg/` | CSL title-to-category local split. |
| `data/discipline_knowledge/` | Small discipline-knowledge files used by some ablation/exploration configs. |
| `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/` | Manuscript source, references, figures, source tables, and artifact manifest. |

## Supporting but Not Central to This Paper

These modules are part of the broader automated research platform. They can remain in the repository for transparency, but they should not be described as required for reproducing the current paper unless the README explicitly says so.

| Path | Why it is secondary |
|---|---|
| `src/newalg/agents.py` | Leaderboard/paper/idea agents from the broader automated-research system. |
| `src/newalg/literature.py` | Literature discovery utilities used during idea generation, not required for rerunning reported experiments. |
| `src/newalg/llm.py` | Optional LLM integration support. |
| `src/newalg/proposer.py` | Early experiment-proposal generation. |
| `src/newalg/optimizer.py` | Closed-loop optimization utilities beyond the paper's final runs. |
| `src/newalg/dashboard.py` | Local dashboard/report visualization. |
| `src/newalg/calibration.py` | Calibration experiments from earlier branches. |
| `src/newalg/ensemble.py` | Ensemble evaluation utilities; not part of the AutoTail-BSFGM paper claim. |
| `src/newalg/inference.py` | Test-prediction export support; useful later, not central to current local validation. |
| `src/newalg/pretraining.py` | TAPT/continued-pretraining experiments; not part of the current method claim. |

## Excluded Experiment History

The local workspace contains many previous configs and reports for TNEWS, IFLYTEK, CLUE scouting, R-Drop variants, calibration, TAPT, and agent-cycle experiments. They are useful historical records, but they are not direct evidence for the current AutoTail-BSFGM scholarly-classification paper.

For the first public repository release, do not publish those legacy configs/reports by default. If later needed, add them under a clearly named `archive/` or `experiments_legacy/` directory with a separate README explaining that they are exploratory and not part of the paper's claim.

## Practical Publishing Rule

Before the first GitHub push, the staged file list should be checked against this rule:

> If a file is not needed to compile the paper, rerun the listed CSL experiments, audit the reported tables/figures, or explain the data availability statement, it should not be in the first public commit.

The only exception is lightweight platform code already imported by the package entrypoint. Such code should be labeled as supporting or legacy rather than presented as paper evidence.
