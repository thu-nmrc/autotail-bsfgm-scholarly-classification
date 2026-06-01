# Repository Publication Plan

Target GitHub repository:

`thu-nmrc/autotail-bsfgm-scholarly-classification`

## Publication Goal

Make the arXiv paper auditable without uploading the entire 24 GB local research workspace.

The public repository should contain:

- source code needed to inspect and rerun the method;
- experiment configs needed to reproduce the reported runs;
- paper source, tables, and editable figures;
- small CSL-derived JSONL splits and label maps;
- an artifact manifest listing run IDs and local-only artifacts.

The public repository should not contain:

- model checkpoints;
- prediction parquet files;
- pretrained model caches;
- local virtual environments;
- exploratory logs and raw outputs;
- hidden-test submission files;
- downloaded copyrighted reference PDFs;
- generated arXiv/review zip packages.

## Current Size Audit

Approximate local workspace size:

| Path | Size | Publish decision |
|---|---:|---|
| `outputs/` | 23 GB | Exclude by default. |
| `.venv/` | 1.2 GB | Exclude. |
| `backups/` | 135 MB | Exclude. |
| `data/` | 9.9 MB | Include; public CSL-derived JSONL conversions. |
| `paper/` | 8.4 MB | Include source files; exclude generated zips and local audit HTML. |
| `references/` | 4.4 MB | Exclude downloaded PDFs; keep only metadata/notes if needed. |
| `src/` | 828 KB | Include. |
| `configs/` | 584 KB | Include only paper-reproduction configs. Exclude earlier TNEWS/IFLYTEK/agent-search configs from the first public commit. |
| `tests/` | 3.6 MB | Include source tests and fixtures as repository health checks; exclude `__pycache__` and temp outputs. The tests do not rerun full paper fine-tuning experiments. |

## GitHub Remote

The local repository should point to:

```text
https://github.com/thu-nmrc/autotail-bsfgm-scholarly-classification.git
```

The repository owner is the team account. The local user can push as a collaborator.

## Before First Push

1. Accept the local Xcode/git license or install a standalone Git client. Current `/usr/bin/git` is blocked by the Xcode license prompt.
2. Confirm `.gitignore` excludes heavy/generated artifacts.
3. Run a dry file audit before staging:

```bash
find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './outputs/artifacts/*' -not -path './outputs/pretrained/*' -not -path './outputs/cache/*' -exec du -h {} + | sort -h | tail -50
```

4. Ensure no generated zip/html/audit-only files are staged.
5. Push only after reviewing the staged file list.

## Suggested First Commit Scope

Include:

- `.gitignore`
- `README.md`
- `DATA_AVAILABILITY.md`
- `docs/repository_publication_plan.md`
- `docs/code_scope_for_paper.md`
- `pyproject.toml`
- `uv.lock`
- `src/newalg/`
- `tests/`
- selected paper configs under `configs/`
- `data/csl_cls_ctg/`
- `data/csl_instruction_discipline_abstract/`
- `data/discipline_knowledge/`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/main.tex`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/references.bib`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/README.md`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/SUBMISSION_CHECKLIST.md`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/artifact_manifest.md`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.pdf`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.svg`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/*.csv`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/*.tex`

Exclude:

- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/main.pdf`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/arxiv_source.zip`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/THU-NMRC_arxiv_review_package_20260529.zip`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/reference_links_visual_20260531.html`
- `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/reference_links_20260530.md`
- `references/jasist_templates/*.pdf`
- legacy configs such as `configs/iflytek_*`, `configs/tnews_*`, `configs/csl_cls_ctg_v*`, `configs/arxiv_*`, and agent-search proposal files
- exploratory standalone scripts under `scripts/`
- all `outputs/artifacts/`, `outputs/pretrained/`, `outputs/cache/`, `outputs/logs/`, and `outputs/submissions/`

## Code-Scope Separation

The local codebase contains both the paper reproduction path and the broader automated research platform used during exploration. The first public release should make this distinction explicit:

- Paper core: `training.py`, `config.py`, `datasets.py`, `registry.py`, `judge.py`, `reporting.py`, `pipeline.py`, `utils.py`, and the selected CSL configs listed in `paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/artifact_manifest.md`.
- Supporting platform code: agent, literature, dashboard, optimizer, calibration, ensemble, inference, and pretraining modules.
- Legacy experiment history: old TNEWS, IFLYTEK, CLUE scouting, TAPT, R-Drop, calibration, and proposal configs/reports.

See `docs/code_scope_for_paper.md` before staging files.

## Git Index Cleanup Needed

The existing local `.git/index` appears to have tracked older files before the current publication filter was added. `.gitignore` will not automatically remove already tracked files.

After the local Git/Xcode license issue is fixed, run a staged-file audit and remove legacy paths from the index before pushing. The intended cleanup is:

```bash
git rm --cached -r outputs configs paper/arxiv_autotail_technical_report references || true
git add .gitignore README.md DATA_AVAILABILITY.md docs/
git add pyproject.toml uv.lock src/newalg tests
git add data/csl_instruction_discipline_abstract data/csl_cls_ctg data/discipline_knowledge
git add configs/research_task.yaml
git add configs/jasist_csl_abstract_screen_seed13.yaml
git add configs/jasist_csl_abstract_screen_seed42.yaml
git add configs/jasist_csl_abstract_confirm_seed3407.yaml
git add configs/jasist_csl_abstract_macbert_seed13.yaml
git add configs/jasist_csl_abstract_macbert_seed42.yaml
git add configs/jasist_csl_abstract_macbert_seed3407.yaml
git add configs/jasist_csl_cls_ctg_macbert_3seed.yaml
git add configs/jasist_csl_abstract_ablation_seed13.yaml
git add configs/jasist_csl_abstract_fgm_ablation_seed13.yaml
git add configs/jasist_csl_abstract_fgm_ablation_seed42.yaml
git add configs/jasist_csl_abstract_tfidf_teacher_ablation_seed13.yaml
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/main.tex
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/references.bib
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/README.md
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/SUBMISSION_CHECKLIST.md
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/artifact_manifest.md
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.pdf
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/figures/*.svg
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/*.csv
git add paper/arxiv_scholarly_autotail_bsaux_fgm_20260528/tables/*.tex
git status --short
```

Do not push until `git status --short` is reviewed and no heavy or legacy artifacts are staged.
