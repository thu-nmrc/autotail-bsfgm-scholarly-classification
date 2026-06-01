# CSL Category Classification Local Copy

Source: https://github.com/ydli-ai/CSL/tree/master/benchmark/cls_ctg

Paper: https://aclanthology.org/2022.coling-1.344/

This directory is a local JSONL conversion of the CSL benchmark `cls_ctg` task:

- `train.jsonl`: 8,000 rows
- `validation.jsonl`: 1,000 rows, converted from upstream `dev.tsv`
- `test.jsonl`: 1,000 rows
- `label_map.json`: source metadata, label mapping, and split-level label counts

The original TSV format is `prompt<TAB>title<TAB>category`. The local classifier input keeps only the title as `sentence`; the constant prompt is preserved as `source_prompt`.

Why this task matters here:

- It is an external benchmark relative to CLUE IFLYTEK.
- It is public and tied to a COLING 2022 dataset paper.
- It is small enough to run on the local M2 machine.
- Its train split is strongly imbalanced: 13 labels, min count 67, max count 3,536, max/min 52.78.

