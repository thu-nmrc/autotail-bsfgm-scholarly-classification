from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Template

from .config import ResearchTaskConfig
from .judge import summarize_runs
from .registry import RunRegistry


REPORT_TEMPLATE = Template(
    """# {{ task.task_name }} Research Report

## Summary

- Primary dataset: `{{ task.primary_dataset }}`
- Sanity dataset: `{{ task.sanity_dataset }}`
- Judge status: `{{ judge.status }}`
- Validation delta: `{{ "%.3f"|format(judge.validation_delta if judge.get("validation_delta") is not none else 0.0) }}`
- Lockbox delta: `{{ "%.3f"|format(judge.lockbox_delta if judge.get("lockbox_delta") is not none else 0.0) }}`
- Cost ratio: `{{ "%.3f"|format(judge.cost_ratio if judge.get("cost_ratio") is not none else 0.0) }}`

## Best Baseline

{% if judge.get("best_baseline") %}
- Model: `{{ judge.best_baseline.model_name }}`
- Validation mean: `{{ "%.3f"|format(judge.best_baseline.validation_mean) }}`
- Lockbox mean: `{{ "%.3f"|format(judge.best_baseline.lockbox_mean) }}`
- Avg runtime: `{{ "%.2f"|format(judge.best_baseline.train_seconds_mean) }}s`
{% else %}
- No baseline results yet.
{% endif %}

## Best Candidate

{% if judge.get("best_candidate") %}
- Model: `{{ judge.best_candidate.model_name }}`
- Validation mean: `{{ "%.3f"|format(judge.best_candidate.validation_mean) }}`
- Lockbox mean: `{{ "%.3f"|format(judge.best_candidate.lockbox_mean) }}`
- Avg runtime: `{{ "%.2f"|format(judge.best_candidate.train_seconds_mean) }}s`
{% else %}
- No proposal results yet.
{% endif %}

## Ranked Experiments

| experiment_id | model_name | kind | validation_mean | lockbox_mean | seeds |
| --- | --- | --- | ---: | ---: | ---: |
{% for row in leaderboard %}
| {{ row.experiment_id }} | {{ row.model_name }} | {{ row.kind }} | {{ "%.3f"|format(row.validation_mean) }} | {{ "%.3f"|format(row.lockbox_mean or 0.0) }} | {{ row.seeds }} |
{% endfor %}

## Decision Notes

{% if judge.get("decision_reasons") %}
{% for reason in judge.decision_reasons %}
- {{ reason }}
{% endfor %}
{% else %}
- Candidate met the current success criteria.
{% endif %}
"""
)


def generate_report(task: ResearchTaskConfig, registry: RunRegistry) -> Path:
    judge_path = task.output_root_path / "judge_report.json"
    judge = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {"status": "missing"}
    runs = registry.fetch_runs("status = 'completed'")
    leaderboard = summarize_runs(runs).to_dict(orient="records") if not runs.empty else []
    report_text = REPORT_TEMPLATE.render(task=task, judge=judge, leaderboard=leaderboard)
    report_dir = task.resolve_path(task.reports_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / "latest_report.md"
    target.write_text(report_text, encoding="utf-8")
    return target
