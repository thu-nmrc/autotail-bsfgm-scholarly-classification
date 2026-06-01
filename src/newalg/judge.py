from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy import stats

from .config import ResearchTaskConfig
from .registry import RunRegistry


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["experiment_id", "model_name", "signature", "method_signature", "kind"], dropna=False)
        .agg(
            validation_mean=("validation_accuracy", "mean"),
            validation_std=("validation_accuracy", "std"),
            lockbox_mean=("lockbox_accuracy", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    grouped["validation_std"] = grouped["validation_std"].fillna(0.0)
    return grouped.sort_values("validation_mean", ascending=False).reset_index(drop=True)


def _extract_scores(frame: pd.DataFrame, signature: str) -> list[float]:
    subset = frame.loc[frame["signature"] == signature, "validation_accuracy"]
    return [float(value) for value in subset.tolist()]


def judge_task(task: ResearchTaskConfig, registry: RunRegistry) -> dict[str, Any]:
    primary_runs = registry.fetch_runs("dataset = ? AND status = 'completed'", [task.primary_dataset])
    if primary_runs.empty:
        report = {"status": "no_runs", "reason": "No completed runs for primary dataset"}
        _write_judge_report(task, report)
        return report

    baselines = primary_runs.loc[primary_runs["kind"] == "baseline"].copy()
    proposals = primary_runs.loc[primary_runs["kind"] != "baseline"].copy()
    if baselines.empty or proposals.empty:
        report = {"status": "insufficient_data", "reason": "Need both baseline and proposal runs to judge"}
        _write_judge_report(task, report)
        return report

    baseline_summary = summarize_runs(baselines).iloc[0]
    proposal_summary = summarize_runs(proposals).iloc[0]
    baseline_runs = baselines.loc[baselines["signature"] == baseline_summary["signature"]]
    proposal_runs = proposals.loc[proposals["signature"] == proposal_summary["signature"]]

    baseline_scores = baseline_runs["validation_accuracy"].astype(float).tolist()
    proposal_scores = proposal_runs["validation_accuracy"].astype(float).tolist()
    ttest = stats.ttest_ind(proposal_scores, baseline_scores, equal_var=False)
    validation_delta = float(proposal_summary["validation_mean"] - baseline_summary["validation_mean"])
    proposal_lockbox = 0.0 if pd.isna(proposal_summary["lockbox_mean"]) else float(proposal_summary["lockbox_mean"])
    baseline_lockbox = 0.0 if pd.isna(baseline_summary["lockbox_mean"]) else float(baseline_summary["lockbox_mean"])
    lockbox_delta = proposal_lockbox - baseline_lockbox
    cost_ratio = float(proposal_summary["train_seconds_mean"] / max(baseline_summary["train_seconds_mean"], 1e-6))

    sanity_runs = registry.fetch_runs(
        "dataset = ? AND status = 'completed' AND kind != 'baseline' AND method_signature = ?",
        [task.sanity_dataset, proposal_summary["method_signature"]],
    )
    sanity_baselines = registry.fetch_runs("dataset = ? AND status = 'completed' AND kind = 'baseline'", [task.sanity_dataset])
    sanity_delta = None
    sanity_ok = True
    if not sanity_runs.empty and not sanity_baselines.empty:
        sanity_candidate = summarize_runs(sanity_runs).iloc[0]
        sanity_baseline = summarize_runs(sanity_baselines).iloc[0]
        sanity_delta = float(sanity_candidate["validation_mean"] - sanity_baseline["validation_mean"])
        sanity_ok = sanity_delta >= -1.0

    passed = (
        validation_delta >= task.validation_delta_threshold
        and lockbox_delta >= task.lockbox_delta_threshold
        and float(ttest.pvalue) <= task.significance_alpha
        and cost_ratio <= task.max_cost_ratio
        and sanity_ok
    )

    report = {
        "status": "passed" if passed else "failed",
        "primary_dataset": task.primary_dataset,
        "best_baseline": {
            "model_name": baseline_summary["model_name"],
            "validation_mean": float(baseline_summary["validation_mean"]),
            "lockbox_mean": baseline_lockbox,
            "train_seconds_mean": float(baseline_summary["train_seconds_mean"]),
        },
        "best_candidate": {
            "model_name": proposal_summary["model_name"],
            "validation_mean": float(proposal_summary["validation_mean"]),
            "lockbox_mean": proposal_lockbox,
            "train_seconds_mean": float(proposal_summary["train_seconds_mean"]),
        },
        "validation_delta": validation_delta,
        "lockbox_delta": lockbox_delta,
        "cost_ratio": cost_ratio,
        "p_value": float(ttest.pvalue),
        "sanity_dataset": task.sanity_dataset,
        "sanity_delta": sanity_delta,
        "thresholds": {
            "validation_delta": task.validation_delta_threshold,
            "lockbox_delta": task.lockbox_delta_threshold,
            "alpha": task.significance_alpha,
            "max_cost_ratio": task.max_cost_ratio,
        },
        "decision_reasons": [
            reason
            for passed_flag, reason in [
                (validation_delta >= task.validation_delta_threshold, "validation delta below threshold"),
                (lockbox_delta >= task.lockbox_delta_threshold, "lockbox delta below threshold"),
                (float(ttest.pvalue) <= task.significance_alpha, "significance not reached"),
                (cost_ratio <= task.max_cost_ratio, "cost ratio too high"),
                (sanity_ok, "sanity dataset collapse"),
            ]
            if not passed_flag
        ],
    }
    _write_judge_report(task, report)
    return report


def _write_judge_report(task: ResearchTaskConfig, report: dict[str, Any]) -> None:
    path = task.output_root_path / "judge_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
