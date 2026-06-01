from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    AugmentationType,
    BudgetLevel,
    ExperimentSpec,
    HeadType,
    LossType,
    PoolingType,
    ResearchTaskConfig,
    ScheduleType,
    dump_yaml,
)
from .judge import judge_task, summarize_runs
from .pipeline import run_loop
from .registry import RunRegistry
from .reporting import generate_report
from .utils import ensure_dir, stable_hash


@dataclass(frozen=True)
class OptimizationRound:
    round_index: int
    proposal_path: Path
    proposed_count: int
    executed_count: int
    judge_status: str
    best_candidate: str | None


def _best_transformer_baseline(task: ResearchTaskConfig) -> tuple[str, str]:
    if not task.baselines.transformer:
        raise ValueError("auto-optimize requires at least one transformer baseline")
    baseline = task.baselines.transformer[-1]
    return baseline.name, baseline.model_id


def _completed_primary_runs(task: ResearchTaskConfig, registry: RunRegistry) -> pd.DataFrame:
    return registry.fetch_runs("dataset = ? AND status = 'completed'", [task.primary_dataset])


def _existing_signatures(registry: RunRegistry) -> set[str]:
    return registry.existing_signatures()


def _candidate_id(prefix: str, payload: dict[str, Any]) -> str:
    return stable_hash(payload, prefix=prefix)


def _spec(
    task: ResearchTaskConfig,
    budget: BudgetLevel,
    baseline_name: str,
    model_id: str,
    pooling: PoolingType,
    head: HeadType,
    loss: LossType,
    schedule: ScheduleType,
    augmentation: AugmentationType = AugmentationType.NONE,
    hyperparameters: dict[str, Any] | None = None,
    rationale: str = "",
    tags: list[str] | None = None,
) -> ExperimentSpec:
    hyperparameters = hyperparameters or {}
    payload = {
        "dataset": task.primary_dataset,
        "model_id": model_id,
        "pooling": pooling.value,
        "head": head.value,
        "loss": loss.value,
        "schedule": schedule.value,
        "augmentation": augmentation.value,
        "hyperparameters": hyperparameters,
        "budget": budget.value,
    }
    return ExperimentSpec(
        experiment_id=_candidate_id("auto-", payload),
        dataset=task.primary_dataset,
        baseline_name=baseline_name,
        model_name=f"{baseline_name}-{pooling.value}-{head.value}-{loss.value}-{schedule.value}",
        model_id=model_id,
        pooling=pooling,
        head=head,
        loss=loss,
        schedule=schedule,
        augmentation=augmentation,
        budget=budget,
        seeds=task.random_seeds[: task.budget_profile(budget).seed_count],
        rationale=rationale,
        expected_risk="low" if head == HeadType.LINEAR and augmentation == AugmentationType.NONE else "medium",
        tags=tags or ["auto_optimize"],
        hyperparameters=hyperparameters,
    )


def _promote_promising_screen_run(task: ResearchTaskConfig, registry: RunRegistry, budget: BudgetLevel) -> list[ExperimentSpec]:
    if budget != BudgetLevel.CONFIRM:
        return []
    runs = _completed_primary_runs(task, registry)
    if runs.empty:
        return []
    proposals = runs[(runs["kind"] != "baseline") & (runs["budget"] == BudgetLevel.SCREEN.value)]
    baselines = runs[runs["kind"] == "baseline"]
    if proposals.empty or baselines.empty:
        return []

    baseline = summarize_runs(baselines).iloc[0]
    best = summarize_runs(proposals).iloc[0]
    if float(best["validation_mean"]) < float(baseline["validation_mean"]) - 0.2:
        return []

    source = proposals[proposals["experiment_id"] == best["experiment_id"]].iloc[-1]
    params = source["params_json"]
    if isinstance(params, str):
        import json

        params = json.loads(params)
    spec = ExperimentSpec.model_validate(params).model_copy(
        update={
            "experiment_id": f"{best['experiment_id']}-confirm",
            "budget": BudgetLevel.CONFIRM,
            "seeds": task.random_seeds[: task.budget_profile(BudgetLevel.CONFIRM).seed_count],
            "rationale": f"Promoted from screen after reaching validation {float(best['validation_mean']):.3f}.",
            "tags": ["auto_optimize", "promoted", "confirm"],
        }
    )
    completed = runs[(runs["experiment_id"] == spec.experiment_id) & (runs["kind"] != "baseline")]
    if set(completed["seed"].astype(int).tolist()) >= set(spec.seeds):
        return []
    return [spec]


def propose_adaptive_experiments(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    budget: BudgetLevel,
    max_candidates: int | None = None,
) -> list[ExperimentSpec]:
    baseline_name, model_id = _best_transformer_baseline(task)
    existing = _existing_signatures(registry)
    candidates: list[ExperimentSpec] = []

    candidates.extend(_promote_promising_screen_run(task, registry, budget))

    # The fastest search path from current evidence is conservative recipe search
    # around CLS pooling, linear heads, label smoothing, and full fine-tuning.
    recipe_grid = [
        (
            PoolingType.CLS,
            HeadType.LABEL_SEMANTIC,
            LossType.LABEL_SMOOTHING,
            ScheduleType.FULL_FT,
            {"label_smoothing": 0.1},
            "Inject label descriptions into the classifier head, then fine-tune with the strongest clean loss recipe.",
        ),
        (
            PoolingType.CLS,
            HeadType.LINEAR,
            LossType.LABEL_SMOOTHING,
            ScheduleType.FULL_FT,
            {"label_smoothing": 0.05},
            "Tune label smoothing below the default value while keeping the strongest current structure.",
        ),
        (
            PoolingType.CLS,
            HeadType.LINEAR,
            LossType.LABEL_SMOOTHING,
            ScheduleType.FULL_FT,
            {"label_smoothing": 0.15},
            "Tune label smoothing above the default value to test whether stronger regularization helps TNEWS.",
        ),
        (
            PoolingType.CLS,
            HeadType.LINEAR,
            LossType.RDROP,
            ScheduleType.FULL_FT,
            {"rdrop_alpha": 0.2},
            "Test a lighter R-Drop penalty without changing the stable CLS plus linear backbone.",
        ),
        (
            PoolingType.CLS,
            HeadType.LINEAR,
            LossType.FOCAL,
            ScheduleType.FULL_FT,
            {"focal_gamma": 1.0},
            "Use a mild focal loss to focus confusing examples without high structural risk.",
        ),
        (
            PoolingType.CLS,
            HeadType.MLP,
            LossType.LABEL_SMOOTHING,
            ScheduleType.FULL_FT,
            {"label_smoothing": 0.1},
            "Add only a lightweight nonlinear head to the strongest current recipe.",
        ),
        (
            PoolingType.ATTENTION,
            HeadType.LINEAR,
            LossType.LABEL_SMOOTHING,
            ScheduleType.FULL_FT,
            {"label_smoothing": 0.1},
            "Test attention pooling without prototype heads or staged unfreezing.",
        ),
        (
            PoolingType.CLS,
            HeadType.LINEAR,
            LossType.LABEL_SMOOTHING,
            ScheduleType.FULL_FT,
            {"label_smoothing": 0.1},
            "Confirm the cleanest current recipe without layer-wise learning-rate decay.",
        ),
    ]

    for pooling, head, loss, schedule, hyperparameters, rationale in recipe_grid:
        candidates.append(
            _spec(
                task,
                budget,
                baseline_name,
                model_id,
                pooling,
                head,
                loss,
                schedule,
                hyperparameters=hyperparameters,
                rationale=rationale,
            )
        )

    deduped: list[ExperimentSpec] = []
    seen: set[str] = set()
    for candidate in candidates:
        is_promotion = "promoted" in candidate.tags
        if (not is_promotion and candidate.signature in existing) or candidate.signature in seen:
            continue
        seen.add(candidate.signature)
        deduped.append(candidate)

    limit = max_candidates or task.budget_profile(budget).max_candidates
    return deduped[:limit]


def write_adaptive_proposals(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    budget: BudgetLevel,
    output_path: str | Path,
    max_candidates: int | None = None,
) -> list[ExperimentSpec]:
    candidates = propose_adaptive_experiments(task, registry, budget, max_candidates=max_candidates)
    dump_yaml(output_path, {"experiments": [candidate.model_dump(mode="json") for candidate in candidates]})
    return candidates


def optimize_closed_loop(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    budget: BudgetLevel,
    rounds: int,
    top_k: int,
    dry_run: bool = False,
) -> list[OptimizationRound]:
    proposal_root = ensure_dir(task.resolve_path(task.proposals_dir) / "auto_optimize")
    summaries: list[OptimizationRound] = []
    for round_index in range(1, rounds + 1):
        proposal_path = proposal_root / f"round-{round_index}-{budget.value}.yaml"
        candidates = write_adaptive_proposals(task, registry, budget, proposal_path, max_candidates=top_k)
        executed_rows: list[dict[str, Any]] = []
        if candidates and not dry_run:
            executed_rows = run_loop(task, registry, budget, top_k, proposal_file=proposal_path, skip_existing=True)
        report = judge_task(task, registry)
        generate_report(task, registry)
        best = report.get("best_candidate", {}).get("model_name") if isinstance(report.get("best_candidate"), dict) else None
        summaries.append(
            OptimizationRound(
                round_index=round_index,
                proposal_path=proposal_path,
                proposed_count=len(candidates),
                executed_count=len(executed_rows),
                judge_status=str(report.get("status")),
                best_candidate=best,
            )
        )
        if report.get("status") == "passed" or not candidates:
            break
    return summaries
