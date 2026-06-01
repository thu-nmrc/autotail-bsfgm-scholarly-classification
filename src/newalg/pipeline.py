from __future__ import annotations

from pathlib import Path

from .config import BudgetLevel, ExperimentSpec, ResearchTaskConfig, load_experiment_spec
from .datasets import DatasetRepository
from .judge import judge_task
from .proposer import propose_experiments
from .registry import RunRegistry
from .training import run_traditional_baseline, run_transformer_experiment
from .utils import ensure_dir, stable_hash


def reproduce_baselines(task: ResearchTaskConfig, registry: RunRegistry) -> list[dict]:
    datasets = DatasetRepository(task)
    artifact_root = ensure_dir(task.resolve_path(task.artifacts_dir))
    rows: list[dict] = []
    for dataset_name in [task.primary_dataset, task.sanity_dataset]:
        prepared = datasets.prepare(dataset_name)
        seed_count = task.budget_profile(BudgetLevel.FINAL).seed_count if dataset_name == task.primary_dataset else 1
        for seed in task.random_seeds[:seed_count]:
            for baseline_name in task.baselines.traditional:
                outcome = run_traditional_baseline(prepared, task, baseline_name, seed, artifact_root)
                registry.upsert_run(outcome.to_row())
                rows.append(outcome.to_row())
        for baseline in task.baselines.transformer:
            spec = ExperimentSpec(
                experiment_id=f"baseline-{baseline.name}",
                dataset=dataset_name,
                baseline_name=baseline.name,
                model_name=baseline.name,
                model_id=baseline.model_id,
                pooling="cls",
                head="linear",
                loss="ce",
                schedule="full_ft",
                augmentation="none",
                budget=BudgetLevel.FINAL if dataset_name == task.primary_dataset else BudgetLevel.SCREEN,
                seeds=task.random_seeds[:seed_count],
                rationale="Baseline reproduction",
                tags=["baseline"],
            )
            for seed in spec.seeds:
                outcome = run_transformer_experiment(spec, prepared, task, seed, artifact_root, kind="baseline")
                registry.upsert_run(outcome.to_row())
                rows.append(outcome.to_row())
    return rows


def resolve_candidates(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    budget: BudgetLevel,
    proposal_file: str | Path | None = None,
) -> list[ExperimentSpec]:
    if proposal_file:
        return load_experiment_spec(proposal_file)
    return propose_experiments(task, registry, budget.value)


def run_loop(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    budget: BudgetLevel,
    top_k: int,
    proposal_file: str | Path | None = None,
    skip_existing: bool = False,
) -> list[dict]:
    datasets = DatasetRepository(task)
    artifact_root = ensure_dir(task.resolve_path(task.artifacts_dir))
    candidates = resolve_candidates(task, registry, budget, proposal_file=proposal_file)
    completed_run_ids: set[str] = set()
    if skip_existing:
        completed = registry.fetch_runs("status = 'completed'")
        if not completed.empty:
            completed_run_ids = {str(value) for value in completed["run_id"].dropna().tolist()}
    rows: list[dict] = []
    for spec in candidates[:top_k]:
        prepared_dataset = datasets.prepare(spec.dataset)
        for seed in spec.seeds:
            run_id = stable_hash(
                {"experiment": spec.experiment_id, "dataset": spec.dataset, "seed": seed},
                prefix="run-",
            )
            if skip_existing and run_id in completed_run_ids:
                continue
            outcome = run_transformer_experiment(
                spec,
                prepared_dataset,
                task,
                seed,
                artifact_root,
                kind="proposal",
            )
            registry.upsert_run(outcome.to_row())
            rows.append(outcome.to_row())

        if budget == BudgetLevel.FINAL:
            sanity_spec = spec.model_copy(
                update={
                    "experiment_id": f"{spec.experiment_id}-sanity",
                    "dataset": task.sanity_dataset,
                    "seeds": [spec.seeds[0]],
                }
            )
            prepared_sanity = datasets.prepare(task.sanity_dataset)
            sanity_outcome = run_transformer_experiment(
                sanity_spec,
                prepared_sanity,
                task,
                sanity_spec.seeds[0],
                artifact_root,
                kind="proposal",
            )
            registry.upsert_run(sanity_outcome.to_row())
            rows.append(sanity_outcome.to_row())

    return rows


def judge_and_report(task: ResearchTaskConfig, registry: RunRegistry) -> tuple[dict, Path]:
    report = judge_task(task, registry)
    from .reporting import generate_report

    report_path = generate_report(task, registry)
    return report, report_path
