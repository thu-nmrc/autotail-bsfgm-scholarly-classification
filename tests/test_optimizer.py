from pathlib import Path

from newalg.config import BudgetLevel, ExperimentSpec, load_task_config
from newalg.optimizer import propose_adaptive_experiments, write_adaptive_proposals
from newalg.registry import RunRegistry


def _row(
    run_id: str,
    experiment_id: str,
    model_name: str,
    kind: str,
    signature: str,
    validation_accuracy: float,
    lockbox_accuracy: float,
    seed: int = 13,
    budget: str = "screen",
    params_json: dict | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dataset": "tnews",
        "model_name": model_name,
        "model_id": model_name,
        "budget": budget,
        "seed": seed,
        "kind": kind,
        "status": "completed",
        "signature": signature,
        "method_signature": signature,
        "train_seconds": 10.0,
        "device": "cpu",
        "train_rows": 10,
        "validation_accuracy": validation_accuracy,
        "lockbox_accuracy": lockbox_accuracy,
        "sanity_accuracy": None,
        "params_json": params_json or {},
        "metrics_json": {},
        "artifact_dir": "tests/tmp",
    }


def _task(tmp_path: Path):
    task = load_task_config("tests/fixtures/research_task_local.yaml")
    return task.model_copy(
        update={
            "output_root": str(tmp_path / "outputs"),
            "registry_path": str(tmp_path / "outputs" / "runs.duckdb"),
            "artifacts_dir": str(tmp_path / "outputs" / "artifacts"),
            "reports_dir": str(tmp_path / "outputs" / "reports"),
            "method_cards_path": str(tmp_path / "outputs" / "method_cards.jsonl"),
            "proposals_dir": str(tmp_path / "outputs" / "proposals"),
        }
    )


def test_hyperparameters_are_part_of_non_empty_signature() -> None:
    base = {
        "experiment_id": "exp",
        "dataset": "tnews",
        "baseline_name": "tiny-bert",
        "model_name": "tiny-bert",
        "model_id": "hf-internal-testing/tiny-random-bert",
        "pooling": "cls",
        "head": "linear",
        "loss": "label_smoothing",
        "schedule": "full_ft",
        "augmentation": "none",
        "budget": "screen",
        "seeds": [13],
        "rationale": "test",
    }
    a = ExperimentSpec.model_validate({**base, "hyperparameters": {"label_smoothing": 0.05}})
    b = ExperimentSpec.model_validate({**base, "hyperparameters": {"label_smoothing": 0.15}})
    c = ExperimentSpec.model_validate({**base, "hyperparameters": {}})

    assert a.signature != b.signature
    assert "|hp=" in a.signature
    assert "|hp=" not in c.signature


def test_adaptive_optimizer_writes_candidates(tmp_path: Path) -> None:
    task = _task(tmp_path)
    registry = RunRegistry(task.registry_path)
    registry.upsert_run(
        _row(
            "b0",
            "baseline-roberta",
            "roberta",
            "baseline",
            "tnews|roberta|cls|linear|ce|full_ft|none",
            57.7,
            58.4,
            budget="final",
        )
    )

    candidates = propose_adaptive_experiments(task, registry, BudgetLevel.SCREEN, max_candidates=3)
    assert len(candidates) == 3
    assert candidates[0].pooling.value == "cls"
    assert candidates[0].head.value in {"linear", "label_semantic"}
    assert candidates[0].hyperparameters

    output = tmp_path / "adaptive.yaml"
    written = write_adaptive_proposals(task, registry, BudgetLevel.SCREEN, output, max_candidates=2)
    assert len(written) == 2
    text = output.read_text(encoding="utf-8")
    assert "experiments:" in text
    assert "label_smoothing" in text


def test_adaptive_optimizer_promotes_promising_screen_candidate(tmp_path: Path) -> None:
    task = _task(tmp_path)
    registry = RunRegistry(task.registry_path)
    registry.upsert_run(
        _row(
            "b0",
            "baseline-roberta",
            "roberta",
            "baseline",
            "tnews|roberta",
            57.7,
            58.4,
            budget="final",
        )
    )
    screen_spec = ExperimentSpec.model_validate(
        {
            "experiment_id": "screen-good",
            "dataset": task.primary_dataset,
            "baseline_name": "tiny-bert",
            "model_name": "tiny-bert-cls-linear-ls",
            "model_id": "hf-internal-testing/tiny-random-bert",
            "pooling": "cls",
            "head": "linear",
            "loss": "label_smoothing",
            "schedule": "full_ft",
            "augmentation": "none",
            "budget": "screen",
            "seeds": [13],
            "rationale": "good screen",
            "hyperparameters": {"label_smoothing": 0.1},
        }
    )
    registry.upsert_run(
        _row(
            "p0",
            screen_spec.experiment_id,
            screen_spec.model_name,
            "proposal",
            screen_spec.signature,
            57.8,
            58.2,
            params_json=screen_spec.model_dump(mode="json"),
        )
    )

    candidates = propose_adaptive_experiments(task, registry, BudgetLevel.CONFIRM, max_candidates=1)
    assert candidates
    assert candidates[0].experiment_id == "screen-good-confirm"
    assert candidates[0].budget == BudgetLevel.CONFIRM
    assert len(candidates[0].seeds) == task.budget_profile(BudgetLevel.CONFIRM).seed_count
