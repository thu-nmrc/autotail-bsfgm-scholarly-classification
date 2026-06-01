import json
from pathlib import Path

from newalg.config import load_task_config
from newalg.judge import judge_task
from newalg.registry import RunRegistry
from newalg.reporting import generate_report


def _make_row(
    run_id: str,
    experiment_id: str,
    dataset: str,
    model_name: str,
    kind: str,
    signature: str,
    method_signature: str,
    validation_accuracy: float,
    lockbox_accuracy: float,
    train_seconds: float,
    seed: int,
) -> dict:
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model_name": model_name,
        "model_id": model_name,
        "budget": "final",
        "seed": seed,
        "kind": kind,
        "status": "completed",
        "signature": signature,
        "method_signature": method_signature,
        "train_seconds": train_seconds,
        "device": "cpu",
        "train_rows": 10,
        "validation_accuracy": validation_accuracy,
        "lockbox_accuracy": lockbox_accuracy,
        "sanity_accuracy": None,
        "params_json": {},
        "metrics_json": {},
        "artifact_dir": "tests/tmp",
    }


def test_judge_and_report(tmp_path: Path) -> None:
    task = load_task_config("tests/fixtures/research_task_local.yaml")
    output_root = tmp_path / "outputs"
    task = task.model_copy(
        update={
            "output_root": str(output_root),
            "registry_path": str(output_root / "runs.duckdb"),
            "artifacts_dir": str(output_root / "artifacts"),
            "reports_dir": str(output_root / "reports"),
            "method_cards_path": str(output_root / "method_cards.jsonl"),
            "proposals_dir": str(output_root / "proposals"),
            "validation_delta_threshold": 0.5,
            "lockbox_delta_threshold": 0.5,
            "significance_alpha": 0.2,
            "max_cost_ratio": 2.0,
        }
    )
    registry = RunRegistry(task.registry_path)

    baseline_scores = [80.0, 79.0, 81.0]
    proposal_scores = [82.0, 83.0, 84.0]
    for idx, score in enumerate(baseline_scores):
        registry.upsert_run(
            _make_row(
                run_id=f"b{idx}",
                experiment_id="baseline-roberta",
                dataset=task.primary_dataset,
                model_name="roberta",
                kind="baseline",
                signature="tnews|roberta",
                method_signature="roberta",
                validation_accuracy=score,
                lockbox_accuracy=score - 1.0,
                train_seconds=10.0,
                seed=idx,
            )
        )
    for idx, score in enumerate(proposal_scores):
        registry.upsert_run(
            _make_row(
                run_id=f"p{idx}",
                experiment_id="proposal-a",
                dataset=task.primary_dataset,
                model_name="proposal-a",
                kind="proposal",
                signature="tnews|proposal-a",
                method_signature="proposal-a",
                validation_accuracy=score,
                lockbox_accuracy=score - 0.5,
                train_seconds=12.0,
                seed=idx,
            )
        )
    registry.upsert_run(
        _make_row(
            run_id="sb0",
            experiment_id="baseline-sanity",
            dataset=task.sanity_dataset,
            model_name="roberta",
            kind="baseline",
            signature="iflytek|roberta",
            method_signature="roberta",
            validation_accuracy=75.0,
            lockbox_accuracy=0.0,
            train_seconds=8.0,
            seed=0,
        )
    )
    registry.upsert_run(
        _make_row(
            run_id="sp0",
            experiment_id="proposal-sanity",
            dataset=task.sanity_dataset,
            model_name="proposal-a",
            kind="proposal",
            signature="iflytek|proposal-a",
            method_signature="proposal-a",
            validation_accuracy=74.5,
            lockbox_accuracy=0.0,
            train_seconds=9.0,
            seed=0,
        )
    )

    report = judge_task(task, registry)
    assert report["status"] == "passed"
    judge_file = output_root / "judge_report.json"
    assert judge_file.exists()
    assert json.loads(judge_file.read_text(encoding="utf-8"))["status"] == "passed"

    report_path = generate_report(task, registry)
    assert report_path.exists()
    assert "proposal-a" in report_path.read_text(encoding="utf-8")

