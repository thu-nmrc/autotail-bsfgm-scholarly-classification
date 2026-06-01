from pathlib import Path

from newalg.config import load_task_config
from newalg.datasets import DatasetRepository
from newalg.training import run_traditional_baseline


def test_traditional_baseline_runs(tmp_path: Path) -> None:
    task = load_task_config("tests/fixtures/research_task_local.yaml")
    task = task.model_copy(
        update={
            "output_root": str(tmp_path / "outputs"),
            "registry_path": str(tmp_path / "outputs" / "runs.duckdb"),
            "artifacts_dir": str(tmp_path / "outputs" / "artifacts"),
        }
    )
    repo = DatasetRepository(task)
    prepared = repo.prepare(task.primary_dataset)
    outcome = run_traditional_baseline(prepared, task, "tfidf_logreg", seed=13, artifact_root=tmp_path / "artifacts")

    assert outcome.validation_accuracy >= 0.0
    assert Path(outcome.artifact_dir).exists()
    assert Path(outcome.artifact_dir, "validation_predictions.parquet").exists() or Path(
        outcome.artifact_dir,
        "validation_predictions.csv",
    ).exists()
