from pathlib import Path

from newalg.config import BudgetLevel, dump_yaml, load_task_config
from newalg.datasets import DatasetRepository
from newalg.pipeline import resolve_candidates
from newalg.proposer import create_method_cards, propose_experiments
from newalg.registry import RunRegistry


def test_dataset_lockbox_and_method_cards(tmp_path: Path) -> None:
    task = load_task_config("tests/fixtures/research_task_local.yaml")
    task = task.model_copy(
        update={
            "output_root": str(tmp_path / "outputs"),
            "registry_path": str(tmp_path / "outputs" / "runs.duckdb"),
            "artifacts_dir": str(tmp_path / "outputs" / "artifacts"),
            "reports_dir": str(tmp_path / "outputs" / "reports"),
            "method_cards_path": str(tmp_path / "outputs" / "method_cards.jsonl"),
            "proposals_dir": str(tmp_path / "outputs" / "proposals"),
        }
    )
    repo = DatasetRepository(task)
    prepared = repo.prepare(task.primary_dataset)

    assert len(prepared.train_df) == 12
    assert len(prepared.lockbox_df) == 3
    assert prepared.num_labels == 3

    cards = create_method_cards(task, "configs/example_papers.yaml")
    assert len(cards) >= 3

    registry = RunRegistry(task.registry_path)
    proposals = propose_experiments(task, registry, "smoke")
    assert proposals
    assert proposals[0].dataset == task.primary_dataset


def test_resolve_candidates_from_explicit_proposal_file(tmp_path: Path) -> None:
    task = load_task_config("tests/fixtures/research_task_local.yaml")
    task = task.model_copy(
        update={
            "output_root": str(tmp_path / "outputs"),
            "registry_path": str(tmp_path / "outputs" / "runs.duckdb"),
            "artifacts_dir": str(tmp_path / "outputs" / "artifacts"),
            "reports_dir": str(tmp_path / "outputs" / "reports"),
            "method_cards_path": str(tmp_path / "outputs" / "method_cards.jsonl"),
            "proposals_dir": str(tmp_path / "outputs" / "proposals"),
        }
    )
    proposal_path = tmp_path / "theory.yaml"
    dump_yaml(
        proposal_path,
        {
            "experiments": [
                {
                    "experiment_id": "theory-exp-1",
                    "dataset": task.primary_dataset,
                    "baseline_name": "tiny-bert",
                    "model_name": "tiny-bert-mean-mlp-ce_supcon",
                    "model_id": "hf-internal-testing/tiny-random-bert",
                    "pooling": "mean",
                    "head": "mlp",
                    "loss": "ce_supcon",
                    "schedule": "layerwise_lr_decay",
                    "augmentation": "none",
                    "budget": "smoke",
                    "seeds": [13],
                    "rationale": "Explicit theory-driven proposal",
                    "expected_risk": "medium",
                    "tags": ["theory_seed"],
                    "hyperparameters": {},
                }
            ]
        },
    )
    registry = RunRegistry(task.registry_path)
    candidates = resolve_candidates(task, registry, BudgetLevel.SMOKE, proposal_file=proposal_path)
    assert len(candidates) == 1
    assert candidates[0].experiment_id == "theory-exp-1"
    assert candidates[0].schedule.value == "layerwise_lr_decay"
