from pathlib import Path

from newalg.config import load_task_config
from newalg.dashboard import generate_cycle_dashboard
from newalg.registry import RunRegistry


def test_cycle_dashboard_renders_agent_audit(tmp_path: Path) -> None:
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
        }
    )
    registry = RunRegistry(task.registry_path)

    path = generate_cycle_dashboard(task, registry)
    text = path.read_text(encoding="utf-8")

    assert "newALG Research Cycle Audit" in text
    assert "Literature Scout" in text
    assert "Strategy Optimizer" in text
