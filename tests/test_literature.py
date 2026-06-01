from pathlib import Path

from newalg.config import load_task_config
from newalg.literature import discover_literature


def test_discover_literature_offline_curated(tmp_path: Path) -> None:
    task = load_task_config("tests/fixtures/research_task_local.yaml")
    output = tmp_path / "papers.yaml"
    papers = discover_literature(task, output, max_papers=4, use_arxiv=False)

    assert output.exists()
    assert len(papers) >= 4
    assert any("label" in " ".join(paper.get("tags", [])) for paper in papers)
    assert "papers:" in output.read_text(encoding="utf-8")
