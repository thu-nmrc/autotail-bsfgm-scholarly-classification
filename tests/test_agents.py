from pathlib import Path

import pandas as pd

from newalg.agents import (
    analyze_papers,
    analyze_sota,
    critique_ideas,
    decide_research,
    design_experiments,
    evaluate_paper_quality,
    export_submission,
    research_agents_cycle,
    scout_leaderboards,
    scout_papers,
    synthesize_ideas,
)
from newalg.config import PaperEvidence, load_task_config
from newalg.registry import RunRegistry
from newalg.utils import write_frame


def _task(tmp_path: Path):
    task = load_task_config("tests/fixtures/research_task_local.yaml")
    output_root = tmp_path / "outputs"
    return task.model_copy(
        update={
            "output_root": str(output_root),
            "registry_path": str(output_root / "runs.duckdb"),
            "artifacts_dir": str(output_root / "artifacts"),
            "reports_dir": str(output_root / "reports"),
            "method_cards_path": str(output_root / "method_cards.jsonl"),
            "proposals_dir": str(output_root / "proposals"),
            "agent_research": task.agent_research.model_copy(
                update={
                    "output_dir": str(output_root / "agents"),
                    "paper_sources": ["arxiv", "openalex"],
                    "per_leaderboard_candidate_budget": 12,
                    "min_qualified_papers": 1,
                    "require_llm_insight_review": False,
                    "require_llm_idea_review": False,
                    "min_worth_synthesizing_papers": 1,
                    "min_full_text_analyses": 0,
                }
            ),
        }
    )


def _run_row(
    run_id: str,
    dataset: str,
    kind: str,
    validation: float,
    lockbox: float,
    experiment_id: str | None = None,
) -> dict:
    experiment_id = experiment_id or run_id
    return {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dataset": dataset,
        "model_name": experiment_id,
        "model_id": experiment_id,
        "budget": "screen",
        "seed": 13,
        "kind": kind,
        "status": "completed",
        "signature": f"{dataset}|{experiment_id}",
        "method_signature": f"{experiment_id}",
        "train_seconds": 1.0,
        "device": "cpu",
        "train_rows": 10,
        "validation_accuracy": validation,
        "lockbox_accuracy": lockbox,
        "sanity_accuracy": None,
        "params_json": {},
        "metrics_json": {},
        "artifact_dir": "tests/tmp",
    }


def test_agent_cycle_dry_run_writes_structured_artifacts(tmp_path: Path, monkeypatch) -> None:
    task = _task(tmp_path)
    registry = RunRegistry(task.registry_path)

    def fake_arxiv(query: str, max_results: int):
        return [
            PaperEvidence(
                paper_id="paper-a",
                title="Prompt Label Semantics for Chinese Short Text Classification",
                source_url="https://arxiv.org/abs/2601.00001",
                source_name="arxiv",
                year=2026,
                venue="arXiv",
                abstract=(
                    "CLUE TNEWS Chinese short text classification benchmark with prompt label semantics, "
                    "calibration, public GitHub code, and state-of-the-art results that outperform BERT."
                ),
                relevance_score=0.95,
                tags=["prompt", "label_semantics", "classification", "chinese"],
            )
        ]

    def fake_openalex(query: str, max_results: int):
        return [
            PaperEvidence(
                paper_id="paper-b",
                title="Contrastive Calibration for Text Classification",
                source_url="https://example.org/paper-b",
                source_name="openalex",
                year=2025,
                venue="ACL",
                abstract=(
                    "CLUE TNEWS Chinese text classification benchmark using supervised contrastive learning, "
                    "calibration, data augmentation, public source code, and competitive leaderboard results."
                ),
                relevance_score=0.9,
                tags=["contrastive", "calibration", "classification"],
            )
        ]

    monkeypatch.setattr("newalg.agents._arxiv_papers", fake_arxiv)
    monkeypatch.setattr("newalg.agents._openalex_papers", fake_openalex)

    decision = research_agents_cycle(task, registry, "tnews", dry_run=True, max_papers=4)
    root = Path(task.agent_research.output_dir) / "tnews"

    assert decision.status.value == "active"
    assert (Path(task.agent_research.output_dir) / "leaderboard_tasks.yaml").exists()
    assert (root / "sota_snapshot.json").exists()
    assert (root / "paper_evidence.jsonl").exists()
    assert (root / "method_cards.jsonl").exists()
    assert (root / "idea_cards.reviewed.jsonl").exists()
    assert (root / "experiment_specs.yaml").exists()
    assert "experiments:" in (root / "experiment_specs.yaml").read_text(encoding="utf-8")


def test_paper_scout_records_fetch_errors(tmp_path: Path, monkeypatch) -> None:
    task = _task(tmp_path)

    def fail(query: str, max_results: int):
        raise RuntimeError("network down")

    monkeypatch.setattr("newalg.agents._arxiv_papers", fail)
    monkeypatch.setattr("newalg.agents._openalex_papers", fail)

    rows = scout_papers(task, "tnews", max_papers=4)
    assert rows
    assert all(row.error for row in rows)
    text = (Path(task.agent_research.output_dir) / "tnews" / "paper_evidence.jsonl").read_text(encoding="utf-8")
    assert "network down" in text


def test_paper_quality_rejects_irrelevant_evidence(tmp_path: Path) -> None:
    task = _task(tmp_path)
    board = scout_leaderboards(task)[0]
    papers = [
        PaperEvidence(
            paper_id="bad-paper",
            title="Thermalization in Quantum Systems",
            source_url="https://example.org/bad",
            source_name="fixture",
            year=2026,
            abstract="This paper studies quantum thermalization and elliptic surfaces.",
            tags=["text_classification"],
        ),
        PaperEvidence(
            paper_id="good-paper",
            title="Prompt Label Semantics for CLUE TNEWS Chinese Short Text Classification",
            source_url="https://example.org/good",
            source_name="fixture",
            year=2026,
            abstract=(
                "CLUE TNEWS Chinese short text classification benchmark with prompt label semantics, "
                "supervised contrastive learning, calibration, public GitHub code, and state-of-the-art "
                "leaderboard results that outperform BERT."
            ),
            tags=["prompt", "label_semantics", "classification", "chinese"],
        ),
    ]

    evaluated, report = evaluate_paper_quality(task, board, papers)
    by_id = {paper.paper_id: paper for paper in evaluated}

    assert not by_id["bad-paper"].quality_pass
    assert by_id["good-paper"].quality_pass
    assert report.status == "passed"


def test_synthesize_critique_and_design_experiments(tmp_path: Path, monkeypatch) -> None:
    task = _task(tmp_path)
    registry = RunRegistry(task.registry_path)
    scout_leaderboards(task)
    analyze_sota(task, "tnews")
    evidence_root = Path(task.agent_research.output_dir) / "tnews"
    evidence_root.mkdir(parents=True, exist_ok=True)
    (evidence_root / "paper_evidence.jsonl").write_text(
        '{"paper_id":"p1","title":"Prompt Label Semantics for CLUE TNEWS Chinese Short Text Classification","source_url":"https://example.org/p1","source_name":"fixture","year":2026,"abstract":"CLUE TNEWS Chinese short text classification benchmark with prompt label semantics, calibration, supervised contrastive learning, public GitHub code, and state-of-the-art leaderboard results that outperform BERT.","tags":["prompt","label_semantics","classification","chinese"]}\n',
        encoding="utf-8",
    )

    cards = analyze_papers(task, "tnews")
    ideas = synthesize_ideas(task, "tnews")
    reviewed = critique_ideas(task, "tnews")
    specs = design_experiments(task, registry, "tnews")

    assert cards
    assert len(ideas) >= task.agent_research.min_idea_count
    assert any(not idea.accepted for idea in reviewed)
    assert specs
    assert all(spec.dataset == "tnews" for spec in specs)


def test_research_lead_abandons_after_candidate_budget(tmp_path: Path) -> None:
    task = _task(tmp_path)
    registry = RunRegistry(task.registry_path)
    registry.upsert_run(_run_row("baseline", "tnews", "baseline", 60.0, 60.0))
    for idx in range(task.agent_research.per_leaderboard_candidate_budget):
        registry.upsert_run(_run_row(f"proposal-{idx}", "tnews", "proposal", 55.0 + idx * 0.01, 55.0))

    decision = decide_research(task, registry, "tnews")
    assert decision.status.value == "abandoned"
    assert decision.next_task_id == "iflytek"


def test_export_submission_writes_jsonl_from_predictions(tmp_path: Path) -> None:
    task = _task(tmp_path)
    registry = RunRegistry(task.registry_path)
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    write_frame(pd.DataFrame({"sample_id": [10, 11], "prediction": [2, 3]}), artifact_dir / "test_predictions")
    row = _run_row("run-export", "tnews", "proposal", 60.0, 60.0)
    row["artifact_dir"] = str(artifact_dir)
    registry.upsert_run(row)

    path = export_submission(task, "run-export", output_path=tmp_path / "submission.jsonl")
    text = path.read_text(encoding="utf-8")
    assert '{"id": 10, "label": "2"}' in text
    assert '{"id": 11, "label": "3"}' in text
