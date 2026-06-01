from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import ExperimentSpec, ResearchTaskConfig, dump_yaml, load_method_cards
from .llm import build_backend
from .registry import RunRegistry
from .utils import ensure_dir


def ingest_papers(task: ResearchTaskConfig, source_path: str | Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(Path(source_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "papers" in payload:
        return list(payload["papers"])
    if isinstance(payload, list):
        return payload
    raise ValueError("Paper source must be a list or a mapping with a 'papers' key")


def create_method_cards(task: ResearchTaskConfig, source_path: str | Path) -> list[dict[str, Any]]:
    backend = build_backend(task)
    paper_rows = ingest_papers(task, source_path)
    cards = backend.create_method_cards(task, paper_rows)
    output_path = task.resolve_path(task.method_cards_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(card.model_dump(mode="json"), ensure_ascii=False) for card in cards) + "\n",
        encoding="utf-8",
    )
    return [card.model_dump(mode="json") for card in cards]


def propose_experiments(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    budget: str,
    output_path: str | Path | None = None,
) -> list[ExperimentSpec]:
    backend = build_backend(task)
    method_cards = load_method_cards(task.method_cards_path)
    candidates = backend.propose_experiments(
        task=task,
        method_cards=method_cards,
        existing_signatures=registry.existing_signatures(),
        budget=budget,
    )
    proposal_dir = ensure_dir(task.resolve_path(task.proposals_dir))
    target = Path(output_path) if output_path else proposal_dir / f"proposals-{budget}.yaml"
    dump_yaml(target, {"experiments": [candidate.model_dump(mode="json") for candidate in candidates]})
    return candidates
