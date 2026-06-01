from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from .config import ResearchTaskConfig, load_experiment_spec
from .judge import summarize_runs
from .registry import RunRegistry


@dataclass
class AgentAudit:
    name: str
    role: str
    status: str
    score: int
    evidence: list[str]
    risks: list[str]
    artifact: str | None = None


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest(path_root: Path, pattern: str) -> Path | None:
    matches = sorted(path_root.glob(pattern), key=lambda path: path.stat().st_mtime)
    return matches[-1] if matches else None


def _score(completed: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(100, round(100 * completed / total)))


def audit_agents(task: ResearchTaskConfig, registry: RunRegistry) -> dict[str, Any]:
    output_root = task.resolve_path(task.output_root)
    agent_root = task.resolve_path(task.agent_research.output_dir)
    active_task = task.primary_dataset
    agent_task_root = agent_root / active_task
    leaderboard_path = agent_root / "leaderboard_tasks.yaml"
    sota_path = agent_task_root / "sota_snapshot.json"
    paper_evidence_path = agent_task_root / "paper_evidence.jsonl"
    paper_quality_path = agent_task_root / "paper_quality_report.json"
    agent_method_cards_path = agent_task_root / "method_cards.jsonl"
    paper_analysis_path = agent_task_root / "paper_analysis.jsonl"
    idea_path = agent_task_root / "idea_cards.jsonl"
    reviewed_idea_path = agent_task_root / "idea_cards.reviewed.jsonl"
    idea_quality_path = agent_task_root / "idea_quality_reviews.jsonl"
    agent_specs_path = agent_task_root / "experiment_specs.yaml"
    decision_path = agent_task_root / "research_decision.json"

    literature_path = _latest(output_root / "literature", "*.yaml")
    literature_payload = _read_yaml(literature_path) if literature_path else None
    papers = list((literature_payload or {}).get("papers", [])) if isinstance(literature_payload, dict) else []
    paper_evidence = _read_jsonl(paper_evidence_path)
    paper_quality = json.loads(paper_quality_path.read_text(encoding="utf-8")) if paper_quality_path.exists() else {}
    paper_errors = [row for row in paper_evidence if row.get("error")]
    qualified_papers = [row for row in paper_evidence if row.get("quality_pass")]
    method_cards = _read_jsonl(task.resolve_path(task.method_cards_path))
    agent_method_cards = _read_jsonl(agent_method_cards_path)
    paper_analyses = _read_jsonl(paper_analysis_path)
    ideas = _read_jsonl(idea_path)
    reviewed_ideas = _read_jsonl(reviewed_idea_path)
    idea_quality_reviews = _read_jsonl(idea_quality_path)
    accepted_ideas = [row for row in reviewed_ideas if row.get("accepted")]
    zero_start_path = task.resolve_path(task.proposals_dir) / "zero-start-screen.yaml"
    adaptive_path = _latest(task.resolve_path(task.proposals_dir) / "auto_optimize", "*.yaml")

    zero_specs = load_experiment_spec(zero_start_path) if zero_start_path.exists() else []
    adaptive_specs = load_experiment_spec(adaptive_path) if adaptive_path and adaptive_path.exists() else []
    agent_specs = load_experiment_spec(agent_specs_path) if agent_specs_path.exists() else []
    runs = registry.fetch_runs("status = 'completed'")
    summaries = summarize_runs(runs).to_dict(orient="records") if not runs.empty else []
    baselines = runs[runs["kind"] == "baseline"] if not runs.empty else runs
    proposals = runs[runs["kind"] != "baseline"] if not runs.empty else runs
    judge_path = output_root / "judge_report.json"
    judge = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path.exists() else {"status": "missing"}
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}

    paper_tags = {tag for paper in papers for tag in paper.get("tags", [])}
    evidence_tags = {tag for paper in paper_evidence for tag in paper.get("tags", [])}
    zero_heads = {spec.head.value for spec in zero_specs}
    zero_losses = {spec.loss.value for spec in zero_specs}
    adaptive_hps = [spec.hyperparameters for spec in adaptive_specs if spec.hyperparameters]

    agents = [
        AgentAudit(
            name="Leaderboard Scout",
            role="Maintain CLUE task pool, target metrics, and submission assumptions.",
            status="ready" if leaderboard_path.exists() else "missing",
            score=_score(int(leaderboard_path.exists()) + int(bool(task.primary_dataset)) + int(bool(task.sanity_dataset)), 3),
            evidence=[
                f"leaderboard_file={leaderboard_path.exists()}",
                f"primary={task.primary_dataset}",
                f"sanity={task.sanity_dataset}",
            ],
            risks=[] if leaderboard_path.exists() else ["No leaderboard task file generated yet."],
            artifact=str(leaderboard_path) if leaderboard_path.exists() else None,
        ),
        AgentAudit(
            name="SOTA Analyst",
            role="Record official benchmark target and separate leaderboard claims from local validation.",
            status="ready" if sota_path.exists() else "missing",
            score=_score(int(sota_path.exists()) + int(bool(decision.get("task_id"))) + int(judge.get("status") in {"passed", "failed", "missing"}), 3),
            evidence=[
                f"sota_snapshot={sota_path.exists()}",
                f"decision_task={decision.get('task_id', 'none')}",
                f"judge_status={judge.get('status', 'missing')}",
            ],
            risks=[] if sota_path.exists() else ["No SOTA snapshot generated yet."],
            artifact=str(sota_path) if sota_path.exists() else None,
        ),
        AgentAudit(
            name="Paper Scout / Literature Scout",
            role="Collect real paper evidence, score quantity/relevance/transferability, and preserve source/errors for audit.",
            status="ready" if paper_evidence else ("ready" if papers else "missing"),
            score=_score(
                int(paper_quality.get("status") == "passed")
                + int(len(qualified_papers) >= task.agent_research.min_qualified_papers)
                + int(paper_evidence_path.exists() and not paper_errors),
                3,
            ),
            evidence=[
                f"paper_evidence={len(paper_evidence)}",
                f"qualified={paper_quality.get('qualified_papers', len(qualified_papers))}/{paper_quality.get('min_required', task.agent_research.min_qualified_papers)}",
                f"avg_quality={paper_quality.get('average_quality_score', 'n/a')}",
                f"errors={len(paper_errors)}",
                f"tags={', '.join(sorted(evidence_tags or paper_tags)[:8]) or 'none'}",
            ],
            risks=(
                list(paper_quality.get("failure_reasons", []))[:3]
                or ([str(row.get("error")) for row in paper_errors[:3]] if paper_errors else [])
                or ([] if paper_evidence or papers else ["No paper evidence found."])
            ),
            artifact=str(paper_quality_path) if paper_quality_path.exists() else (str(paper_evidence_path) if paper_evidence_path.exists() else (str(literature_path) if literature_path else None)),
        ),
        AgentAudit(
            name="Paper Analyst",
            role="Extract algorithm/model/theory details from qualified paper evidence before method cards.",
            status="ready" if paper_analyses else ("ready" if agent_method_cards or method_cards else "incomplete"),
            score=_score(
                int(bool(paper_analyses))
                + int(any(row.get("worth_synthesizing") for row in paper_analyses))
                + int(any(row.get("analysis_depth") == "full_text_pdf" for row in paper_analyses)),
                3,
            ),
            evidence=[
                f"paper_analyses={len(paper_analyses)}",
                f"full_text={sum(1 for row in paper_analyses if row.get('analysis_depth') == 'full_text_pdf')}",
                f"worth_synthesizing={sum(1 for row in paper_analyses if row.get('worth_synthesizing'))}",
                f"method_cards={len(agent_method_cards or method_cards)}",
            ],
            risks=[] if paper_analyses else ["No detailed paper analyses generated."],
            artifact=str(paper_analysis_path) if paper_analysis_path.exists() else (str(agent_method_cards_path) if agent_method_cards_path.exists() else (str(task.resolve_path(task.method_cards_path)) if method_cards else None)),
        ),
        AgentAudit(
            name="Idea Synthesizer",
            role="Generate innovation cards with paper traceability and baseline differences.",
            status="ready" if ideas else "missing",
            score=_score(int(len(ideas) >= task.agent_research.min_idea_count) + int(any(row.get("source_paper_ids") for row in ideas)) + int(any(row.get("difference_from_baseline") for row in ideas)), 3),
            evidence=[
                f"ideas={len(ideas)}",
                f"min_required={task.agent_research.min_idea_count}",
                f"with_sources={sum(1 for row in ideas if row.get('source_paper_ids'))}",
            ],
            risks=[] if ideas else ["No idea cards generated."],
            artifact=str(idea_path) if idea_path.exists() else None,
        ),
        AgentAudit(
            name="Critic",
            role="Reject pseudo-innovation with LLM-first idea quality review and required ablations.",
            status="ready" if reviewed_ideas else "missing",
            score=_score(
                int(bool(idea_quality_reviews))
                + int(bool(accepted_ideas))
                + int(any(not row.get("accepted") for row in reviewed_ideas)),
                3,
            ),
            evidence=[
                f"idea_quality_reviews={len(idea_quality_reviews)}",
                f"reviewed={len(reviewed_ideas)}",
                f"accepted={len(accepted_ideas)}",
                f"rejected={len(reviewed_ideas) - len(accepted_ideas)}",
            ],
            risks=[] if accepted_ideas else ["No accepted ideas yet."],
            artifact=str(idea_quality_path) if idea_quality_path.exists() else (str(reviewed_idea_path) if reviewed_idea_path.exists() else None),
        ),
        AgentAudit(
            name="Experiment Designer",
            role="Validate specs, budgets, seeds, modules, and novelty before training.",
            status="ready" if agent_specs or zero_specs or adaptive_specs else "missing",
            score=_score(
                int(all(spec.dataset == task.primary_dataset for spec in agent_specs + zero_specs + adaptive_specs))
                + int(all(spec.seeds for spec in agent_specs + zero_specs + adaptive_specs))
                + int(len({spec.signature for spec in agent_specs + zero_specs + adaptive_specs}) == len(agent_specs + zero_specs + adaptive_specs)),
                3,
            ),
            evidence=[
                f"agent_specs={len(agent_specs)}",
                f"legacy_specs={len(zero_specs) + len(adaptive_specs)}",
                f"unique_signatures={len({spec.signature for spec in agent_specs + zero_specs + adaptive_specs})}",
            ],
            risks=[] if agent_specs or zero_specs or adaptive_specs else ["No proposal specs found."],
            artifact=str(agent_specs_path) if agent_specs_path.exists() else (str(adaptive_path) if adaptive_path else None),
        ),
        AgentAudit(
            name="Runner",
            role="Train candidates, evaluate validation and lockbox, and store artifacts.",
            status="ready" if not proposals.empty else "waiting",
            score=_score(
                int(not baselines.empty) + int(not proposals.empty) + int("artifact_dir" in runs.columns and runs["artifact_dir"].notna().any()),
                3,
            ),
            evidence=[
                f"baseline_runs={len(baselines)}",
                f"proposal_runs={len(proposals)}",
                f"completed_runs={len(runs)}",
            ],
            risks=[] if not proposals.empty else ["No completed proposal runs yet for this cycle."],
            artifact=str(task.resolve_path(task.artifacts_dir)),
        ),
        AgentAudit(
            name="Judge",
            role="Compare best candidate against the strongest baseline with thresholds.",
            status=str(judge.get("status", "missing")),
            score=_score(
                int(judge.get("best_baseline") is not None)
                + int(judge.get("best_candidate") is not None)
                + int(judge.get("status") in {"passed", "failed"}),
                3,
            ),
            evidence=[
                f"status={judge.get('status', 'missing')}",
                f"validation_delta={judge.get('validation_delta', 'n/a')}",
                f"lockbox_delta={judge.get('lockbox_delta', 'n/a')}",
            ],
            risks=list(judge.get("decision_reasons", [])) if isinstance(judge.get("decision_reasons"), list) else [],
            artifact=str(judge_path) if judge_path.exists() else None,
        ),
        AgentAudit(
            name="Research Lead / Strategy Optimizer",
            role="Decide continue, promote, abandon, switch leaderboard, or prepare submission.",
            status=str(decision.get("status", "missing")),
            score=_score(
                int(bool(decision)) + int(decision.get("candidate_budget") == task.agent_research.per_leaderboard_candidate_budget) + int(decision.get("status") in {"active", "promising", "abandoned", "ready_for_submission"}),
                3,
            ),
            evidence=[
                f"tested={decision.get('tested_candidates', 'n/a')}",
                f"budget={decision.get('candidate_budget', task.agent_research.per_leaderboard_candidate_budget)}",
                f"next={decision.get('next_task_id', 'none')}",
            ],
            risks=[decision.get("reason", "No research decision generated.")] if not decision else [],
            artifact=str(decision_path) if decision_path.exists() else None,
        ),
    ]

    return {
        "agents": agents,
        "summaries": summaries,
        "judge": judge,
        "zero_start_path": str(zero_start_path) if zero_start_path.exists() else None,
        "adaptive_path": str(adaptive_path) if adaptive_path else None,
        "literature_path": str(literature_path) if literature_path else None,
    }


DASHBOARD_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>newALG Research Cycle Audit</title>
  <style>
    :root {
      --paper: #f7f6f1;
      --panel: #ffffff;
      --ink: #20231f;
      --muted: #686d63;
      --rule: rgba(32, 35, 31, 0.12);
      --good: #1b7f5a;
      --warn: #9c6b14;
      --bad: #b24335;
      --accent: #275d8c;
      --code: #eef1eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, "Avenir Next", "Helvetica Neue", Arial, sans-serif;
      background: var(--paper);
      color: var(--ink);
      line-height: 1.45;
    }
    header {
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--rule);
      background: linear-gradient(180deg, #fbfaf6, #f1f2ea);
    }
    h1 { margin: 0; font-size: 28px; font-weight: 750; letter-spacing: 0; }
    .meta { margin-top: 6px; color: var(--muted); font-size: 13px; }
    main { padding: 24px 32px 40px; max-width: 1320px; margin: 0 auto; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--rule);
      border-radius: 8px;
      padding: 16px;
    }
    .agent-head { display: flex; align-items: start; justify-content: space-between; gap: 12px; }
    h2 { margin: 0 0 4px; font-size: 16px; }
    .role { color: var(--muted); font-size: 12px; min-height: 34px; }
    .score {
      width: 52px; height: 52px; border-radius: 50%;
      display: grid; place-items: center;
      border: 6px solid var(--rule);
      font-weight: 800; font-size: 14px;
      flex: 0 0 auto;
    }
    .score.good { border-color: rgba(27, 127, 90, .35); color: var(--good); }
    .score.warn { border-color: rgba(156, 107, 20, .35); color: var(--warn); }
    .score.bad { border-color: rgba(178, 67, 53, .35); color: var(--bad); }
    .status { display: inline-block; margin-top: 8px; padding: 3px 8px; border-radius: 999px; background: var(--code); font-size: 12px; }
    ul { padding-left: 18px; margin: 12px 0 0; }
    li { margin: 4px 0; font-size: 13px; }
    .risks li { color: var(--bad); }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code { background: var(--code); padding: 1px 5px; border-radius: 4px; }
    table { width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--rule); text-align: left; font-size: 13px; }
    th { background: #ecefe7; font-size: 12px; color: var(--muted); }
    .section-title { margin: 28px 0 12px; font-size: 18px; }
    .path { overflow-wrap: anywhere; }
    @media (max-width: 980px) { .grid { grid-template-columns: 1fr; } main, header { padding-left: 18px; padding-right: 18px; } }
  </style>
</head>
<body>
  <header>
    <h1>newALG Research Cycle Audit</h1>
    <div class="meta">Task <code>{{ task.task_name }}</code> · primary <code>{{ task.primary_dataset }}</code> · sanity <code>{{ task.sanity_dataset }}</code></div>
  </header>
  <main>
    <div class="grid">
      {% for agent in agents %}
      <section class="card">
        <div class="agent-head">
          <div>
            <h2>{{ agent.name }}</h2>
            <div class="role">{{ agent.role }}</div>
            <span class="status">{{ agent.status }}</span>
          </div>
          <div class="score {% if agent.score >= 80 %}good{% elif agent.score >= 50 %}warn{% else %}bad{% endif %}">{{ agent.score }}</div>
        </div>
        <ul>
          {% for item in agent.evidence %}
          <li><code>{{ item }}</code></li>
          {% endfor %}
        </ul>
        {% if agent.risks %}
        <ul class="risks">
          {% for risk in agent.risks %}
          <li>{{ risk }}</li>
          {% endfor %}
        </ul>
        {% endif %}
        {% if agent.artifact %}
        <p class="path"><a href="../{{ agent.artifact }}">{{ agent.artifact }}</a></p>
        {% endif %}
      </section>
      {% endfor %}
    </div>

    <h2 class="section-title">Judge Snapshot</h2>
    <section class="card">
      <p>Status: <code>{{ judge.status }}</code></p>
      <p>Validation delta: <code>{{ judge.get("validation_delta", "n/a") }}</code> · Lockbox delta: <code>{{ judge.get("lockbox_delta", "n/a") }}</code> · Cost ratio: <code>{{ judge.get("cost_ratio", "n/a") }}</code></p>
    </section>

    <h2 class="section-title">Completed Experiment Leaderboard</h2>
    <table>
      <thead>
        <tr>
          <th>experiment</th>
          <th>model</th>
          <th>kind</th>
          <th>validation</th>
          <th>lockbox</th>
          <th>seeds</th>
        </tr>
      </thead>
      <tbody>
        {% for row in summaries[:18] %}
        <tr>
          <td><code>{{ row.experiment_id }}</code></td>
          <td>{{ row.model_name }}</td>
          <td>{{ row.kind }}</td>
          <td>{{ "%.3f"|format(row.validation_mean) }}</td>
          <td>{{ "%.3f"|format(row.lockbox_mean or 0) }}</td>
          <td>{{ row.seeds }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </main>
</body>
</html>
"""
)


def generate_cycle_dashboard(task: ResearchTaskConfig, registry: RunRegistry) -> Path:
    audit = audit_agents(task, registry)
    target = task.resolve_path(task.reports_dir) / "cycle_dashboard.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        DASHBOARD_TEMPLATE.render(task=task, **audit),
        encoding="utf-8",
    )
    return target
