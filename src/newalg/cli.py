from __future__ import annotations

from pathlib import Path

import typer

from .calibration import DEFAULT_ALPHAS, run_bias_calibration_oof
from .agents import (
    analyze_paper_methods,
    analyze_papers,
    analyze_sota,
    assess_research_readiness,
    critique_ideas,
    design_experiments,
    export_submission,
    research_agents_cycle,
    review_paper_insights,
    scout_leaderboards,
    scout_papers,
    synthesize_ideas,
)
from .config import BudgetLevel, load_task_config
from .dashboard import generate_cycle_dashboard
from .ensemble import evaluate_prediction_ensemble
from .inference import predict_test_for_run
from .judge import judge_task
from .literature import DEFAULT_LITERATURE_QUERIES, discover_literature
from .optimizer import optimize_closed_loop, write_adaptive_proposals
from .pipeline import judge_and_report, reproduce_baselines, run_loop
from .pretraining import continue_pretrain_mlm, default_tapt_output_dir
from .proposer import create_method_cards, propose_experiments
from .registry import RunRegistry
from .reporting import generate_report

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("ingest-papers")
def ingest_papers_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    source: str = typer.Option(..., "--source", help="Path to paper YAML/JSON source"),
) -> None:
    config = load_task_config(task)
    cards = create_method_cards(config, source)
    typer.echo(f"Wrote {len(cards)} method cards to {config.method_cards_path}")


@app.command("discover-literature")
def discover_literature_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    output: str = typer.Option("outputs/literature/seed_papers.yaml", "--output", help="Output paper YAML path"),
    max_papers: int = typer.Option(8, "--max-papers", min=1, help="Maximum papers to keep"),
    arxiv: bool = typer.Option(True, "--arxiv/--no-arxiv", help="Use arXiv API in addition to curated seeds"),
    query: list[str] | None = typer.Option(None, "--query", help="Literature search query. Can be repeated."),
) -> None:
    config = load_task_config(task)
    papers = discover_literature(config, output, queries=query or DEFAULT_LITERATURE_QUERIES, max_papers=max_papers, use_arxiv=arxiv)
    typer.echo(f"Wrote {len(papers)} literature seeds to {output}")


@app.command("scout-leaderboards")
def scout_leaderboards_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    output: str | None = typer.Option(None, "--output", help="Optional leaderboard task YAML path"),
) -> None:
    config = load_task_config(task)
    boards = scout_leaderboards(config, output_path=output)
    typer.echo(f"Wrote {len(boards)} leaderboard tasks")


@app.command("sota-snapshot")
def sota_snapshot_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    snapshot = analyze_sota(config, leaderboard_task)
    typer.echo(f"Wrote SOTA snapshot for {snapshot.task_id}")


@app.command("scout-papers")
def scout_papers_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
    max_papers: int = typer.Option(12, "--max-papers", min=1, help="Maximum non-error papers to keep"),
) -> None:
    config = load_task_config(task)
    papers = scout_papers(config, leaderboard_task, max_papers=max_papers)
    error_count = sum(1 for paper in papers if paper.error)
    typer.echo(f"Wrote {len(papers)} paper evidence rows ({error_count} errors)")


@app.command("analyze-papers")
def analyze_papers_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    cards = analyze_papers(config, leaderboard_task)
    typer.echo(f"Wrote {len(cards)} method cards")


@app.command("analyze-paper-methods")
def analyze_paper_methods_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    analyses = analyze_paper_methods(config, leaderboard_task)
    full_text = sum(1 for item in analyses if item.analysis_depth == "full_text_pdf")
    typer.echo(f"Wrote {len(analyses)} paper analyses ({full_text} full-text)")


@app.command("review-paper-insights")
def review_paper_insights_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    reviews = review_paper_insights(config, leaderboard_task)
    accepted = sum(1 for item in reviews if item.approved_for_synthesis)
    typer.echo(f"Wrote {len(reviews)} paper insight reviews; approved={accepted}")


@app.command("research-readiness")
def research_readiness_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    report = assess_research_readiness(config, leaderboard_task)
    typer.echo(f"Research readiness: {report.status}; blockers={len(report.blocking_reasons)}")


@app.command("synthesize-ideas")
def synthesize_ideas_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    ideas = synthesize_ideas(config, leaderboard_task)
    typer.echo(f"Wrote {len(ideas)} idea cards")


@app.command("critique-ideas")
def critique_ideas_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    ideas = critique_ideas(config, leaderboard_task)
    accepted = sum(1 for idea in ideas if idea.accepted)
    typer.echo(f"Reviewed {len(ideas)} ideas; accepted={accepted}")


@app.command("design-experiments")
def design_experiments_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    specs = design_experiments(config, registry, leaderboard_task)
    typer.echo(f"Wrote {len(specs)} experiment specs")


@app.command("research-agents-cycle")
def research_agents_cycle_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    leaderboard_task: str | None = typer.Option(None, "--leaderboard-task", help="Leaderboard task id"),
    dry_run: bool = typer.Option(True, "--dry-run/--execute", help="Generate agent outputs without training by default"),
    max_papers: int = typer.Option(12, "--max-papers", min=1, help="Maximum paper evidence rows per cycle"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    decision = research_agents_cycle(config, registry, leaderboard_task, dry_run=dry_run, max_papers=max_papers)
    typer.echo(f"Research decision: {decision.task_id} status={decision.status.value} reason={decision.reason}")


@app.command("export-submission")
def export_submission_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    run_id: str = typer.Option(..., "--run-id", help="Completed run id"),
    output: str | None = typer.Option(None, "--output", help="Optional submission path"),
) -> None:
    config = load_task_config(task)
    path = export_submission(config, run_id, output_path=output)
    typer.echo(f"Submission file written to {path}")


@app.command("predict-test")
def predict_test_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    run_id: str = typer.Option(..., "--run-id", help="Completed transformer run id"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    path = predict_test_for_run(config, registry, run_id)
    typer.echo(f"Test predictions written to {path}")


@app.command("calibrate-bias")
def calibrate_bias_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    base_run_id: str = typer.Option(..., "--base-run-id", help="Completed transformer run to calibrate"),
    folds: int = typer.Option(5, "--folds", min=2, help="Stratified validation folds"),
    alpha: list[float] | None = typer.Option(None, "--alpha", help="Candidate alpha. Repeatable."),
    l2: float = typer.Option(0.02, "--l2", help="L2 penalty on class-bias magnitude"),
    seed: int = typer.Option(20260511, "--seed", help="Calibration fold seed"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    outcome = run_bias_calibration_oof(
        config,
        registry,
        base_run_id,
        folds=folds,
        alphas=alpha or DEFAULT_ALPHAS,
        l2=l2,
        seed=seed,
    )
    typer.echo(
        f"Calibration {outcome['run_id']} validation={outcome['validation_accuracy']:.4f} "
        f"lockbox={outcome['lockbox_accuracy']:.4f}"
    )


@app.command("reproduce-baselines")
def reproduce_baselines_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    rows = reproduce_baselines(config, registry)
    typer.echo(f"Completed {len(rows)} baseline runs")


@app.command("propose")
def propose_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    budget: BudgetLevel = typer.Option(BudgetLevel.SCREEN, "--budget", help="Budget tier"),
    output: str | None = typer.Option(None, "--output", help="Optional proposal output path"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    candidates = propose_experiments(config, registry, budget.value, output_path=output)
    typer.echo(f"Wrote {len(candidates)} proposals")


@app.command("run-loop")
def run_loop_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    budget: BudgetLevel = typer.Option(BudgetLevel.SCREEN, "--budget", help="Budget tier"),
    top_k: int = typer.Option(4, "--top-k", help="How many candidates to execute"),
    proposal_file: str | None = typer.Option(None, "--proposal-file", help="Optional experiment spec YAML"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    rows = run_loop(config, registry, budget, top_k, proposal_file=proposal_file)
    typer.echo(f"Completed {len(rows)} proposal runs")


@app.command("continue-pretrain")
def continue_pretrain_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    dataset: str = typer.Option("tnews", "--dataset", help="Dataset to use for task-adaptive MLM"),
    base_model_id: str = typer.Option("hfl/chinese-roberta-wwm-ext", "--base-model-id", help="Base masked language model"),
    output: str | None = typer.Option(None, "--output", help="Output model directory"),
    sample_cap: int = typer.Option(4096, "--sample-cap", min=1, help="Maximum training texts"),
    epochs: int = typer.Option(1, "--epochs", min=1, help="MLM epochs"),
    batch_size: int = typer.Option(16, "--batch-size", min=1, help="MLM batch size"),
    learning_rate: float = typer.Option(5e-5, "--learning-rate", help="MLM learning rate"),
    mlm_probability: float = typer.Option(0.15, "--mlm-probability", min=0.01, max=0.8, help="Masking probability"),
    seed: int = typer.Option(13, "--seed", help="Random seed"),
) -> None:
    config = load_task_config(task)
    output_dir = Path(output) if output else default_tapt_output_dir(config, dataset, base_model_id, sample_cap, seed)
    outcome = continue_pretrain_mlm(
        config,
        dataset,
        base_model_id,
        output_dir,
        sample_cap=sample_cap,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        mlm_probability=mlm_probability,
        seed=seed,
    )
    typer.echo(
        f"TAPT model written to {outcome.output_dir}; "
        f"rows={outcome.train_rows} loss={outcome.final_loss:.4f} seconds={outcome.train_seconds:.1f}"
    )


@app.command("evaluate-ensemble")
def evaluate_ensemble_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    run_id: list[str] = typer.Option(..., "--run-id", help="Completed run id. Repeat for each member."),
    mode: str = typer.Option("hard", "--mode", help="Voting mode: hard or validation_weight"),
    name: str | None = typer.Option(None, "--name", help="Optional ensemble experiment id"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    outcome = evaluate_prediction_ensemble(config, registry, run_id, mode=mode, name=name)
    typer.echo(
        f"Ensemble {outcome['run_id']} validation={outcome['validation_accuracy']:.4f} "
        f"lockbox={outcome['lockbox_accuracy']:.4f}"
    )


@app.command("judge")
def judge_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    report = judge_task(config, registry)
    typer.echo(f"Judge status: {report['status']}")


@app.command("report")
def report_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    report_path = generate_report(config, registry)
    typer.echo(f"Report written to {report_path}")


@app.command("dashboard")
def dashboard_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    dashboard_path = generate_cycle_dashboard(config, registry)
    typer.echo(f"Dashboard written to {dashboard_path}")


@app.command("auto-propose")
def auto_propose_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    budget: BudgetLevel = typer.Option(BudgetLevel.SCREEN, "--budget", help="Budget tier"),
    output: str | None = typer.Option(None, "--output", help="Optional proposal output path"),
    max_candidates: int | None = typer.Option(None, "--max-candidates", help="Override candidate count"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    target = output or str(config.resolve_path(config.proposals_dir) / f"auto-{budget.value}.yaml")
    candidates = write_adaptive_proposals(config, registry, budget, target, max_candidates=max_candidates)
    typer.echo(f"Wrote {len(candidates)} adaptive proposals to {target}")


@app.command("auto-optimize")
def auto_optimize_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    budget: BudgetLevel = typer.Option(BudgetLevel.SCREEN, "--budget", help="Budget tier"),
    rounds: int = typer.Option(1, "--rounds", min=1, help="Adaptive optimization rounds"),
    top_k: int = typer.Option(2, "--top-k", min=1, help="How many candidates to execute per round"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Write adaptive proposals without executing training"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    summaries = optimize_closed_loop(config, registry, budget, rounds, top_k, dry_run=dry_run)
    for summary in summaries:
        typer.echo(
            f"Round {summary.round_index}: proposed={summary.proposed_count} "
            f"executed={summary.executed_count} judge={summary.judge_status} "
            f"proposals={summary.proposal_path}"
        )


@app.command("research-cycle")
def research_cycle_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    source: str | None = typer.Option(None, "--source", help="Optional existing paper YAML/JSON source"),
    budget: BudgetLevel = typer.Option(BudgetLevel.SCREEN, "--budget", help="Budget tier"),
    top_k: int = typer.Option(3, "--top-k", min=1, help="How many theory candidates to execute"),
    optimize_rounds: int = typer.Option(1, "--optimize-rounds", min=0, help="Adaptive rounds after theory testing"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Generate proposals without executing training"),
    arxiv: bool = typer.Option(True, "--arxiv/--no-arxiv", help="Discover literature from arXiv when source is omitted"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    paper_source = source or str(config.resolve_path(config.output_root) / "literature" / "zero_start_papers.yaml")
    if source is None:
        papers = discover_literature(config, paper_source, max_papers=8, use_arxiv=arxiv)
        typer.echo(f"Discovered {len(papers)} literature seeds")
    create_method_cards(config, paper_source)

    theory_proposal = str(config.resolve_path(config.proposals_dir) / f"zero-start-{budget.value}.yaml")
    propose_experiments(config, registry, budget.value, output_path=theory_proposal)
    if not dry_run:
        run_loop(config, registry, budget, top_k, proposal_file=theory_proposal, skip_existing=True)
        report = judge_task(config, registry)
        typer.echo(f"Theory judge status: {report['status']}")
    else:
        typer.echo(f"Wrote zero-start proposals to {theory_proposal}")

    if optimize_rounds > 0:
        summaries = optimize_closed_loop(config, registry, budget, optimize_rounds, top_k, dry_run=dry_run)
        for summary in summaries:
            typer.echo(
                f"Optimize round {summary.round_index}: proposed={summary.proposed_count} "
                f"executed={summary.executed_count} judge={summary.judge_status} "
                f"proposals={summary.proposal_path}"
            )
    report_path = generate_report(config, registry)
    dashboard_path = generate_cycle_dashboard(config, registry)
    typer.echo(f"Report written to {report_path}")
    typer.echo(f"Dashboard written to {dashboard_path}")


@app.command("full-cycle")
def full_cycle_command(
    task: str = typer.Option(..., "--task", help="Path to research_task.yaml"),
    source: str = typer.Option(..., "--source", help="Path to paper YAML/JSON source"),
    budget: BudgetLevel = typer.Option(BudgetLevel.SMOKE, "--budget", help="Budget tier"),
    top_k: int = typer.Option(4, "--top-k", help="How many candidates to execute"),
    proposal_file: str | None = typer.Option(None, "--proposal-file", help="Optional experiment spec YAML"),
) -> None:
    config = load_task_config(task)
    registry = RunRegistry(config.registry_path)
    create_method_cards(config, source)
    reproduce_baselines(config, registry)
    run_loop(config, registry, budget, top_k, proposal_file=proposal_file)
    report, report_path = judge_and_report(config, registry)
    typer.echo(f"Judge status: {report['status']}")
    typer.echo(f"Report written to {report_path}")
