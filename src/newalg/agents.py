from __future__ import annotations

import json
import os
import io
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import pandas as pd
import httpx

from .config import (
    AugmentationType,
    BudgetLevel,
    ExperimentSpec,
    HeadType,
    IdeaCard,
    IdeaQualityReview,
    LeaderboardStatus,
    LeaderboardTask,
    LLMProvider,
    LossType,
    MethodCard,
    MethodModuleHints,
    PaperEvidence,
    PaperInsightReview,
    PaperMethodAnalysis,
    PaperQualityReport,
    ResearchReadinessReport,
    PoolingType,
    ResearchDecision,
    ResearchTaskConfig,
    ScheduleType,
    SotaSnapshot,
    dump_jsonl,
    dump_yaml,
)
from .judge import summarize_runs
from .llm import build_backend
from .registry import RunRegistry
from .utils import ensure_dir, read_frame, stable_hash


CLUE_BASELINES: dict[str, list[dict[str, Any]]] = {
    "tnews": [
        {"model": "BERT-base", "dev": 56.09, "test": 56.58},
        {"model": "BERT-wwm-ext-base", "dev": 56.77, "test": 56.86},
        {"model": "ERNIE-base", "dev": 58.24, "test": 58.33},
        {"model": "RoBERTa-wwm-ext", "dev": 57.51, "test": 56.94},
        {"model": "RoBERTa-wwm-large-ext", "dev": 58.32, "test": 58.61},
        {"model": "ALBERT-xxlarge", "dev": None, "test": 59.46},
    ],
    "iflytek": [
        {"model": "BERT-base", "dev": 60.29, "test": 60.19},
        {"model": "BERT-wwm-ext-base", "dev": 60.70, "test": 60.71},
        {"model": "RoBERTa-wwm-ext", "dev": 60.69, "test": 60.71},
    ],
}


def agent_root(task: ResearchTaskConfig) -> Path:
    return ensure_dir(task.resolve_path(task.agent_research.output_dir))


def task_dir(task: ResearchTaskConfig, task_id: str) -> Path:
    return ensure_dir(agent_root(task) / task_id)


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _paper_pdf_url(url: str) -> str | None:
    if not url:
        return None
    clean = url.split("#", 1)[0]
    if clean.endswith(".pdf") or "/pdf" in clean:
        return clean
    if "arxiv.org/abs/" in clean:
        return clean.replace("/abs/", "/pdf/") + ".pdf"
    if "aclanthology.org/" in clean:
        return clean.rstrip("/") + ".pdf"
    if "mdpi.com/" in clean and "/pdf" not in clean:
        return clean.rstrip("/") + "/pdf"
    return None


def _extract_pdf_text(url: str, max_pages: int = 8, max_chars: int = 16000) -> str:
    pdf_url = _paper_pdf_url(url)
    if not pdf_url:
        return ""
    try:
        from pypdf import PdfReader

        content = _fetch_url(pdf_url, timeout=12.0)
        reader = PdfReader(io.BytesIO(content))
        chunks: list[str] = []
        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)
            if sum(len(chunk) for chunk in chunks) >= max_chars:
                break
        return " ".join(" ".join(chunk.split()) for chunk in chunks)[:max_chars]
    except Exception:
        return ""


def _response_text(data: dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])
    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"} and content.get("text"):
                chunks.append(str(content["text"]))
    if chunks:
        return "\n".join(chunks)
    raise RuntimeError(f"LLM response missing text: {data}")


def _structured_llm(task: ResearchTaskConfig, prompt: str, schema: dict[str, Any], name: str) -> dict[str, Any]:
    if task.llm.provider != LLMProvider.OPENAI:
        raise RuntimeError("Structured LLM is disabled because provider is not openai")
    api_key = os.getenv(task.llm.api_key_env)
    if not api_key:
        raise RuntimeError(f"{task.llm.api_key_env} is not set")
    payload = {
        "model": task.llm.model,
        "input": prompt,
        "temperature": task.llm.temperature,
        "max_output_tokens": task.llm.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": name,
                "schema": schema,
                "strict": True,
            }
        },
    }
    response = httpx.post(
        task.llm.base_url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=90.0,
    )
    response.raise_for_status()
    return json.loads(_response_text(response.json()))


def _default_leaderboards(task: ResearchTaskConfig) -> list[LeaderboardTask]:
    defaults: list[LeaderboardTask] = []
    for index, dataset_name in enumerate(["tnews", "iflytek"]):
        if dataset_name not in task.datasets:
            continue
        cfg = task.datasets[dataset_name]
        strong = CLUE_BASELINES.get(dataset_name, [])
        best = max(
            (row for row in strong if row.get("test") is not None),
            key=lambda row: float(row["test"]),
            default={},
        )
        defaults.append(
            LeaderboardTask(
                task_id=dataset_name,
                dataset=dataset_name,
                display_name=f"CLUE {dataset_name.upper()}",
                metric=task.metric_name,
                official_url="https://github.com/CLUEbenchmark/CLUE",
                submission_format="jsonl",
                strong_baseline_name=str(best.get("model", "")),
                strong_baseline_score=float(best["test"]) if best.get("test") is not None else None,
                priority=100 - index,
                notes=f"{cfg.hf_path or cfg.source}:{cfg.hf_name or dataset_name}",
            )
        )
    return defaults


def scout_leaderboards(task: ResearchTaskConfig, output_path: str | Path | None = None) -> list[LeaderboardTask]:
    leaderboards = task.leaderboards or _default_leaderboards(task)
    target = Path(output_path) if output_path else agent_root(task) / "leaderboard_tasks.yaml"
    dump_yaml(target, {"leaderboards": [board.model_dump(mode="json") for board in leaderboards]})
    return leaderboards


def load_leaderboards(task: ResearchTaskConfig) -> list[LeaderboardTask]:
    path = agent_root(task) / "leaderboard_tasks.yaml"
    if path.exists():
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return [LeaderboardTask.model_validate(item) for item in payload.get("leaderboards", [])]
    return scout_leaderboards(task)


def select_leaderboard(task: ResearchTaskConfig, task_id: str | None = None) -> LeaderboardTask:
    boards = sorted(load_leaderboards(task), key=lambda item: item.priority, reverse=True)
    if task_id:
        for board in boards:
            if board.task_id == task_id:
                return board
        raise ValueError(f"Leaderboard task not found: {task_id}")
    active = [board for board in boards if board.status != LeaderboardStatus.ABANDONED]
    if not active:
        raise ValueError("No active leaderboard tasks available")
    return active[0]


def analyze_sota(task: ResearchTaskConfig, task_id: str | None = None) -> SotaSnapshot:
    board = select_leaderboard(task, task_id)
    baselines = CLUE_BASELINES.get(board.dataset, [])
    target_score = board.strong_baseline_score
    if target_score is None and baselines:
        scored = [row for row in baselines if row.get("test") is not None]
        target_score = max(float(row["test"]) for row in scored) if scored else None
    snapshot = SotaSnapshot(
        task_id=board.task_id,
        dataset=board.dataset,
        metric=board.metric,
        official_url=board.official_url,
        baselines=baselines,
        target_score=target_score,
        submission_format=board.submission_format,
        notes=[
            "Local validation and lockbox are screening signals only.",
            "Official leaderboard claims require exported test predictions and manual submission.",
        ],
    )
    _write_json(task_dir(task, board.task_id) / "sota_snapshot.json", snapshot.model_dump(mode="json"))
    return snapshot


def _fetch_url(url: str, timeout: float = 8.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "newalg-research-agent/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _tags_for_text(text: str) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    keyword_map = [
        ("short text", "short_text"),
        ("classification", "classification"),
        ("prompt", "prompt"),
        ("label", "label_semantics"),
        ("verbalizer", "label_semantics"),
        ("contrastive", "contrastive"),
        ("adapter", "adapter"),
        ("ensemble", "ensemble"),
        ("distillation", "distillation"),
        ("calibration", "calibration"),
        ("augmentation", "augmentation"),
        ("r-drop", "rdrop"),
        ("dropout", "rdrop"),
        ("chinese", "chinese"),
    ]
    for keyword, tag in keyword_map:
        if keyword in lowered and tag not in tags:
            tags.append(tag)
    return tags or ["text_classification"]


def _score_paper(title: str, abstract: str, year: int | None, code_url: str | None) -> float:
    text = f"{title} {abstract}".lower()
    score = 0.2
    keywords = [
        "classification",
        "short text",
        "prompt",
        "label",
        "chinese",
        "contrastive",
        "augmentation",
        "distillation",
        "calibration",
    ]
    for keyword in keywords:
        if keyword in text:
            score += 0.08
    if code_url:
        score += 0.1
    if year and year >= datetime.now().year - 3:
        score += 0.15
    return min(1.0, round(score, 3))


def _paper_scout_queries(board: LeaderboardTask) -> list[str]:
    dataset = board.dataset.lower()
    base = [
        f"CLUE {dataset} text classification",
        f"{dataset} Chinese text classification",
        "prompt learning Chinese few-shot text classification",
        "Chinese text classification prompt tuning",
        "Chinese text classification label words verbalizer",
        "Chinese text classification unlabeled data prompt learning",
        "Chinese BERT text classification contrastive learning",
        "Chinese short text classification graph attention",
        "Chinese short text classification benchmark",
        "Chinese news title classification BERT",
        "Chinese text classification label semantics prompt learning",
        "Chinese text classification data augmentation distillation calibration",
        "few-shot Chinese text classification label semantics",
        "text classification supervised contrastive learning calibration",
    ]
    if dataset == "tnews":
        base.extend(
            [
                "TNEWS Toutiao news title classification",
                "Chinese news classification CLUE TNEWS",
                "short news title classification prompt label verbalizer",
            ]
        )
    if dataset == "iflytek":
        base.extend(
            [
                "IFLYTEK app description classification CLUE",
                "Chinese app description classification",
                "long Chinese text classification IFLYTEK benchmark",
            ]
        )
    return list(dict.fromkeys(base))


def _clip_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _keyword_score(text: str, weighted_terms: dict[str, float]) -> float:
    return _clip_score(sum(weight for term, weight in weighted_terms.items() if term in text))


def _arxiv_papers(query: str, max_results: int) -> list[PaperEvidence]:
    search_query = f'cat:cs.CL AND all:"{query}"'
    params = urllib.parse.urlencode(
        {
            "search_query": search_query,
            "start": 0,
            "max_results": max(max_results * 2, max_results),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    root = ET.fromstring(_fetch_url(f"https://export.arxiv.org/api/query?{params}"))
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers: list[PaperEvidence] = []
    for entry in root.findall("atom:entry", ns):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
        abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
        url = entry.findtext("atom:id", default="", namespaces=ns) or ""
        published = entry.findtext("atom:published", default="", namespaces=ns) or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        paper = PaperEvidence(
            paper_id=stable_hash({"source": url, "title": title}, prefix="paper-"),
            title=title,
            source_url=url,
            source_name="arxiv",
            year=year,
            venue="arXiv",
            abstract=abstract,
            relevance_score=_score_paper(title, abstract, year, None),
            tags=_tags_for_text(f"{title} {abstract}"),
            query=query,
        )
        papers.append(paper)
        if len(papers) >= max_results:
            break
    return papers


def _abstract_from_openalex(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""
    words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            words.append((int(position), word))
    return " ".join(word for _, word in sorted(words))


def _openalex_papers(query: str, max_results: int) -> list[PaperEvidence]:
    params = urllib.parse.urlencode(
        {
            "search": query,
            "filter": "concepts.id:C204321447",
            "per-page": max(max_results * 2, max_results),
            "sort": "relevance_score:desc",
        }
    )
    payload = json.loads(_fetch_url(f"https://api.openalex.org/works?{params}"))
    papers: list[PaperEvidence] = []
    for item in payload.get("results", []):
        title = item.get("title") or ""
        abstract = _abstract_from_openalex(item.get("abstract_inverted_index"))
        url = item.get("doi") or item.get("id") or ""
        year = item.get("publication_year")
        primary_location = item.get("primary_location") or {}
        venue = primary_location.get("source", {}) or {}
        open_access = item.get("open_access") or {}
        source_url = open_access.get("oa_url") or primary_location.get("landing_page_url") or item.get("doi") or item.get("id") or ""
        paper = PaperEvidence(
            paper_id=stable_hash({"source": source_url or url, "title": title}, prefix="paper-"),
            title=title,
            source_url=str(source_url or url),
            source_name="openalex",
            year=int(year) if year else None,
            venue=venue.get("display_name") or primary_location.get("raw_source_name"),
            abstract=abstract,
            code_url=None,
            relevance_score=_score_paper(title, abstract, int(year) if year else None, None),
            tags=_tags_for_text(f"{title} {abstract}"),
            query=query,
        )
        papers.append(paper)
        if len(papers) >= max_results:
            break
    return papers


def _acl_papers(query: str, max_results: int) -> list[PaperEvidence]:
    params = urllib.parse.urlencode({"q": query})
    text = _fetch_url(f"https://aclanthology.org/search/?{params}").decode("utf-8", errors="ignore")
    papers: list[PaperEvidence] = []
    for marker in text.split('href="/')[: max_results + 1]:
        if not marker.startswith(("20", "P", "N", "D")):
            continue
        path = marker.split('"', 1)[0]
        title = path.replace("/", " ").replace("-", " ").strip() or query
        url = f"https://aclanthology.org/{path}"
        papers.append(
            PaperEvidence(
                paper_id=stable_hash({"source": url, "title": title}, prefix="paper-"),
                title=title,
                source_url=url,
                source_name="acl_anthology",
                abstract="",
                venue="ACL Anthology",
                relevance_score=_score_paper(title, "", None, None),
                tags=_tags_for_text(title),
                query=query,
            )
        )
        if len(papers) >= max_results:
            break
    return papers


def evaluate_paper_quality(task: ResearchTaskConfig, board: LeaderboardTask, papers: list[PaperEvidence]) -> tuple[list[PaperEvidence], PaperQualityReport]:
    """Score collected evidence before it can drive innovation generation."""
    current_year = datetime.now().year
    dataset = board.dataset.lower()
    task_terms = {
        "text classification": 0.18,
        "short text classification": 0.16,
        "classification": 0.08,
        "chinese": 0.14,
        "chinese bert": 0.12,
        "bert": 0.06,
        "nlp tasks": 0.1,
        "nlu tasks": 0.1,
        "natural language processing": 0.08,
        "news classification": 0.12,
        "title classification": 0.12,
        "app description": 0.12,
        "few-shot": 0.08,
        "prompt": 0.08,
        "label semantics": 0.1,
        "verbalizer": 0.08,
        "contrastive": 0.08,
        "augmentation": 0.07,
        "distillation": 0.07,
        "calibration": 0.07,
    }
    benchmark_terms = {
        "clue": 0.3,
        dataset: 0.28,
        "tnews": 0.28,
        "iflytek": 0.28,
        "fewclue": 0.2,
        "benchmark": 0.12,
        "toutiao": 0.15,
        "public dataset": 0.08,
    }
    transfer_terms = {
        "prompt": 0.1,
        "prompt learning": 0.16,
        "label semantics": 0.12,
        "verbalizer": 0.1,
        "label words": 0.08,
        "contrastive": 0.1,
        "supervised contrastive": 0.13,
        "pre-trained language models": 0.16,
        "pre-trained language model": 0.16,
        "pretrained language models": 0.08,
        "pretrained language model": 0.08,
        "robust": 0.07,
        "adversarial": 0.07,
        "data augmentation": 0.1,
        "distillation": 0.1,
        "calibration": 0.1,
        "attention network": 0.08,
        "graph attention": 0.08,
        "hybrid neural": 0.06,
        "adapter": 0.08,
        "ensemble": 0.08,
        "dropout": 0.06,
        "regularized dropout": 0.1,
        "r-drop": 0.08,
        "entailment": 0.08,
        "glyph": 0.08,
        "pinyin": 0.08,
    }
    sota_terms = {
        "state-of-the-art": 0.2,
        "sota": 0.2,
        "outperform": 0.12,
        "leaderboard": 0.15,
        "benchmark": 0.1,
        "rank": 0.08,
        "competitive": 0.06,
    }
    irrelevant_terms = {
        "quantum",
        "thermalization",
        "elliptic",
        "conspiracy",
        "legal text",
        "duplicate-step",
        "behaviour-driven",
        "behavior-driven",
        "software engineering",
        "medical image",
    }
    huge_model_terms = {
        "70b",
        "32b",
        "mixture-of-experts",
        "from scratch",
        "pretraining corpus",
        "large language model fine-tuning",
    }
    evaluated: list[PaperEvidence] = []
    for paper in papers:
        if paper.error:
            evaluated.append(paper)
            continue
        text = f"{paper.title} {paper.abstract}".lower()
        task_relevance = _keyword_score(text, task_terms)
        benchmark_relevance = _keyword_score(text, benchmark_terms)
        transferability = _keyword_score(text, transfer_terms)
        code = 1.0 if paper.code_url or "github" in text or "code is available" in text or "source code" in text else 0.0
        sota = _keyword_score(text, sota_terms)
        feasibility = 0.75
        if paper.year and paper.year >= current_year - task.agent_research.evidence_year_window:
            feasibility += 0.1
        if any(term in text for term in huge_model_terms):
            feasibility -= 0.35
        if any(term in text for term in irrelevant_terms):
            task_relevance -= 0.25
            transferability -= 0.15
            feasibility -= 0.15
        task_relevance = _clip_score(task_relevance)
        benchmark_relevance = _clip_score(benchmark_relevance)
        transferability = _clip_score(transferability)
        sota = _clip_score(sota)
        feasibility = _clip_score(feasibility)
        quality = _clip_score(
            0.12
            + 0.38 * task_relevance
            + 0.12 * benchmark_relevance
            + 0.28 * transferability
            + 0.05 * code
            + 0.1 * sota
            + 0.07 * feasibility
            + (0.08 if paper.source_name == "bootstrap" else 0.0)
        )
        reasons: list[str] = []
        if task_relevance < 0.45:
            reasons.append("task_relevance_low")
        if benchmark_relevance < 0.18:
            reasons.append("benchmark_relevance_low")
        if transferability < 0.18:
            reasons.append("method_transferability_low")
        if code <= 0:
            reasons.append("no_code_signal")
        if any(term in text for term in irrelevant_terms):
            reasons.append("irrelevant_domain_signal")
        if feasibility < 0.45:
            reasons.append("local_feasibility_low")
        accepted = (
            quality >= task.agent_research.paper_quality_threshold
            and task_relevance >= 0.45
            and transferability >= 0.18
            and feasibility >= 0.45
        )
        evaluated.append(
            paper.model_copy(
                update={
                    "quality_score": quality,
                    "quality_pass": accepted,
                    "task_relevance_score": task_relevance,
                    "benchmark_relevance_score": benchmark_relevance,
                    "method_transferability_score": transferability,
                    "code_availability_score": code,
                    "sota_evidence_score": sota,
                    "implementation_feasibility_score": feasibility,
                    "quality_reasons": reasons or ["qualified"],
                }
            )
        )
    real_papers = [paper for paper in evaluated if not paper.error]
    qualified = [paper for paper in real_papers if paper.quality_pass]
    source_counts: dict[str, int] = {}
    query_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    for paper in real_papers:
        source_counts[paper.source_name] = source_counts.get(paper.source_name, 0) + 1
        if paper.query:
            query_counts[paper.query] = query_counts.get(paper.query, 0) + 1
        for tag in paper.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    failure_reasons: list[str] = []
    if len(real_papers) < task.agent_research.min_qualified_papers:
        failure_reasons.append(f"raw_candidate_count_low:{len(real_papers)}")
    if len(qualified) < task.agent_research.min_qualified_papers:
        failure_reasons.append(f"qualified_paper_count_low:{len(qualified)}/{task.agent_research.min_qualified_papers}")
    if real_papers:
        avg_quality = round(sum(paper.quality_score for paper in real_papers) / len(real_papers), 3)
    else:
        avg_quality = 0.0
    avg_qualified = round(sum(paper.quality_score for paper in qualified) / len(qualified), 3) if qualified else 0.0
    report = PaperQualityReport(
        task_id=board.task_id,
        dataset=board.dataset,
        status="passed" if len(qualified) >= task.agent_research.min_qualified_papers else "failed",
        total_candidates=len(evaluated),
        raw_papers=len(real_papers),
        qualified_papers=len(qualified),
        min_required=task.agent_research.min_qualified_papers,
        quality_threshold=task.agent_research.paper_quality_threshold,
        average_quality_score=avg_quality,
        average_qualified_score=avg_qualified,
        source_counts=source_counts,
        query_counts=dict(sorted(query_counts.items(), key=lambda item: item[1], reverse=True)[:12]),
        top_keywords=[tag for tag, _ in sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:12]],
        qualified_paper_ids=[paper.paper_id for paper in qualified],
        failure_reasons=failure_reasons,
        next_actions=[
            "Expand source/query matrix and rerun Paper Scout before generating ideas."
            if len(qualified) < task.agent_research.min_qualified_papers
            else "Proceed to Paper Analyst with qualified evidence only."
        ],
    )
    return evaluated, report


def _bootstrap_reference_papers(board: LeaderboardTask) -> list[PaperEvidence]:
    """Authoritative seed references used to avoid cold-starting from noisy search APIs."""
    common = [
        {
            "title": "CLUE: A Chinese Language Understanding Evaluation Benchmark",
            "source_url": "https://arxiv.org/abs/2004.05986",
            "year": 2020,
            "venue": "COLING",
            "abstract": (
                "Introduces the CLUE Chinese language understanding benchmark with Chinese NLU tasks, "
                "including short text classification and long text classification. It defines the benchmark "
                "context for CLUE leaderboard work and baseline evaluation."
            ),
            "tags": ["benchmark", "chinese", "classification"],
        },
        {
            "title": "FewCLUE: A Chinese Few-shot Learning Evaluation Benchmark",
            "source_url": "https://arxiv.org/abs/2107.07498",
            "year": 2021,
            "venue": "arXiv",
            "abstract": (
                "Introduces a Chinese few-shot learning benchmark across Chinese NLU tasks and evaluates "
                "prompt learning baselines, making it relevant to TNEWS-style text classification under "
                "limited supervision."
            ),
            "tags": ["benchmark", "chinese", "prompt", "classification"],
        },
        {
            "title": "Making Pre-trained Language Models Better Few-shot Learners",
            "source_url": "https://arxiv.org/abs/2012.15723",
            "year": 2020,
            "venue": "ACL",
            "abstract": (
                "Presents LM-BFF, a prompt-based fine-tuning framework for text classification with automatic "
                "prompt generation, demonstrations, calibration, and few-shot evaluation."
            ),
            "tags": ["prompt", "calibration", "classification"],
        },
        {
            "title": "Knowledgeable Prompt-tuning: Incorporating Knowledge into Prompt Verbalizer for Text Classification",
            "source_url": "https://aclanthology.org/2022.acl-long.158/",
            "year": 2022,
            "venue": "ACL",
            "abstract": (
                "Improves prompt tuning for text classification by expanding and refining label words in the "
                "verbalizer with external knowledge, reducing label word bias in zero-shot and few-shot settings."
            ),
            "tags": ["prompt", "label_semantics", "classification"],
        },
        {
            "title": "RoCBert: Robust Chinese Bert with Multimodal Contrastive Pretraining",
            "source_url": "https://aclanthology.org/2022.acl-long.65.pdf",
            "year": 2022,
            "venue": "ACL",
            "abstract": (
                "Proposes a robust Chinese BERT pretrained with contrastive learning over semantic, phonetic, "
                "and visual features, and reports strong performance across Chinese NLU tasks."
            ),
            "tags": ["chinese", "contrastive", "classification"],
        },
        {
            "title": "ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information",
            "source_url": "https://aclanthology.org/2021.acl-long.161/",
            "year": 2021,
            "venue": "ACL",
            "abstract": (
                "Enhances Chinese pretrained language models with glyph and pinyin signals for Chinese NLP tasks, "
                "providing transfer mechanisms for robust Chinese text representations."
            ),
            "tags": ["chinese", "classification"],
        },
        {
            "title": "R-Drop: Regularized Dropout for Neural Networks",
            "source_url": "https://arxiv.org/abs/2106.14448",
            "year": 2021,
            "venue": "NeurIPS",
            "abstract": (
                "Uses regularized dropout consistency to improve neural network fine-tuning stability and "
                "generalization; it is directly transferable to transformer text classification training."
            ),
            "tags": ["rdrop", "classification"],
        },
    ]
    if board.dataset.lower() == "tnews":
        common.append(
            {
                "title": "Investigating Prompt Learning for Chinese Few-Shot Text Classification with Pre-Trained Language Models",
                "source_url": "https://www.mdpi.com/2076-3417/12/21/11117",
                "year": 2022,
                "venue": "Applied Sciences",
                "abstract": (
                    "Studies prompt learning for Chinese few-shot text classification with pre-trained language "
                    "models, label semantics, and classification-oriented prompt design."
                ),
                "tags": ["prompt", "label_semantics", "classification", "chinese"],
            }
        )
    papers = []
    for row in common:
        papers.append(
            PaperEvidence(
                paper_id=stable_hash({"source": row["source_url"], "title": row["title"]}, prefix="paper-"),
                title=str(row["title"]),
                source_url=str(row["source_url"]),
                source_name="bootstrap",
                year=int(row["year"]),
                venue=str(row["venue"]),
                abstract=str(row["abstract"]),
                relevance_score=_score_paper(str(row["title"]), str(row["abstract"]), int(row["year"]), None),
                tags=list(row["tags"]),
                query="bootstrap_authoritative_seed",
            )
        )
    return papers


def scout_papers(task: ResearchTaskConfig, task_id: str | None = None, max_papers: int = 12) -> list[PaperEvidence]:
    board = select_leaderboard(task, task_id)
    queries = _paper_scout_queries(board)[: max(1, task.agent_research.max_scout_rounds * 3)]
    per_query = max(2, max_papers // max(1, len(queries)))
    papers: list[PaperEvidence] = []
    errors: list[PaperEvidence] = []
    fetchers = {"arxiv": _arxiv_papers, "openalex": _openalex_papers, "acl_anthology": _acl_papers}
    if "bootstrap" in task.agent_research.paper_sources:
        papers.extend(_bootstrap_reference_papers(board))
    disabled_sources: set[str] = set()
    for query in queries:
        for source in task.agent_research.paper_sources:
            if source == "bootstrap":
                continue
            if source in disabled_sources:
                continue
            fetcher = fetchers.get(source)
            if fetcher is None:
                errors.append(
                    PaperEvidence(
                        paper_id=stable_hash({"source": source, "query": query, "error": "unsupported"}, prefix="paper-error-"),
                        title=f"{source} unsupported",
                        source_url=source,
                        source_name=source,
                        query=query,
                        error="Unsupported paper source",
                        relevance_score=0.0,
                    )
                )
                continue
            try:
                papers.extend(fetcher(query, per_query))
            except Exception as exc:
                error_message = str(exc)
                errors.append(
                    PaperEvidence(
                        paper_id=stable_hash({"source": source, "query": query, "error": error_message}, prefix="paper-error-"),
                        title=f"{source} fetch failed",
                        source_url=source,
                        source_name=source,
                        query=query,
                        error=error_message,
                        relevance_score=0.0,
                    )
                )
                if "429" in error_message or "Too Many Requests" in error_message:
                    disabled_sources.add(source)
    seen: set[str] = set()
    deduped: list[PaperEvidence] = []
    for paper in papers:
        key = paper.source_url or paper.title
        if key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    evaluated, report = evaluate_paper_quality(task, board, deduped + errors)
    real_sorted = sorted(
        [paper for paper in evaluated if not paper.error],
        key=lambda item: (item.quality_pass, item.quality_score, item.relevance_score),
        reverse=True,
    )
    result = real_sorted[:max_papers] + errors
    root = task_dir(task, board.task_id)
    _write_json(
        root / "paper_scout_strategy.json",
        {
            "task_id": board.task_id,
            "dataset": board.dataset,
            "sources": task.agent_research.paper_sources,
            "queries": queries,
            "max_papers": max_papers,
            "max_scout_rounds": task.agent_research.max_scout_rounds,
            "quality_gate": {
                "min_qualified_papers": task.agent_research.min_qualified_papers,
                "paper_quality_threshold": task.agent_research.paper_quality_threshold,
                "requires_task_relevance_at_least": 0.45,
                "requires_method_transferability_at_least": 0.18,
                "requires_local_feasibility_at_least": 0.45,
            },
            "source_notes": {
                "bootstrap": "Transparent authoritative seed references for benchmark and proven method cold start; not counted as live API search.",
                "openalex": "NLP concept-filtered metadata search sorted by relevance.",
                "arxiv": "Direct arXiv API source; rate-limited sources are fused after 429.",
                "acl_anthology": "ACL website search fallback; may return sparse results because search is Google-CSE backed.",
            },
        },
    )
    dump_jsonl(root / "paper_evidence.raw.jsonl", evaluated)
    dump_jsonl(root / "paper_evidence.jsonl", result)
    _write_json(root / "paper_quality_report.json", report.model_dump(mode="json"))
    return result


def _module_hints_from_text(text: str) -> MethodModuleHints:
    lowered = text.lower()
    hints = MethodModuleHints()
    if any(term in lowered for term in ["contrastive", "supervised contrastive"]):
        hints.loss.append(LossType.CE_SUPCON)
    if any(term in lowered for term in ["dropout", "consistency", "r-drop", "regularized dropout"]):
        hints.loss.append(LossType.RDROP)
    if any(term in lowered for term in ["label smoothing", "smooth labels"]):
        hints.loss.append(LossType.LABEL_SMOOTHING)
    if any(term in lowered for term in ["focal", "hard example", "class imbalance"]):
        hints.loss.append(LossType.FOCAL)
    if any(term in lowered for term in ["prompt", "verbalizer", "label word", "label semantics"]):
        hints.head.append(HeadType.LABEL_SEMANTIC)
        hints.loss.append(LossType.LABEL_ANCHOR_CONTRASTIVE)
    if any(term in lowered for term in ["attention", "graph attention"]):
        hints.pooling.append(PoolingType.ATTENTION)
    if any(term in lowered for term in ["mean pooling", "sentence representation", "representation geometry"]):
        hints.pooling.append(PoolingType.MEAN)
    if any(term in lowered for term in ["augmentation", "unlabeled data", "adversarial", "perturbation"]):
        hints.augmentation.append(AugmentationType.TOKEN_MASK)
    if any(term in lowered for term in ["layer-wise", "layerwise", "fine-tuning"]):
        hints.schedule.append(ScheduleType.LAYERWISE_LR_DECAY)
    return hints


def _analysis_family(text: str) -> str:
    lowered = text.lower()
    if "prompt" in lowered or "verbalizer" in lowered or "label word" in lowered:
        return "prompt_and_label_semantics"
    if "contrastive" in lowered:
        return "contrastive_representation_learning"
    if "benchmark" in lowered or "evaluation" in lowered:
        return "benchmark_and_evaluation"
    if "dropout" in lowered or "consistency" in lowered:
        return "regularization"
    if "glyph" in lowered or "pinyin" in lowered or "pretraining" in lowered:
        return "chinese_representation_pretraining"
    return "text_classification_method"


def _analyze_single_paper(paper: PaperEvidence) -> PaperMethodAnalysis:
    full_text = _extract_pdf_text(paper.source_url)
    evidence = full_text or paper.abstract
    depth = "full_text_pdf" if full_text else "abstract_only"
    analysis_text = f"{paper.title}\n{paper.abstract}\n{evidence[:12000]}"
    lowered = analysis_text.lower()
    hints = _module_hints_from_text(analysis_text)
    family = _analysis_family(analysis_text)
    title_lower = paper.title.lower()
    transferable: list[str] = []
    requirements: list[str] = []
    theory: list[str] = []
    risks: list[str] = []
    takeaways: list[str] = []
    objective = "Cross-entropy classification objective"
    expected = "Potentially improves TNEWS validation if the mechanism targets short-text ambiguity."

    if family == "prompt_and_label_semantics":
        transferable.extend(["Use label semantics/verbalizer information instead of treating labels as anonymous ids.", "Calibrate class probabilities when label words introduce prior bias."])
        requirements.extend(["Add or reuse label descriptions for TNEWS classes.", "Map prompt/verbalizer logic to label_semantic head or an implementation ticket."])
        theory.extend(["Label-word semantics can inject task prior knowledge into low-data or ambiguous text classification."])
        takeaways.append("Most relevant when class names carry useful semantic information.")
        objective = "Masked/prompted classification or cross-entropy with label-semantic classifier"
        expected = "May help confused short news categories where title text is sparse and label semantics are informative."
    if family == "contrastive_representation_learning":
        transferable.extend(["Add supervised contrastive pressure so same-label titles cluster and confusing labels separate.", "Use robustness perturbations as positive-pair consistency signals."])
        requirements.extend(["Implement CE + supervised contrastive loss or reuse ce_supcon runner path.", "Batch composition must contain enough same-class examples."])
        theory.extend(["Representation geometry affects class separability under short and ambiguous inputs."])
        takeaways.append("Useful if baseline errors are caused by weak inter-class margins.")
        objective = "Cross-entropy plus contrastive or consistency regularization"
        expected = "May improve lockbox stability by making representation clusters less brittle."
    if family == "benchmark_and_evaluation":
        transferable.extend(["Use benchmark task taxonomy and baseline results to define realistic target and failure analysis.", "Use CLUE/FewCLUE task evidence to avoid optimizing against a private or non-comparable metric."])
        requirements.extend(["No new runner module required; use as SOTA/benchmark context for idea constraints."])
        theory.extend(["A valid leaderboard claim requires comparable benchmark split and metric discipline."])
        takeaways.append("Reference is important for task framing, but may not itself contain an executable algorithm.")
        expected = "Improves research validity rather than directly improving model accuracy."
    if family == "regularization":
        transferable.extend(["Enforce prediction consistency across stochastic dropout passes.", "Use regularization to reduce seed variance in fine-tuning."])
        requirements.extend(["Use rdrop loss path and tune KL/CE weight under smoke and screen budgets."])
        theory.extend(["Consistency regularization reduces train-test mismatch introduced by dropout noise."])
        takeaways.append("Low implementation cost and suitable for local base encoder fine-tuning.")
        objective = "Cross-entropy plus bidirectional KL consistency between dropout passes"
        expected = "May improve generalization and reduce seed variance, but can underfit with too small budget."
    if family == "chinese_representation_pretraining":
        transferable.extend(["Borrow Chinese-specific robustness ideas such as glyph/pinyin/noise perturbation.", "Use token masking as a local approximation when pretraining is out of scope."])
        requirements.extend(["Do not pretrain a new encoder locally; map to augmentation or implementation ticket only."])
        theory.extend(["Chinese character form and pronunciation can carry semantic clues missing from vanilla wordpiece IDs."])
        risks.append("Full method may require pretraining and therefore exceed local compute constraints.")
        takeaways.append("Valuable for direction, but base-level local implementation must be a lightweight approximation.")
        expected = "May inspire augmentation/robustness variants, not a direct one-night leaderboard method."
    if "unlabeled" in lowered:
        transferable.append("Use unlabeled data through pseudo-labeling or consistency learning if public unlabeled samples are allowed.")
        requirements.append("Need an explicit rule to avoid leaking official test labels.")
        risks.append("Pseudo-label noise can amplify baseline mistakes.")
    if "large language model" in lowered or "llm" in lowered:
        risks.append("May depend on larger models than local M2/16GB can train.")
    if not risks:
        risks.append("Transfer may fail because source benchmark distribution differs from TNEWS.")

    worth = bool(transferable) and family != "benchmark_and_evaluation"
    quality = (
        "strong_method_reference" if worth and paper.quality_pass else
        "benchmark_context_only" if family == "benchmark_and_evaluation" else
        "weak_or_indirect_reference"
    )
    model_name = paper.title
    core_mechanism = (evidence[:700] or paper.abstract[:700]).strip()
    if "making pre-trained language models better few-shot learners" in title_lower:
        model_name = "LM-BFF: prompt template search + demonstrations + calibration"
        core_mechanism = (
            "LM-BFF converts classification into masked language modeling, searches or generates prompt "
            "templates, augments each input with demonstrations, and calibrates verbalizer probabilities to "
            "reduce prompt-induced label bias."
        )
        transferable = [
            "Represent TNEWS labels through verbalizer words or label descriptions.",
            "Use calibration on validation/lockbox to correct prior bias from label words.",
            "Use demonstration-style nearest examples only if they can be selected from train without leakage.",
        ]
        requirements = [
            "Implement prompt/verbalizer inference or approximate it with label_semantic head.",
            "Add calibration parameters and ensure they are fitted only on training/dev data.",
            "If demonstrations are used, add deterministic example retrieval and audit leakage risk.",
        ]
        theory = ["Prompt format and label-word priors strongly affect few-shot/fine-tuning behavior."]
        hints = MethodModuleHints(head=[HeadType.LABEL_SEMANTIC], loss=[LossType.LABEL_ANCHOR_CONTRASTIVE], schedule=[ScheduleType.LAYERWISE_LR_DECAY])
        takeaways = ["Most promising part for local implementation is calibrated label semantics, not full prompt search."]
        risks = ["Prompt search can overfit dev data.", "Demonstration retrieval adds inference cost and leakage risk."]
        expected = "Could improve ambiguous TNEWS labels by using label meaning and calibration rather than anonymous class ids."
    elif "knowledgeable prompt-tuning" in title_lower:
        model_name = "KPT: knowledge-expanded verbalizer for prompt classification"
        core_mechanism = (
            "KPT expands label words with external knowledge, refines candidate verbalizers, and uses the "
            "expanded verbalizer distribution to make prompt-based text classification less dependent on a "
            "single manually chosen label word."
        )
        transferable = [
            "Expand each TNEWS class name into multiple semantically related label descriptions.",
            "Regularize or initialize classifier prototypes from label-word embeddings.",
            "Use calibration so frequent or overly generic label words do not dominate.",
        ]
        requirements = [
            "Create audited TNEWS label description file.",
            "Implement multi-label-word label_semantic head or prototype initialization.",
            "Add ablation: raw class id vs class name vs expanded label words.",
        ]
        theory = ["Multiple label words reduce verbalizer sparsity and class-name mismatch."]
        hints = MethodModuleHints(head=[HeadType.LABEL_SEMANTIC], loss=[LossType.LABEL_ANCHOR_CONTRASTIVE])
        takeaways = ["This is a direct innovation source for a TNEWS label-semantics classifier."]
        risks = ["External label words may inject wrong semantics for fine-grained news categories."]
        expected = "Likely helps if TNEWS errors concentrate among semantically close categories."
    elif "r-drop" in title_lower or "regularized dropout" in title_lower:
        model_name = "R-Drop: dropout consistency regularization"
        core_mechanism = (
            "R-Drop runs the same example twice under different dropout masks and optimizes cross-entropy plus "
            "bidirectional KL divergence between the two predictive distributions."
        )
        transferable = [
            "Apply rdrop loss to RoBERTa TNEWS fine-tuning to reduce seed variance.",
            "Combine with label smoothing only after checking underfitting in smoke/screen budgets.",
        ]
        requirements = ["Use existing rdrop runner path.", "Tune KL weight under screen budget.", "Compare seed variance against CE baseline."]
        theory = ["Dropout-induced prediction consistency can reduce overfitting and improve generalization."]
        hints = MethodModuleHints(loss=[LossType.RDROP])
        takeaways = ["Low-risk executable regularization baseline; not novel alone but useful as a building block."]
        risks = ["Can underfit if smoke budget is too small.", "May add cost because each batch needs two stochastic passes."]
        expected = "May improve robustness and reduce variance, especially for RoBERTa fine-tuning."
    elif "fewclue" in title_lower:
        model_name = "FewCLUE benchmark and prompt-learning evidence"
        core_mechanism = (
            "FewCLUE is primarily a Chinese few-shot benchmark. Its value here is benchmark/task evidence and "
            "prompt-learning baselines across Chinese NLU tasks, not a single new model architecture."
        )
        transferable = [
            "Use FewCLUE prompt-learning results to constrain which mechanisms are plausible on Chinese tasks.",
            "Use it as evidence that label semantics and prompt formulation matter in Chinese classification.",
        ]
        requirements = ["Treat as benchmark/SOTA context, not a standalone experiment spec."]
        theory = ["Chinese few-shot classification is sensitive to prompt formulation and class semantics."]
        hints = MethodModuleHints(head=[HeadType.LABEL_SEMANTIC])
        takeaways = ["Good for research direction; weak as direct algorithm source."]
        risks = ["Benchmark-level evidence may not transfer to full-data TNEWS fine-tuning."]
        expected = "Improves research framing and hypothesis selection, not directly a runnable model."
        worth = False
        quality = "benchmark_context_only"
    elif "improved prompt learning and unlabeled data" in title_lower:
        model_name = "Improved prompt learning with unlabeled-data self-training"
        core_mechanism = (
            "The method combines prompt-based Chinese few-shot classification with unlabeled data, typically by "
            "using high-confidence predictions or consistency-style supervision to expand the effective training signal."
        )
        transferable = [
            "Use public train-only unlabeled views or confidence-filtered pseudo labels without touching test labels.",
            "Combine label_semantic head with token masking consistency to mimic unlabeled robustness.",
        ]
        requirements = ["Add pseudo-label cache and confidence threshold if implementing fully.", "Audit that official validation/test labels are never used as unlabeled training signal."]
        theory = ["Unlabeled data can stabilize prompt decisions when labeled examples are sparse."]
        hints = MethodModuleHints(head=[HeadType.LABEL_SEMANTIC], augmentation=[AugmentationType.TOKEN_MASK])
        takeaways = ["Promising direction but needs strict leakage controls."]
        risks = ["Pseudo-label confirmation bias can hurt lockbox.", "Full self-training loop increases runtime."]
        expected = "Could help if extra unlabeled public TNEWS text is available; otherwise use as consistency augmentation inspiration."
    return PaperMethodAnalysis(
        paper_id=paper.paper_id,
        title=paper.title,
        source_url=paper.source_url,
        analysis_depth=depth,
        evidence_chars=len(evidence),
        algorithm_family=family,
        model_or_algorithm=model_name,
        core_mechanism=core_mechanism,
        training_objective=objective,
        theoretical_basis=theory,
        transferable_mechanisms=list(dict.fromkeys(transferable)),
        runner_mapping=hints,
        implementation_requirements=list(dict.fromkeys(requirements)),
        expected_effect_on_tnews=expected,
        risks=list(dict.fromkeys(risks)),
        novelty_takeaways=list(dict.fromkeys(takeaways)),
        quality_assessment=quality,
        worth_synthesizing=worth,
    )


def analyze_paper_methods(task: ResearchTaskConfig, task_id: str | None = None) -> list[PaperMethodAnalysis]:
    board = select_leaderboard(task, task_id)
    root = task_dir(task, board.task_id)
    evidence_path = root / "paper_evidence.jsonl"
    candidates = [
        PaperEvidence.model_validate(row)
        for row in _jsonl_rows(evidence_path)
        if not row.get("error") and (row.get("quality_pass") or row.get("source_name") == "bootstrap")
    ]
    papers_by_title: dict[str, PaperEvidence] = {}
    for paper in sorted(candidates, key=lambda item: (item.quality_pass, item.source_name == "bootstrap", item.quality_score), reverse=True):
        key = paper.title.lower().strip()
        papers_by_title.setdefault(key, paper)
    papers = list(papers_by_title.values())
    analyses = [_analyze_single_paper(paper) for paper in papers]
    dump_jsonl(root / "paper_analysis.jsonl", analyses)
    return analyses


def _llm_available(task: ResearchTaskConfig) -> bool:
    return task.llm.provider == LLMProvider.OPENAI and bool(os.getenv(task.llm.api_key_env))


def _rulebased_insight_review(analysis: PaperMethodAnalysis) -> PaperInsightReview:
    algorithm_steps = [
        item
        for item in [
            analysis.core_mechanism,
            *analysis.transferable_mechanisms[:3],
            *analysis.implementation_requirements[:3],
        ]
        if item
    ][:5]
    evidence_strength = 0.75 if analysis.analysis_depth == "full_text_pdf" else 0.45
    implementation_clarity = min(1.0, 0.2 + 0.18 * len(analysis.implementation_requirements))
    transferability = min(1.0, 0.15 + 0.18 * len(analysis.transferable_mechanisms))
    novelty = 0.65 if analysis.algorithm_family not in {"benchmark_and_evaluation", ""} else 0.25
    understanding = min(1.0, 0.25 + 0.15 * len(algorithm_steps) + (0.15 if analysis.training_objective else 0.0))
    approved = (
        analysis.worth_synthesizing
        and evidence_strength >= 0.45
        and implementation_clarity >= 0.45
        and transferability >= 0.45
        and novelty >= 0.5
    )
    missing: list[str] = []
    if analysis.analysis_depth != "full_text_pdf":
        missing.append("full_text_not_available")
    if not analysis.implementation_requirements:
        missing.append("implementation_plan_missing")
    if not analysis.training_objective:
        missing.append("training_objective_missing")
    return PaperInsightReview(
        paper_id=analysis.paper_id,
        title=analysis.title,
        reviewer="structured_fallback",
        confidence=0.45,
        algorithm_understanding_score=round(understanding, 3),
        transferability_score=round(transferability, 3),
        novelty_source_score=round(novelty, 3),
        implementation_clarity_score=round(implementation_clarity, 3),
        evidence_strength_score=round(evidence_strength, 3),
        key_algorithm_steps=algorithm_steps,
        missing_information=missing,
        critique=(
            "Fallback structural review only. It checks whether the extracted analysis contains "
            "algorithm steps, transfer path, and implementation requirements, but it is not a "
            "frontier-model semantic review."
        ),
        approved_for_synthesis=approved,
    )


def _llm_insight_review(task: ResearchTaskConfig, analysis: PaperMethodAnalysis) -> PaperInsightReview:
    schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number"},
            "algorithm_understanding_score": {"type": "number"},
            "transferability_score": {"type": "number"},
            "novelty_source_score": {"type": "number"},
            "implementation_clarity_score": {"type": "number"},
            "evidence_strength_score": {"type": "number"},
            "key_algorithm_steps": {"type": "array", "items": {"type": "string"}},
            "missing_information": {"type": "array", "items": {"type": "string"}},
            "critique": {"type": "string"},
            "approved_for_synthesis": {"type": "boolean"},
        },
        "required": [
            "confidence",
            "algorithm_understanding_score",
            "transferability_score",
            "novelty_source_score",
            "implementation_clarity_score",
            "evidence_strength_score",
            "key_algorithm_steps",
            "missing_information",
            "critique",
            "approved_for_synthesis",
        ],
        "additionalProperties": False,
    }
    prompt = (
        "You are a senior ML research critic for a CLUE TNEWS leaderboard project. "
        "Review whether this extracted paper analysis is strong enough to inspire a real, "
        "implementable algorithmic innovation. Reject vague, non-transferable, or non-local ideas. "
        "Scores must be 0-1.\n\n"
        f"Paper analysis JSON:\n{json.dumps(analysis.model_dump(mode='json'), ensure_ascii=False)}"
    )
    payload = _structured_llm(task, prompt, schema, "paper_insight_review")
    return PaperInsightReview(
        paper_id=analysis.paper_id,
        title=analysis.title,
        reviewer=f"openai:{task.llm.model}",
        confidence=_clip_score(float(payload["confidence"])),
        algorithm_understanding_score=_clip_score(float(payload["algorithm_understanding_score"])),
        transferability_score=_clip_score(float(payload["transferability_score"])),
        novelty_source_score=_clip_score(float(payload["novelty_source_score"])),
        implementation_clarity_score=_clip_score(float(payload["implementation_clarity_score"])),
        evidence_strength_score=_clip_score(float(payload["evidence_strength_score"])),
        key_algorithm_steps=list(payload["key_algorithm_steps"]),
        missing_information=list(payload["missing_information"]),
        critique=str(payload["critique"]),
        approved_for_synthesis=bool(payload["approved_for_synthesis"]),
    )


def review_paper_insights(task: ResearchTaskConfig, task_id: str | None = None) -> list[PaperInsightReview]:
    board = select_leaderboard(task, task_id)
    root = task_dir(task, board.task_id)
    analysis_path = root / "paper_analysis.jsonl"
    if not analysis_path.exists():
        analyze_paper_methods(task, board.task_id)
    analyses = [PaperMethodAnalysis.model_validate(row) for row in _jsonl_rows(analysis_path)]
    reviews: list[PaperInsightReview] = []
    for analysis in analyses:
        if _llm_available(task):
            try:
                reviews.append(_llm_insight_review(task, analysis))
                continue
            except Exception as exc:
                fallback = _rulebased_insight_review(analysis)
                reviews.append(
                    fallback.model_copy(
                        update={
                            "critique": f"LLM review failed: {exc}. {fallback.critique}",
                            "approved_for_synthesis": False if task.agent_research.require_llm_insight_review else fallback.approved_for_synthesis,
                        }
                    )
                )
                continue
        reviews.append(_rulebased_insight_review(analysis))
    dump_jsonl(root / "paper_insight_reviews.jsonl", reviews)
    return reviews


def assess_research_readiness(task: ResearchTaskConfig, task_id: str | None = None) -> ResearchReadinessReport:
    board = select_leaderboard(task, task_id)
    root = task_dir(task, board.task_id)
    analyses = [PaperMethodAnalysis.model_validate(row) for row in _jsonl_rows(root / "paper_analysis.jsonl")]
    if not analyses:
        analyses = analyze_paper_methods(task, board.task_id)
    reviews = review_paper_insights(task, board.task_id)
    quality_payload = {}
    quality_path = root / "paper_quality_report.json"
    if quality_path.exists():
        quality_payload = json.loads(quality_path.read_text(encoding="utf-8"))
    mechanism_clusters: dict[str, int] = {}
    for analysis in analyses:
        mechanism_clusters[analysis.algorithm_family] = mechanism_clusters.get(analysis.algorithm_family, 0) + 1
    approved = [review for review in reviews if review.approved_for_synthesis]
    worth = [analysis for analysis in analyses if analysis.worth_synthesizing]
    full_text = [analysis for analysis in analyses if analysis.analysis_depth == "full_text_pdf"]
    llm_ready = _llm_available(task)
    blocking: list[str] = []
    if task.agent_research.require_llm_insight_review and not llm_ready:
        blocking.append("llm_insight_review_required_but_unavailable")
    if int(quality_payload.get("qualified_papers", 0)) < task.agent_research.min_qualified_papers:
        blocking.append(
            f"qualified_evidence_low:{quality_payload.get('qualified_papers', 0)}/{task.agent_research.min_qualified_papers}"
        )
    if len(worth) < task.agent_research.min_worth_synthesizing_papers:
        blocking.append(f"worth_synthesizing_low:{len(worth)}/{task.agent_research.min_worth_synthesizing_papers}")
    if len(full_text) < task.agent_research.min_full_text_analyses:
        blocking.append(f"full_text_analysis_low:{len(full_text)}/{task.agent_research.min_full_text_analyses}")
    if len(approved) < task.agent_research.min_worth_synthesizing_papers:
        blocking.append(f"approved_insights_low:{len(approved)}/{task.agent_research.min_worth_synthesizing_papers}")
    status = "ready_for_idea_synthesis" if not blocking else "blocked"
    strongest = [
        family
        for family, _ in sorted(mechanism_clusters.items(), key=lambda item: item[1], reverse=True)
        if family not in {"benchmark_and_evaluation", ""}
    ][:5]
    report = ResearchReadinessReport(
        task_id=board.task_id,
        dataset=board.dataset,
        status=status,
        reviewer=f"openai:{task.llm.model}" if llm_ready else "structured_fallback",
        llm_review_required=task.agent_research.require_llm_insight_review,
        llm_review_available=llm_ready,
        paper_count=len(analyses),
        qualified_evidence_count=int(quality_payload.get("qualified_papers", 0)),
        worth_synthesizing_count=len(worth),
        full_text_analysis_count=len(full_text),
        approved_insight_count=len(approved),
        mechanism_clusters=mechanism_clusters,
        strongest_mechanisms=strongest,
        blocking_reasons=blocking,
        next_actions=[
            "Enable provider=openai with OPENAI_API_KEY for semantic paper and idea review."
            if "llm_insight_review_required_but_unavailable" in blocking
            else "Proceed to idea synthesis with reviewed paper insights."
        ],
    )
    _write_json(root / "research_readiness_report.json", report.model_dump(mode="json"))
    return report


def analyze_papers(task: ResearchTaskConfig, task_id: str | None = None) -> list[MethodCard]:
    board = select_leaderboard(task, task_id)
    root = task_dir(task, board.task_id)
    evidence_path = task_dir(task, board.task_id) / "paper_evidence.jsonl"
    rows = [row for row in _jsonl_rows(evidence_path) if not row.get("error") and row.get("quality_pass")]
    quality_path = root / "paper_quality_report.json"
    if not quality_path.exists() and evidence_path.exists():
        evaluated, report = evaluate_paper_quality(task, board, [PaperEvidence.model_validate(row) for row in _jsonl_rows(evidence_path)])
        dump_jsonl(root / "paper_evidence.jsonl", sorted(evaluated, key=lambda item: item.quality_score, reverse=True))
        _write_json(quality_path, report.model_dump(mode="json"))
        rows = [paper.model_dump(mode="json") for paper in evaluated if not paper.error and paper.quality_pass]
    analyses = analyze_paper_methods(task, board.task_id)
    if len(rows) < task.agent_research.min_qualified_papers:
        _write_json(
            root / "method_card_error.json",
            {
                "error": "insufficient_qualified_paper_evidence",
                "qualified_papers": len(rows),
                "paper_analyses": len(analyses),
                "min_required": task.agent_research.min_qualified_papers,
                "action": "rerun scout-papers with improved sources/queries before analyzing papers",
            },
        )
        dump_jsonl(root / "method_cards.jsonl", [])
        dump_jsonl(task.resolve_path(task.method_cards_path), [])
        return []
    backend = build_backend(task)
    cards = backend.create_method_cards(
        task,
        [
            {
                "paper_id": analysis.paper_id,
                "title": analysis.title,
                "source": analysis.source_url,
                "abstract": analysis.core_mechanism,
                "tags": [analysis.algorithm_family],
                "expected_gain": analysis.expected_effect_on_tnews,
                "risks": analysis.risks,
                "assumptions": analysis.implementation_requirements,
            }
            for analysis in analyses
            if analysis.worth_synthesizing
        ],
    )
    dump_jsonl(root / "method_cards.jsonl", cards)
    dump_jsonl(task.resolve_path(task.method_cards_path), cards)
    return cards


def _idea(
    task_id: str,
    title: str,
    source_ids: list[str],
    mechanism: str,
    difference: str,
    modules: MethodModuleHints,
    hyperparameters: dict[str, Any] | None = None,
    ticket: str | None = None,
) -> IdeaCard:
    payload = {"task_id": task_id, "title": title, "sources": source_ids, "mechanism": mechanism}
    return IdeaCard(
        idea_id=stable_hash(payload, prefix="idea-"),
        task_id=task_id,
        title=title,
        source_paper_ids=source_ids,
        mechanism=mechanism,
        difference_from_baseline=difference,
        expected_gain="Improve validation accuracy by changing representation or supervision rather than only tuning scalar hyperparameters.",
        implementation_plan="Map the mechanism to an executable ExperimentSpec when the existing runner supports it.",
        mapped_modules=modules,
        hyperparameters=hyperparameters or {},
        risks=["May fail if TNEWS label ambiguity differs from the source paper setting."],
        implementation_ticket=ticket,
    )


def _llm_synthesize_ideas(task: ResearchTaskConfig, board: LeaderboardTask, method_cards: list[dict[str, Any]]) -> list[IdeaCard]:
    schema = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "array",
                "minItems": task.agent_research.min_idea_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "source_paper_ids": {"type": "array", "items": {"type": "string"}},
                        "mechanism": {"type": "string"},
                        "difference_from_baseline": {"type": "string"},
                        "expected_gain": {"type": "string"},
                        "implementation_plan": {"type": "string"},
                        "pooling": {"type": "string", "enum": [item.value for item in PoolingType]},
                        "head": {"type": "string", "enum": [item.value for item in HeadType]},
                        "loss": {"type": "string", "enum": [item.value for item in LossType]},
                        "schedule": {"type": "string", "enum": [item.value for item in ScheduleType]},
                        "augmentation": {"type": "string", "enum": [item.value for item in AugmentationType]},
                        "hyperparameters": {"type": "object"},
                        "risks": {"type": "array", "items": {"type": "string"}},
                        "implementation_ticket": {"type": ["string", "null"]},
                    },
                    "required": [
                        "title",
                        "source_paper_ids",
                        "mechanism",
                        "difference_from_baseline",
                        "expected_gain",
                        "implementation_plan",
                        "pooling",
                        "head",
                        "loss",
                        "schedule",
                        "augmentation",
                        "hyperparameters",
                        "risks",
                        "implementation_ticket",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["ideas"],
        "additionalProperties": False,
    }
    prompt = (
        "You are the Idea Synthesizer for a CLUE leaderboard research system. "
        "Generate diverse, paper-grounded, executable innovation candidates for the task. "
        "Avoid producing only scalar hyperparameter tuning. Use only the provided runner modules.\n\n"
        f"Task: {board.task_id} / {board.display_name}\n"
        f"Allowed pooling: {[item.value for item in PoolingType]}\n"
        f"Allowed head: {[item.value for item in HeadType]}\n"
        f"Allowed loss: {[item.value for item in LossType]}\n"
        f"Allowed schedule: {[item.value for item in ScheduleType]}\n"
        f"Allowed augmentation: {[item.value for item in AugmentationType]}\n"
        f"Method cards JSON:\n{json.dumps(method_cards[:12], ensure_ascii=False)}"
    )
    payload = _structured_llm(task, prompt, schema, "idea_cards")
    ideas: list[IdeaCard] = []
    for row in payload["ideas"]:
        modules = MethodModuleHints(
            pooling=[PoolingType(row["pooling"])],
            head=[HeadType(row["head"])],
            loss=[LossType(row["loss"])],
            schedule=[ScheduleType(row["schedule"])],
            augmentation=[AugmentationType(row["augmentation"])],
        )
        ideas.append(
            IdeaCard(
                idea_id=stable_hash({"task": board.task_id, "title": row["title"], "mechanism": row["mechanism"]}, prefix="idea-"),
                task_id=board.task_id,
                title=row["title"],
                source_paper_ids=row["source_paper_ids"],
                mechanism=row["mechanism"],
                difference_from_baseline=row["difference_from_baseline"],
                expected_gain=row["expected_gain"],
                implementation_plan=row["implementation_plan"],
                mapped_modules=modules,
                hyperparameters=row["hyperparameters"],
                risks=row["risks"],
                implementation_ticket=row["implementation_ticket"],
            )
        )
    return ideas


def synthesize_ideas(task: ResearchTaskConfig, task_id: str | None = None) -> list[IdeaCard]:
    board = select_leaderboard(task, task_id)
    root = task_dir(task, board.task_id)
    readiness = assess_research_readiness(task, board.task_id)
    if readiness.status != "ready_for_idea_synthesis":
        _write_json(
            root / "idea_synthesizer_error.json",
            {
                "error": "research_readiness_blocked",
                "blocking_reasons": readiness.blocking_reasons,
                "fallback": "blocked",
            },
        )
        dump_jsonl(root / "idea_cards.jsonl", [])
        return []
    rows = _jsonl_rows(root / "method_cards.jsonl")
    if len(rows) < task.agent_research.min_qualified_papers:
        _write_json(
            root / "idea_synthesizer_error.json",
            {
                "error": "insufficient_method_cards",
                "method_cards": len(rows),
                "min_required": task.agent_research.min_qualified_papers,
                "fallback": "blocked",
            },
        )
        dump_jsonl(root / "idea_cards.jsonl", [])
        return []
    source_ids = [str(row.get("paper_id")) for row in rows[:3]] or ["no-paper-evidence"]
    try:
        ideas = _llm_synthesize_ideas(task, board, rows)
        dump_jsonl(task_dir(task, board.task_id) / "idea_cards.jsonl", ideas)
        return ideas
    except Exception as exc:
        _write_json(
            task_dir(task, board.task_id) / "idea_synthesizer_error.json",
            {"error": str(exc), "fallback": "rulebased"},
        )
    ideas = [
        _idea(
            board.task_id,
            "Label-anchored verbalizer contrastive classifier",
            source_ids[:2],
            "Initialize class decision vectors with Chinese label descriptions and add a contrastive loss that pulls samples toward their label semantic anchor.",
            "Baseline treats labels as anonymous ids; this makes label meaning an explicit train-time geometry constraint.",
            MethodModuleHints(pooling=[PoolingType.CLS], head=[HeadType.LABEL_SEMANTIC], loss=[LossType.LABEL_ANCHOR_CONTRASTIVE]),
            {"label_smoothing": 0.05, "label_anchor_weight": 0.2, "label_anchor_temperature": 0.07},
        ),
        _idea(
            board.task_id,
            "Contrastive mean pooling for confusing short texts",
            source_ids[:2],
            "Shape sentence representations with supervised contrastive pressure before classification.",
            "Baseline optimizes only cross-entropy; this adds class-aware representation geometry.",
            MethodModuleHints(pooling=[PoolingType.MEAN], head=[HeadType.MLP], loss=[LossType.CE_SUPCON], schedule=[ScheduleType.LAYERWISE_LR_DECAY]),
            {"supcon_weight": 0.08, "lrd_decay": 0.9},
        ),
        _idea(
            board.task_id,
            "Attention pooling with mild focal calibration",
            source_ids[:2],
            "Use learnable token pooling and a mild hard-example loss for short ambiguous titles.",
            "Baseline uses CLS only; this lets the model emphasize informative title spans.",
            MethodModuleHints(pooling=[PoolingType.ATTENTION], head=[HeadType.LINEAR], loss=[LossType.FOCAL]),
            {"focal_gamma": 1.0},
        ),
        _idea(
            board.task_id,
            "Token masking robustness screen",
            source_ids[:2],
            "Use lightweight token masking to reduce shortcut reliance in title classification.",
            "Baseline sees only clean titles; this trains under small lexical perturbations.",
            MethodModuleHints(pooling=[PoolingType.CLS], head=[HeadType.LINEAR], loss=[LossType.CE], augmentation=[AugmentationType.TOKEN_MASK]),
        ),
        _idea(
            board.task_id,
            "Span cutoff with label smoothing",
            source_ids[:2],
            "Mask short contiguous spans and smooth labels to improve robustness on entity-heavy titles.",
            "Baseline can overfit isolated entities; this encourages broader evidence use.",
            MethodModuleHints(pooling=[PoolingType.CLS], head=[HeadType.LINEAR], loss=[LossType.LABEL_SMOOTHING], augmentation=[AugmentationType.SPAN_CUTOFF]),
            {"label_smoothing": 0.05},
        ),
        _idea(
            board.task_id,
            "Confusion-class specialist ensemble",
            source_ids[:2],
            "Train specialist classifiers for the most confused label groups and route uncertain samples.",
            "Baseline is one flat classifier; this changes decision structure for leaderboard errors.",
            MethodModuleHints(),
            ticket="Implement confusion-group routing and ensemble inference before this can run.",
        ),
    ]
    dump_jsonl(task_dir(task, board.task_id) / "idea_cards.jsonl", ideas)
    return ideas


def _fallback_idea_review(idea: IdeaCard, has_sources: bool, executable: bool) -> IdeaQualityReview:
    novelty = 0.35 + (0.25 if idea.difference_from_baseline else 0.0) + (0.15 if len(idea.source_paper_ids) >= 2 else 0.0)
    evidence = 0.65 if has_sources else 0.2
    baseline_difference = 0.65 if idea.difference_from_baseline else 0.25
    feasibility = 0.8 if executable else 0.25
    potential = 0.6 if has_sources and executable else 0.25
    risk = 0.25 if executable else 0.8
    approved = has_sources and executable and min(evidence, novelty, baseline_difference, feasibility, potential) >= 0.5
    return IdeaQualityReview(
        idea_id=idea.idea_id,
        title=idea.title,
        reviewer="structured_fallback",
        evidence_grounding_score=_clip_score(evidence),
        novelty_score=_clip_score(novelty),
        baseline_difference_score=_clip_score(baseline_difference),
        feasibility_score=_clip_score(feasibility),
        leaderboard_potential_score=_clip_score(potential),
        risk_score=_clip_score(risk),
        critique=(
            "Fallback idea review only. It checks structured evidence links and implementation fields, "
            "but does not semantically judge whether the idea is truly novel."
        ),
        required_ablation=["Compare against strongest reproduced baseline.", "Run lockbox validation before any breakthrough claim."],
        required_implementation_checks=["Verify no test/validation leakage.", "Measure runtime cost against baseline."],
        approved_for_experiment=approved,
    )


def _llm_idea_review(task: ResearchTaskConfig, board: LeaderboardTask, idea: IdeaCard, analyses: list[dict[str, Any]]) -> IdeaQualityReview:
    schema = {
        "type": "object",
        "properties": {
            "evidence_grounding_score": {"type": "number"},
            "novelty_score": {"type": "number"},
            "baseline_difference_score": {"type": "number"},
            "feasibility_score": {"type": "number"},
            "leaderboard_potential_score": {"type": "number"},
            "risk_score": {"type": "number"},
            "critique": {"type": "string"},
            "required_ablation": {"type": "array", "items": {"type": "string"}},
            "required_implementation_checks": {"type": "array", "items": {"type": "string"}},
            "approved_for_experiment": {"type": "boolean"},
        },
        "required": [
            "evidence_grounding_score",
            "novelty_score",
            "baseline_difference_score",
            "feasibility_score",
            "leaderboard_potential_score",
            "risk_score",
            "critique",
            "required_ablation",
            "required_implementation_checks",
            "approved_for_experiment",
        ],
        "additionalProperties": False,
    }
    prompt = (
        "You are the Critic for a CLUE leaderboard research system. Decide whether this idea is "
        "a real algorithmic research candidate, not just tuning. Require paper grounding, a clear "
        "difference from RoBERTa/BERT fine-tuning baselines, local implementability on M2/16GB, and "
        "a plausible path to leaderboard improvement. Scores are 0-1.\n\n"
        f"Task: {board.task_id} / {board.display_name}\n"
        f"Idea JSON:\n{json.dumps(idea.model_dump(mode='json'), ensure_ascii=False)}\n\n"
        f"Relevant paper analyses:\n{json.dumps(analyses[:12], ensure_ascii=False)}"
    )
    payload = _structured_llm(task, prompt, schema, "idea_quality_review")
    return IdeaQualityReview(
        idea_id=idea.idea_id,
        title=idea.title,
        reviewer=f"openai:{task.llm.model}",
        evidence_grounding_score=_clip_score(float(payload["evidence_grounding_score"])),
        novelty_score=_clip_score(float(payload["novelty_score"])),
        baseline_difference_score=_clip_score(float(payload["baseline_difference_score"])),
        feasibility_score=_clip_score(float(payload["feasibility_score"])),
        leaderboard_potential_score=_clip_score(float(payload["leaderboard_potential_score"])),
        risk_score=_clip_score(float(payload["risk_score"])),
        critique=str(payload["critique"]),
        required_ablation=list(payload["required_ablation"]),
        required_implementation_checks=list(payload["required_implementation_checks"]),
        approved_for_experiment=bool(payload["approved_for_experiment"]),
    )


def critique_ideas(task: ResearchTaskConfig, task_id: str | None = None) -> list[IdeaCard]:
    board = select_leaderboard(task, task_id)
    root = task_dir(task, board.task_id)
    rows = [IdeaCard.model_validate(row) for row in _jsonl_rows(root / "idea_cards.jsonl")]
    analyses = _jsonl_rows(root / "paper_analysis.jsonl")
    reviewed: list[IdeaCard] = []
    idea_reviews: list[IdeaQualityReview] = []
    for idea in rows:
        has_sources = bool(idea.source_paper_ids) and idea.source_paper_ids != ["no-paper-evidence"]
        executable = not idea.implementation_ticket and any(
            [idea.mapped_modules.pooling, idea.mapped_modules.head, idea.mapped_modules.loss, idea.mapped_modules.augmentation]
        )
        if _llm_available(task):
            try:
                idea_review = _llm_idea_review(task, board, idea, analyses)
            except Exception as exc:
                fallback = _fallback_idea_review(idea, has_sources, executable)
                idea_review = fallback.model_copy(
                    update={
                        "critique": f"LLM idea review failed: {exc}. {fallback.critique}",
                        "approved_for_experiment": False if task.agent_research.require_llm_idea_review else fallback.approved_for_experiment,
                    }
                )
        else:
            fallback = _fallback_idea_review(idea, has_sources, executable)
            idea_review = fallback.model_copy(
                update={
                    "approved_for_experiment": False
                    if task.agent_research.require_llm_idea_review
                    else fallback.approved_for_experiment,
                    "critique": (
                        "LLM idea review required but unavailable. "
                        + fallback.critique
                        if task.agent_research.require_llm_idea_review
                        else fallback.critique
                    ),
                }
            )
        accepted = idea_review.approved_for_experiment
        idea_reviews.append(idea_review)
        reviewed.append(
            idea.model_copy(
                update={
                    "novelty_score": idea_review.novelty_score,
                    "feasibility_score": idea_review.feasibility_score,
                    "leaderboard_potential_score": idea_review.leaderboard_potential_score,
                    "cost_risk": "low" if executable else "high",
                    "accepted": accepted,
                    "rejection_reason": None if accepted else idea_review.critique,
                }
            )
        )
    dump_jsonl(root / "idea_quality_reviews.jsonl", idea_reviews)
    dump_jsonl(root / "idea_cards.reviewed.jsonl", reviewed)
    return reviewed


def design_experiments(task: ResearchTaskConfig, registry: RunRegistry, task_id: str | None = None) -> list[ExperimentSpec]:
    board = select_leaderboard(task, task_id)
    ideas = [IdeaCard.model_validate(row) for row in _jsonl_rows(task_dir(task, board.task_id) / "idea_cards.reviewed.jsonl")]
    baseline = task.baselines.transformer[-1]
    existing = registry.existing_signatures()
    specs: list[ExperimentSpec] = []
    for idea in ideas:
        if not idea.accepted:
            continue
        pooling = (idea.mapped_modules.pooling or [PoolingType.CLS])[0]
        head = (idea.mapped_modules.head or [HeadType.LINEAR])[0]
        loss = (idea.mapped_modules.loss or [LossType.CE])[0]
        schedule = (idea.mapped_modules.schedule or [ScheduleType.FULL_FT])[0]
        augmentation = (idea.mapped_modules.augmentation or [AugmentationType.NONE])[0]
        spec = ExperimentSpec(
            experiment_id=stable_hash({"idea": idea.idea_id, "dataset": board.dataset, "hp": idea.hyperparameters}, prefix="agent-"),
            dataset=board.dataset,
            baseline_name=baseline.name,
            model_name=f"{baseline.name}-{pooling.value}-{head.value}-{loss.value}-{augmentation.value}",
            model_id=baseline.model_id,
            pooling=pooling,
            head=head,
            loss=loss,
            schedule=schedule,
            augmentation=augmentation,
            budget=BudgetLevel.SCREEN,
            seeds=task.random_seeds[: task.budget_profile(BudgetLevel.SCREEN).seed_count],
            rationale=f"{idea.title}: {idea.mechanism}",
            expected_risk=idea.cost_risk,
            tags=["research_agent", idea.idea_id],
            hyperparameters=idea.hyperparameters,
        )
        if spec.signature in existing:
            continue
        specs.append(spec)
        if len(specs) >= task.agent_research.per_leaderboard_candidate_budget:
            break
    target = task_dir(task, board.task_id) / "experiment_specs.yaml"
    dump_yaml(target, {"experiments": [spec.model_dump(mode="json") for spec in specs]})
    return specs


def decide_research(task: ResearchTaskConfig, registry: RunRegistry, task_id: str | None = None) -> ResearchDecision:
    boards = load_leaderboards(task)
    board = select_leaderboard(task, task_id)
    runs = registry.fetch_runs("dataset = ? AND status = 'completed'", [board.dataset])
    tested = int(len(runs[runs["kind"] != "baseline"])) if not runs.empty else 0
    best_validation = None
    baseline_validation = None
    promote_experiment_id = None
    status = LeaderboardStatus.ACTIVE
    reason = "Need more candidates."
    if not runs.empty:
        baselines = runs[runs["kind"] == "baseline"]
        proposals = runs[runs["kind"] != "baseline"]
        if not baselines.empty:
            baseline_summary = summarize_runs(baselines).iloc[0]
            baseline_validation = float(baseline_summary["validation_mean"])
        if not proposals.empty:
            proposal_summary = summarize_runs(proposals).iloc[0]
            best_validation = float(proposal_summary["validation_mean"])
            lockbox_delta = 0.0
            if baseline_validation is not None and not baselines.empty:
                baseline_lockbox = 0.0 if pd.isna(baseline_summary["lockbox_mean"]) else float(baseline_summary["lockbox_mean"])
                proposal_lockbox = 0.0 if pd.isna(proposal_summary["lockbox_mean"]) else float(proposal_summary["lockbox_mean"])
                lockbox_delta = proposal_lockbox - baseline_lockbox
            if baseline_validation is not None and best_validation > baseline_validation and lockbox_delta >= -task.agent_research.lockbox_regression_tolerance:
                status = LeaderboardStatus.PROMISING
                promote_experiment_id = str(proposal_summary["experiment_id"])
                reason = "Best candidate beats the strongest reproduced baseline and lockbox does not regress materially."
    if status == LeaderboardStatus.ACTIVE and tested >= task.agent_research.per_leaderboard_candidate_budget:
        status = LeaderboardStatus.ABANDONED
        reason = f"Candidate budget exhausted: {tested}/{task.agent_research.per_leaderboard_candidate_budget}."
    next_task_id = None
    if status == LeaderboardStatus.ABANDONED:
        for candidate in sorted(boards, key=lambda item: item.priority, reverse=True):
            if candidate.task_id != board.task_id and candidate.status != LeaderboardStatus.ABANDONED:
                next_task_id = candidate.task_id
                break
    decision = ResearchDecision(
        task_id=board.task_id,
        dataset=board.dataset,
        status=status,
        reason=reason,
        tested_candidates=tested,
        candidate_budget=task.agent_research.per_leaderboard_candidate_budget,
        best_validation=best_validation,
        best_baseline_validation=baseline_validation,
        promote_experiment_id=promote_experiment_id,
        next_task_id=next_task_id,
    )
    _write_json(task_dir(task, board.task_id) / "research_decision.json", decision.model_dump(mode="json"))
    return decision


def research_agents_cycle(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    task_id: str | None = None,
    dry_run: bool = True,
    max_papers: int = 12,
) -> ResearchDecision:
    board = select_leaderboard(task, task_id)
    scout_leaderboards(task)
    analyze_sota(task, board.task_id)
    scout_papers(task, board.task_id, max_papers=max_papers)
    analyze_papers(task, board.task_id)
    synthesize_ideas(task, board.task_id)
    critique_ideas(task, board.task_id)
    design_experiments(task, registry, board.task_id)
    if not dry_run:
        from .pipeline import run_loop

        run_loop(
            task.model_copy(update={"primary_dataset": board.dataset}),
            registry,
            BudgetLevel.SCREEN,
            task.agent_research.per_leaderboard_candidate_budget,
            proposal_file=task_dir(task, board.task_id) / "experiment_specs.yaml",
            skip_existing=True,
        )
    return decide_research(task, registry, board.task_id)


def export_submission(task: ResearchTaskConfig, run_id: str, output_path: str | Path | None = None) -> Path:
    registry = RunRegistry(task.registry_path)
    runs = registry.fetch_runs("run_id = ? AND status = 'completed'", [run_id])
    if runs.empty:
        raise ValueError(f"Completed run not found: {run_id}")
    run = runs.iloc[-1]
    artifact_dir = Path(str(run["artifact_dir"]))
    predictions = read_frame(artifact_dir / "test_predictions")
    target = Path(output_path) if output_path else task.output_root_path / "submissions" / f"{run_id}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for _, row in predictions.iterrows():
        sample_id = int(row["sample_id"]) if "sample_id" in row else int(_)
        label = str(row["label_name"]) if "label_name" in row else str(int(row["prediction"]))
        lines.append(json.dumps({"id": sample_id, "label": label}, ensure_ascii=False))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
