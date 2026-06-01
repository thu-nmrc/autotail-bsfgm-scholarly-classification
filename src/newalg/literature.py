from __future__ import annotations

import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .config import ResearchTaskConfig, dump_yaml


DEFAULT_LITERATURE_QUERIES = [
    "short text classification prompt label semantics",
    "text classification label smoothing r-drop",
    "Chinese short text classification prompt learning",
    "text classification label descriptions verbalizer",
]


CURATED_PAPERS: list[dict[str, Any]] = [
    {
        "paper_id": "label-semantics-short-text",
        "title": "Label Semantics and Prompt-style Classification for Short Text",
        "source": "curated:label-semantics",
        "abstract": (
            "Short text classification can benefit from making label meaning explicit. "
            "Instead of treating labels as anonymous ids, encode label descriptions and use "
            "them to initialize or regularize the classifier."
        ),
        "tags": ["label_semantics", "prompt", "short_text", "classification"],
        "expected_gain": "Improve confusing short-title categories by injecting class-name semantics into the classifier head.",
        "risks": ["Label descriptions may be too coarse or mismatched with dataset annotation policy."],
    },
    {
        "paper_id": "modern-encoder-classification",
        "title": "Modern Encoder Models for Discriminative Text Classification",
        "source": "curated:modern-encoder",
        "abstract": (
            "Recent encoder-only models such as DeBERTaV3 and ModernBERT improve representation "
            "quality through stronger pretraining and architecture changes. For local research, "
            "their ideas motivate conservative encoder fine-tuning recipes before larger model swaps."
        ),
        "tags": ["encoder", "deberta", "modernbert", "classification"],
        "expected_gain": "Guide local search toward strong encoder fine-tuning recipes with low structural risk.",
        "risks": ["A larger or unavailable backbone may exceed the local hardware budget."],
    },
    {
        "paper_id": "consistency-regularized-classification",
        "title": "Consistency Regularization for Neural Text Classification",
        "source": "curated:consistency",
        "abstract": (
            "R-Drop and related consistency methods regularize predictions from stochastic subnetworks. "
            "They are especially useful as low-risk loss modifications when the classifier is already strong."
        ),
        "tags": ["rdrop", "consistency", "dropout", "regularization"],
        "expected_gain": "Improve generalization without changing the encoder architecture.",
        "risks": ["The extra forward pass increases cost and the KL weight may need tuning."],
    },
    {
        "paper_id": "calibrated-loss-classification",
        "title": "Calibrated Losses for Fine-grained Text Classification",
        "source": "curated:calibrated-loss",
        "abstract": (
            "Label smoothing and focal loss change how confidence and hard examples are optimized. "
            "They are simple, local changes that can help fine-grained or noisy classification tasks."
        ),
        "tags": ["label_smoothing", "focal", "calibration", "loss"],
        "expected_gain": "Improve robustness on ambiguous TNEWS classes with minimal implementation risk.",
        "risks": ["Too much smoothing can underfit and focal loss can over-focus noisy labels."],
    },
]


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:64] or "paper"


def fetch_arxiv_papers(queries: list[str], max_results: int = 8, timeout: float = 20.0) -> list[dict[str, Any]]:
    papers: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_query = max(1, max_results // max(1, len(queries)))
    for query in queries:
        params = urllib.parse.urlencode(
            {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": per_query,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
        url = f"https://export.arxiv.org/api/query?{params}"
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = response.read()
        root = ET.fromstring(payload)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
            abstract = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
            source = entry.findtext("atom:id", default="", namespaces=ns) or url
            if not title or source in seen:
                continue
            seen.add(source)
            papers.append(
                {
                    "paper_id": _slug(title),
                    "title": title,
                    "source": source,
                    "abstract": abstract,
                    "tags": _infer_tags(title, abstract),
                    "expected_gain": "Potential transferable improvement for Chinese short text classification.",
                    "risks": ["May not transfer from the paper setting to TNEWS without adaptation."],
                }
            )
            if len(papers) >= max_results:
                return papers
    return papers


def _infer_tags(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    tags: list[str] = []
    for keyword, tag in [
        ("prompt", "prompt"),
        ("label", "label_semantics"),
        ("verbalizer", "label_semantics"),
        ("dropout", "rdrop"),
        ("r-drop", "rdrop"),
        ("contrastive", "contrastive"),
        ("focal", "focal"),
        ("smoothing", "label_smoothing"),
        ("attention", "attention"),
        ("prototype", "prototype"),
        ("modernbert", "modernbert"),
        ("deberta", "deberta"),
    ]:
        if keyword in text and tag not in tags:
            tags.append(tag)
    return tags or ["text_classification"]


def discover_literature(
    task: ResearchTaskConfig,
    output_path: str | Path,
    queries: list[str] | None = None,
    max_papers: int = 8,
    use_arxiv: bool = True,
) -> list[dict[str, Any]]:
    papers = list(CURATED_PAPERS)
    if use_arxiv:
        try:
            papers.extend(fetch_arxiv_papers(queries or DEFAULT_LITERATURE_QUERIES, max_results=max_papers))
        except Exception:
            # The curated seed keeps the zero-start loop usable offline.
            pass
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for paper in papers:
        key = str(paper.get("source") or paper.get("title"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    dump_yaml(output_path, {"papers": deduped[: max(max_papers, len(CURATED_PAPERS))]})
    return deduped
