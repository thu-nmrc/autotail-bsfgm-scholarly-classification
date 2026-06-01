from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import (
    ExperimentSpec,
    LLMProvider,
    MethodCard,
    MethodModuleHints,
    PoolingType,
    HeadType,
    LossType,
    ScheduleType,
    AugmentationType,
    ResearchTaskConfig,
)
from .utils import stable_hash


class LLMBackend(ABC):
    @abstractmethod
    def create_method_cards(self, task: ResearchTaskConfig, paper_rows: list[dict[str, Any]]) -> list[MethodCard]:
        raise NotImplementedError

    @abstractmethod
    def propose_experiments(
        self,
        task: ResearchTaskConfig,
        method_cards: list[MethodCard],
        existing_signatures: set[str],
        budget: str,
    ) -> list[ExperimentSpec]:
        raise NotImplementedError


class RuleBasedBackend(LLMBackend):
    def create_method_cards(self, task: ResearchTaskConfig, paper_rows: list[dict[str, Any]]) -> list[MethodCard]:
        cards: list[MethodCard] = []
        for row in paper_rows:
            text = " ".join([row.get("title", ""), row.get("abstract", ""), " ".join(row.get("tags", []))]).lower()
            hints = MethodModuleHints()
            if "contrastive" in text:
                hints.loss.append(LossType.CE_SUPCON)
            if "dropout" in text or "consistency" in text:
                hints.loss.append(LossType.RDROP)
            if "focal" in text or "hard examples" in text or "imbalance" in text:
                hints.loss.append(LossType.FOCAL)
            if "prototype" in text or "metric" in text:
                hints.head.append(HeadType.PROTOTYPE)
            if "label_semantics" in text or "label semantics" in text or "verbalizer" in text or "prompt" in text:
                hints.head.append(HeadType.LABEL_SEMANTIC)
                hints.loss.append(LossType.LABEL_ANCHOR_CONTRASTIVE)
            if "attention" in text:
                hints.pooling.append(PoolingType.ATTENTION)
            if "smoothing" in text:
                hints.loss.append(LossType.LABEL_SMOOTHING)
            if "layerwise" in text:
                hints.schedule.append(ScheduleType.LAYERWISE_LR_DECAY)

            cards.append(
                MethodCard(
                    paper_id=row["paper_id"],
                    title=row["title"],
                    source=row["source"],
                    summary=row.get("abstract", "").strip(),
                    core_idea=row.get("abstract", "").strip(),
                    assumptions=row.get("assumptions", []),
                    expected_gain=row.get("expected_gain", "Potential validation accuracy lift under constrained search"),
                    risks=row.get("risks", ["May improve one split but fail to generalize"]),
                    tags=row.get("tags", []),
                    mapped_modules=hints,
                )
            )
        return cards

    def propose_experiments(
        self,
        task: ResearchTaskConfig,
        method_cards: list[MethodCard],
        existing_signatures: set[str],
        budget: str,
    ) -> list[ExperimentSpec]:
        seeds = task.random_seeds[: task.budgets[budget].seed_count]
        model = task.baselines.transformer[-1]
        base_grid: list[dict[str, Any]] = []
        for method in method_cards:
            uses_label_semantics = HeadType.LABEL_SEMANTIC in method.mapped_modules.head
            pooling_values = method.mapped_modules.pooling or ([PoolingType.CLS] if uses_label_semantics else [PoolingType.MEAN])
            head_values = method.mapped_modules.head or [HeadType.MLP]
            loss_values = method.mapped_modules.loss or ([LossType.LABEL_SMOOTHING] if uses_label_semantics else [LossType.CE])
            schedule_values = method.mapped_modules.schedule or [ScheduleType.FULL_FT]
            augmentation_values = method.mapped_modules.augmentation or [AugmentationType.NONE]
            for pooling in pooling_values:
                for head in head_values:
                    for loss in loss_values:
                        for schedule in schedule_values:
                            for augmentation in augmentation_values:
                                base_grid.append(
                                    {
                                        "pooling": pooling,
                                        "head": head,
                                        "loss": loss,
                                        "schedule": schedule,
                                        "augmentation": augmentation,
                                        "rationale": f"{method.title}: {method.expected_gain}",
                                        "tags": method.tags,
                                    }
                                )

        if not base_grid:
            base_grid = [
                {
                    "pooling": PoolingType.MEAN,
                    "head": HeadType.MLP,
                    "loss": LossType.LABEL_SMOOTHING,
                    "schedule": ScheduleType.LAYERWISE_LR_DECAY,
                    "augmentation": AugmentationType.NONE,
                    "rationale": "Default robust search candidate",
                    "tags": ["fallback"],
                },
                {
                    "pooling": PoolingType.ATTENTION,
                    "head": HeadType.PROTOTYPE,
                    "loss": LossType.CE_SUPCON,
                    "schedule": ScheduleType.FULL_FT,
                    "augmentation": AugmentationType.TOKEN_MASK,
                    "rationale": "Representation-focused candidate",
                    "tags": ["fallback"],
                },
            ]

        candidates: list[ExperimentSpec] = []
        for candidate in base_grid:
            spec = ExperimentSpec(
                experiment_id=stable_hash(
                    {
                        "dataset": task.primary_dataset,
                        "model_id": model.model_id,
                        **{key: value.value if hasattr(value, "value") else value for key, value in candidate.items() if key in {"pooling", "head", "loss", "schedule", "augmentation"}},
                    },
                    prefix="exp-",
                ),
                dataset=task.primary_dataset,
                baseline_name=model.name,
                model_name=f"{model.name}-{candidate['pooling'].value}-{candidate['head'].value}-{candidate['loss'].value}",
                model_id=model.model_id,
                pooling=candidate["pooling"],
                head=candidate["head"],
                loss=candidate["loss"],
                schedule=candidate["schedule"],
                augmentation=candidate["augmentation"],
                budget=budget,
                seeds=seeds,
                rationale=candidate["rationale"],
                tags=candidate["tags"],
            )
            if spec.signature in existing_signatures:
                continue
            candidates.append(spec)

        return candidates[: task.budgets[budget].max_candidates]


class OpenAIBackend(RuleBasedBackend):
    def __init__(self, task: ResearchTaskConfig) -> None:
        api_key = os.getenv(task.llm.api_key_env)
        if not api_key:
            raise RuntimeError(f"Environment variable {task.llm.api_key_env} is required for provider=openai")
        self.task = task
        self.api_key = api_key

    def create_method_cards(self, task: ResearchTaskConfig, paper_rows: list[dict[str, Any]]) -> list[MethodCard]:
        fallback_cards = super().create_method_cards(task, paper_rows)
        enriched_cards: list[MethodCard] = []
        for fallback, row in zip(fallback_cards, paper_rows, strict=True):
            schema = {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "core_idea": {"type": "string"},
                    "assumptions": {"type": "array", "items": {"type": "string"}},
                    "expected_gain": {"type": "string"},
                    "risks": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["summary", "core_idea", "assumptions", "expected_gain", "risks"],
                "additionalProperties": False,
            }
            prompt = (
                "Read the following paper note and produce compact JSON for a Chinese text classification "
                "research pipeline. Focus on transferable ideas for TNEWS.\n\n"
                f"Title: {row['title']}\n"
                f"Abstract: {row.get('abstract', '')}\n"
                f"Tags: {', '.join(row.get('tags', []))}\n"
            )
            try:
                response = self._structured_call(prompt, schema, name="method_card")
                enriched_cards.append(
                    fallback.model_copy(
                        update={
                            "summary": response["summary"],
                            "core_idea": response["core_idea"],
                            "assumptions": response["assumptions"],
                            "expected_gain": response["expected_gain"],
                            "risks": response["risks"],
                        }
                    )
                )
            except Exception:
                enriched_cards.append(fallback)
        return enriched_cards

    def _structured_call(self, prompt: str, schema: dict[str, Any], name: str) -> dict[str, Any]:
        payload = {
            "model": self.task.llm.model,
            "input": prompt,
            "temperature": self.task.llm.temperature,
            "max_output_tokens": self.task.llm.max_output_tokens,
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
            self.task.llm.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        output_text = data.get("output_text")
        if not output_text:
            raise RuntimeError(f"OpenAI response missing output_text: {data}")
        return json.loads(output_text)


def build_backend(task: ResearchTaskConfig) -> LLMBackend:
    if task.llm.provider == LLMProvider.OPENAI:
        return OpenAIBackend(task)
    return RuleBasedBackend()
