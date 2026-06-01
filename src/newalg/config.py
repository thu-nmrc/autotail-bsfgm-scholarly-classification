from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class SourceType(StrEnum):
    HUGGINGFACE = "huggingface"
    LOCAL = "local"


class BudgetLevel(StrEnum):
    SMOKE = "smoke"
    SCREEN = "screen"
    CONFIRM = "confirm"
    FINAL = "final"


class PoolingType(StrEnum):
    CLS = "cls"
    MEAN = "mean"
    ATTENTION = "attention"


class HeadType(StrEnum):
    LINEAR = "linear"
    MLP = "mlp"
    PROTOTYPE = "prototype"
    LABEL_SEMANTIC = "label_semantic"
    LABEL_ATTENTION = "label_attention"
    RESIDUAL_LABEL_ATTENTION = "residual_label_attention"


class LossType(StrEnum):
    CE = "ce"
    FOCAL = "focal"
    LABEL_SMOOTHING = "label_smoothing"
    CLASS_BALANCED = "class_balanced"
    BALANCED_SOFTMAX = "balanced_softmax"
    LDAM = "ldam"
    LDAM_DRW = "ldam_drw"
    PRIOR_ADJUSTED_LABEL_SMOOTHING = "prior_adjusted_label_smoothing"
    TAIL_WEIGHTED_LABEL_SMOOTHING = "tail_weighted_label_smoothing"
    CE_SUPCON = "ce_supcon"
    LABEL_ANCHOR_CONTRASTIVE = "label_anchor_contrastive"
    RDROP = "rdrop"


class ScheduleType(StrEnum):
    FULL_FT = "full_ft"
    GRADUAL_UNFREEZE = "gradual_unfreeze"
    LAYERWISE_LR_DECAY = "layerwise_lr_decay"


class AugmentationType(StrEnum):
    NONE = "none"
    TOKEN_MASK = "token_mask"
    SPAN_CUTOFF = "span_cutoff"


class LLMProvider(StrEnum):
    RULEBASED = "rulebased"
    OPENAI = "openai"


class PromotionGate(StrEnum):
    BEAT_STRONG_BASELINE = "beat_strong_baseline"


class LeaderboardStatus(StrEnum):
    ACTIVE = "active"
    PROMISING = "promising"
    ABANDONED = "abandoned"
    READY_FOR_SUBMISSION = "ready_for_submission"


class DatasetConfig(BaseModel):
    source: SourceType = SourceType.HUGGINGFACE
    hf_path: str | None = None
    hf_name: str | None = None
    text_field: str
    label_field: str
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    max_length: int = 128
    local_path: str | None = None


class LLMConfig(BaseModel):
    provider: LLMProvider = LLMProvider.RULEBASED
    model: str = "gpt-5.4-mini"
    temperature: float = 0.1
    max_output_tokens: int = 2000
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1/responses"


class BudgetProfile(BaseModel):
    epochs: int
    sample_cap: int | None = None
    seed_count: int
    max_candidates: int


class BaselineModel(BaseModel):
    name: str
    model_id: str


class BaselineConfig(BaseModel):
    traditional: list[str] = Field(default_factory=list)
    transformer: list[BaselineModel] = Field(default_factory=list)


class TrainingConfig(BaseModel):
    train_batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    max_epochs: int = 3
    smoke_epochs: int = 1
    num_workers: int = 0


class AllowedModules(BaseModel):
    pooling: list[PoolingType]
    head: list[HeadType]
    loss: list[LossType]
    schedule: list[ScheduleType]
    augmentation: list[AugmentationType]


class StopConditions(BaseModel):
    max_total_runs: int = 30
    no_improvement_rounds: int = 3


class AgentResearchConfig(BaseModel):
    per_leaderboard_candidate_budget: int = 12
    promotion_gate: PromotionGate = PromotionGate.BEAT_STRONG_BASELINE
    lockbox_regression_tolerance: float = 0.1
    min_idea_count: int = 6
    min_qualified_papers: int = 8
    paper_quality_threshold: float = 0.45
    max_scout_rounds: int = 4
    require_llm_insight_review: bool = True
    require_llm_idea_review: bool = True
    min_worth_synthesizing_papers: int = 5
    min_full_text_analyses: int = 4
    evidence_year_window: int = 3
    paper_sources: list[str] = Field(default_factory=lambda: ["bootstrap", "openalex", "arxiv", "acl_anthology"])
    output_dir: str = "outputs/agents"


class LeaderboardTask(BaseModel):
    task_id: str
    dataset: str
    display_name: str
    benchmark: str = "CLUE"
    metric: str = "accuracy"
    train_rows: int | None = None
    validation_rows: int | None = None
    test_rows: int | None = None
    official_url: str = "https://github.com/CLUEbenchmark/CLUE"
    submission_format: str = "jsonl"
    strong_baseline_name: str = ""
    strong_baseline_score: float | None = None
    status: LeaderboardStatus = LeaderboardStatus.ACTIVE
    priority: int = 100
    notes: str = ""


class ResearchTaskConfig(BaseModel):
    task_name: str
    output_root: str = "outputs"
    registry_path: str
    artifacts_dir: str
    reports_dir: str
    method_cards_path: str
    proposals_dir: str
    primary_dataset: str
    sanity_dataset: str
    metric_name: str = "accuracy"
    lockbox_fraction: float = 0.1
    lockbox_seed: int = 3407
    random_seeds: list[int] = Field(default_factory=lambda: [42])
    max_cost_ratio: float = 1.5
    validation_delta_threshold: float = 0.5
    lockbox_delta_threshold: float = 0.3
    significance_alpha: float = 0.05
    device_preference: str = "auto"
    datasets: dict[str, DatasetConfig]
    llm: LLMConfig = Field(default_factory=LLMConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    budgets: dict[BudgetLevel, BudgetProfile]
    baselines: BaselineConfig
    allowed_modules: AllowedModules
    stop_conditions: StopConditions = Field(default_factory=StopConditions)
    agent_research: AgentResearchConfig = Field(default_factory=AgentResearchConfig)
    leaderboards: list[LeaderboardTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dataset_keys(self) -> "ResearchTaskConfig":
        if self.primary_dataset not in self.datasets:
            raise ValueError(f"primary_dataset {self.primary_dataset!r} missing from datasets")
        if self.sanity_dataset not in self.datasets:
            raise ValueError(f"sanity_dataset {self.sanity_dataset!r} missing from datasets")
        return self

    def resolve_path(self, path: str | Path) -> Path:
        return Path(path)

    @property
    def output_root_path(self) -> Path:
        return self.resolve_path(self.output_root)

    def budget_profile(self, budget: str | BudgetLevel) -> BudgetProfile:
        key = budget if isinstance(budget, BudgetLevel) else BudgetLevel(budget)
        return self.budgets[key]


class MethodModuleHints(BaseModel):
    pooling: list[PoolingType] = Field(default_factory=list)
    head: list[HeadType] = Field(default_factory=list)
    loss: list[LossType] = Field(default_factory=list)
    schedule: list[ScheduleType] = Field(default_factory=list)
    augmentation: list[AugmentationType] = Field(default_factory=list)


class MethodCard(BaseModel):
    paper_id: str
    title: str
    source: str
    summary: str
    core_idea: str
    assumptions: list[str] = Field(default_factory=list)
    expected_gain: str = ""
    risks: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    mapped_modules: MethodModuleHints = Field(default_factory=MethodModuleHints)


class PaperEvidence(BaseModel):
    paper_id: str
    title: str
    source_url: str
    source_name: str
    year: int | None = None
    venue: str | None = None
    abstract: str = ""
    code_url: str | None = None
    benchmark: str | None = None
    relevance_score: float = 0.0
    tags: list[str] = Field(default_factory=list)
    error: str | None = None
    query: str | None = None
    quality_score: float = 0.0
    quality_pass: bool = False
    task_relevance_score: float = 0.0
    benchmark_relevance_score: float = 0.0
    method_transferability_score: float = 0.0
    code_availability_score: float = 0.0
    sota_evidence_score: float = 0.0
    implementation_feasibility_score: float = 0.0
    quality_reasons: list[str] = Field(default_factory=list)


class PaperQualityReport(BaseModel):
    task_id: str
    dataset: str
    status: str
    total_candidates: int
    raw_papers: int
    qualified_papers: int
    min_required: int
    quality_threshold: float
    average_quality_score: float = 0.0
    average_qualified_score: float = 0.0
    source_counts: dict[str, int] = Field(default_factory=dict)
    query_counts: dict[str, int] = Field(default_factory=dict)
    top_keywords: list[str] = Field(default_factory=list)
    qualified_paper_ids: list[str] = Field(default_factory=list)
    failure_reasons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class PaperMethodAnalysis(BaseModel):
    paper_id: str
    title: str
    source_url: str
    analysis_depth: str = "abstract_only"
    evidence_chars: int = 0
    algorithm_family: str = ""
    model_or_algorithm: str = ""
    core_mechanism: str = ""
    training_objective: str = ""
    theoretical_basis: list[str] = Field(default_factory=list)
    transferable_mechanisms: list[str] = Field(default_factory=list)
    runner_mapping: MethodModuleHints = Field(default_factory=MethodModuleHints)
    implementation_requirements: list[str] = Field(default_factory=list)
    expected_effect_on_tnews: str = ""
    risks: list[str] = Field(default_factory=list)
    novelty_takeaways: list[str] = Field(default_factory=list)
    quality_assessment: str = ""
    worth_synthesizing: bool = False


class PaperInsightReview(BaseModel):
    paper_id: str
    title: str
    reviewer: str = "rulebased"
    confidence: float = 0.0
    algorithm_understanding_score: float = 0.0
    transferability_score: float = 0.0
    novelty_source_score: float = 0.0
    implementation_clarity_score: float = 0.0
    evidence_strength_score: float = 0.0
    key_algorithm_steps: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    critique: str = ""
    approved_for_synthesis: bool = False


class ResearchReadinessReport(BaseModel):
    task_id: str
    dataset: str
    status: str
    reviewer: str = "rulebased"
    llm_review_required: bool = True
    llm_review_available: bool = False
    paper_count: int = 0
    qualified_evidence_count: int = 0
    worth_synthesizing_count: int = 0
    full_text_analysis_count: int = 0
    approved_insight_count: int = 0
    mechanism_clusters: dict[str, int] = Field(default_factory=dict)
    strongest_mechanisms: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class IdeaQualityReview(BaseModel):
    idea_id: str
    title: str
    reviewer: str = "rulebased"
    evidence_grounding_score: float = 0.0
    novelty_score: float = 0.0
    baseline_difference_score: float = 0.0
    feasibility_score: float = 0.0
    leaderboard_potential_score: float = 0.0
    risk_score: float = 0.0
    critique: str = ""
    required_ablation: list[str] = Field(default_factory=list)
    required_implementation_checks: list[str] = Field(default_factory=list)
    approved_for_experiment: bool = False


class SotaSnapshot(BaseModel):
    task_id: str
    dataset: str
    benchmark: str = "CLUE"
    metric: str = "accuracy"
    official_url: str
    baselines: list[dict[str, Any]] = Field(default_factory=list)
    target_score: float | None = None
    submission_format: str = "jsonl"
    notes: list[str] = Field(default_factory=list)


class IdeaCard(BaseModel):
    idea_id: str
    task_id: str
    title: str
    source_paper_ids: list[str]
    mechanism: str
    difference_from_baseline: str
    expected_gain: str
    implementation_plan: str
    mapped_modules: MethodModuleHints = Field(default_factory=MethodModuleHints)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    risks: list[str] = Field(default_factory=list)
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    leaderboard_potential_score: float = 0.0
    cost_risk: str = "medium"
    accepted: bool = False
    rejection_reason: str | None = None
    implementation_ticket: str | None = None


class ResearchDecision(BaseModel):
    task_id: str
    dataset: str
    status: LeaderboardStatus
    reason: str
    tested_candidates: int = 0
    candidate_budget: int = 12
    best_validation: float | None = None
    best_baseline_validation: float | None = None
    promote_experiment_id: str | None = None
    next_task_id: str | None = None


class ExperimentSpec(BaseModel):
    experiment_id: str
    dataset: str
    baseline_name: str
    model_name: str
    model_id: str
    pooling: PoolingType
    head: HeadType
    loss: LossType
    schedule: ScheduleType
    augmentation: AugmentationType
    budget: BudgetLevel
    seeds: list[int]
    rationale: str
    expected_risk: str = "medium"
    tags: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    def _hyperparameter_signature(self) -> str:
        if not self.hyperparameters:
            return ""
        payload = json.dumps(self.hyperparameters, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"|hp={payload}"

    @property
    def signature(self) -> str:
        base = "|".join(
            [
                self.dataset,
                self.model_id,
                self.pooling.value,
                self.head.value,
                self.loss.value,
                self.schedule.value,
                self.augmentation.value,
            ]
        )
        return base + self._hyperparameter_signature()

    @property
    def method_signature(self) -> str:
        base = "|".join(
            [
                self.model_id,
                self.pooling.value,
                self.head.value,
                self.loss.value,
                self.schedule.value,
                self.augmentation.value,
            ]
        )
        return base + self._hyperparameter_signature()


def _read_path(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_task_config(path: str | Path) -> ResearchTaskConfig:
    return ResearchTaskConfig.model_validate(yaml.safe_load(_read_path(path)))


def load_method_cards(path: str | Path) -> list[MethodCard]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    if file_path.suffix == ".jsonl":
        items = [json.loads(line) for line in _read_path(path).splitlines() if line.strip()]
        return [MethodCard.model_validate(item) for item in items]
    payload = yaml.safe_load(_read_path(path))
    if isinstance(payload, dict) and "papers" in payload:
        payload = payload["papers"]
    return [MethodCard.model_validate(item) for item in payload or []]


def load_experiment_spec(path: str | Path) -> list[ExperimentSpec]:
    payload = yaml.safe_load(_read_path(path))
    if isinstance(payload, dict) and "experiments" in payload:
        payload = payload["experiments"]
    return [ExperimentSpec.model_validate(item) for item in payload or []]


def dump_yaml(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def dump_jsonl(path: str | Path, rows: list[BaseModel | dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in rows:
        payload = row.model_dump(mode="json") if isinstance(row, BaseModel) else row
        lines.append(json.dumps(payload, ensure_ascii=False))
    file_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
