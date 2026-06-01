from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError:  # pragma: no cover - exercised indirectly in lightweight environments
    torch = None
    F = None
    nn = None
    AdamW = None
    DataLoader = None
    Dataset = object
    WeightedRandomSampler = None
    AutoModel = None
    AutoTokenizer = None
    get_linear_schedule_with_warmup = None

from .config import (
    AugmentationType,
    ExperimentSpec,
    HeadType,
    LossType,
    PoolingType,
    ResearchTaskConfig,
    ScheduleType,
)
from .datasets import PreparedDataset, sample_frame
from .utils import ensure_dir, set_seed, slugify, stable_hash, write_frame


@dataclass
class RunOutcome:
    run_id: str
    experiment_id: str
    dataset: str
    model_name: str
    model_id: str
    budget: str
    seed: int
    kind: str
    signature: str
    method_signature: str
    train_seconds: float
    device: str
    train_rows: int
    validation_accuracy: float
    lockbox_accuracy: float | None
    metrics_json: dict[str, Any]
    params_json: dict[str, Any]
    artifact_dir: str

    def to_row(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "dataset": self.dataset,
            "model_name": self.model_name,
            "model_id": self.model_id,
            "budget": self.budget,
            "seed": self.seed,
            "kind": self.kind,
            "status": "completed",
            "signature": self.signature,
            "method_signature": self.method_signature,
            "train_seconds": self.train_seconds,
            "device": self.device,
            "train_rows": self.train_rows,
            "validation_accuracy": self.validation_accuracy,
            "lockbox_accuracy": self.lockbox_accuracy,
            "sanity_accuracy": None,
            "params_json": self.params_json,
            "metrics_json": self.metrics_json,
            "artifact_dir": self.artifact_dir,
        }


def resolve_device(preference: str = "auto") -> tuple[Any, str]:
    if torch is None:
        return "cpu", "cpu"
    if preference == "cpu":
        return torch.device("cpu"), "cpu"
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


class TextFrameDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        text_field: str,
        label_field: str,
        text_template: str | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.text_field = text_field
        self.label_field = label_field
        self.text_template = text_template

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        text = str(row[self.text_field])
        if self.text_template:
            text = self.text_template.format(text=text)
        return {
            "text": text,
            "label": int(row[self.label_field]),
            "sample_id": int(row["sample_id"]),
        }


if torch is not None:
    class TransformerClassifier(nn.Module):
        def __init__(
            self,
            model_id: str,
            num_labels: int,
            pooling: PoolingType,
            head: HeadType,
            dropout: float,
        ) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_id)
            hidden_size = self.encoder.config.hidden_size
            self.pooling = pooling
            self.head_type = head
            self.dropout = nn.Dropout(dropout)
            self.attention_pool = nn.Linear(hidden_size, 1) if pooling == PoolingType.ATTENTION else None
            self.prototypes = None
            self.label_anchors = None
            self.scale = nn.Parameter(torch.tensor(10.0)) if head in {HeadType.PROTOTYPE, HeadType.LABEL_ATTENTION, HeadType.RESIDUAL_LABEL_ATTENTION} else None
            self.label_queries = None
            self.label_bias = None
            self.label_attention_gate = None
            self.aux_classifier = None

            if head in {HeadType.LINEAR, HeadType.LABEL_SEMANTIC, HeadType.RESIDUAL_LABEL_ATTENTION}:
                self.classifier = nn.Linear(hidden_size, num_labels)
                if head == HeadType.RESIDUAL_LABEL_ATTENTION:
                    self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_size))
                    self.label_bias = nn.Parameter(torch.zeros(num_labels))
                    self.label_attention_gate = nn.Parameter(torch.tensor(-4.0))
            elif head == HeadType.MLP:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, num_labels),
                )
            elif head == HeadType.LABEL_ATTENTION:
                self.label_queries = nn.Parameter(torch.randn(num_labels, hidden_size))
                self.label_bias = nn.Parameter(torch.zeros(num_labels))
                self.classifier = None
            else:
                self.prototypes = nn.Parameter(torch.randn(num_labels, hidden_size))
                self.classifier = None

        def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            if self.pooling == PoolingType.CLS:
                return hidden_states[:, 0]
            if self.pooling == PoolingType.MEAN:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            assert self.attention_pool is not None
            scores = self.attention_pool(hidden_states).squeeze(-1)
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
            weights = scores.softmax(dim=1).unsqueeze(-1)
            return (hidden_states * weights).sum(dim=1)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            if self.head_type in {HeadType.LABEL_ATTENTION, HeadType.RESIDUAL_LABEL_ATTENTION}:
                assert self.label_queries is not None and self.label_bias is not None
                hidden_states = self.dropout(outputs.last_hidden_state)
                queries = F.normalize(self.label_queries, dim=-1)
                token_scores = torch.einsum("blh,ch->blc", hidden_states, queries)
                token_scores = token_scores.masked_fill(attention_mask.unsqueeze(-1) == 0, torch.finfo(token_scores.dtype).min)
                token_weights = token_scores.softmax(dim=1)
                class_features = torch.einsum("blc,blh->bch", token_weights, hidden_states)
                attention_logits = (F.normalize(class_features, dim=-1) * queries.unsqueeze(0)).sum(dim=-1)
                attention_logits = attention_logits * (self.scale if self.scale is not None else 10.0) + self.label_bias
                features = self._pool(outputs.last_hidden_state, attention_mask)
                if self.head_type == HeadType.RESIDUAL_LABEL_ATTENTION:
                    assert self.classifier is not None and self.label_attention_gate is not None
                    cls_logits = self.classifier(self.dropout(features))
                    logits = cls_logits + torch.sigmoid(self.label_attention_gate) * attention_logits
                else:
                    logits = attention_logits
                return logits, features
            features = self._pool(outputs.last_hidden_state, attention_mask)
            features = self.dropout(features)
            if self.head_type == HeadType.PROTOTYPE:
                proto = F.normalize(self.prototypes, dim=-1)
                feat = F.normalize(features, dim=-1)
                logits = self.scale * feat @ proto.t()
            else:
                logits = self.classifier(features)
            return logits, features
else:
    class TransformerClassifier:  # pragma: no cover - only used when torch is unavailable
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("torch and transformers are required for transformer experiments")


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    ce = F.cross_entropy(logits, labels, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def class_balanced_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: torch.Tensor,
    beta: float = 0.9999,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    beta_tensor = torch.tensor(beta, dtype=class_counts.dtype, device=class_counts.device)
    effective_num = 1.0 - torch.pow(beta_tensor, class_counts.clamp_min(1.0))
    weights = (1.0 - beta) / effective_num.clamp_min(1e-12)
    weights = weights / weights[labels].mean().clamp_min(1e-12)
    losses = F.cross_entropy(logits, labels, label_smoothing=label_smoothing, reduction="none")
    return (losses * weights[labels]).mean()


def class_balanced_weights(class_counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    beta_tensor = torch.tensor(beta, dtype=class_counts.dtype, device=class_counts.device)
    effective_num = 1.0 - torch.pow(beta_tensor, class_counts.clamp_min(1.0))
    return (1.0 - beta) / effective_num.clamp_min(1e-12)


def balanced_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: torch.Tensor,
    tau: float = 1.0,
) -> torch.Tensor:
    log_counts = torch.log(class_counts.clamp_min(1.0)).to(logits.device)
    return F.cross_entropy(logits + tau * log_counts.unsqueeze(0), labels)


def ldam_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: torch.Tensor,
    max_margin: float = 0.5,
    scale: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    margins = 1.0 / torch.sqrt(torch.sqrt(class_counts.clamp_min(1.0)))
    margins = margins * (max_margin / margins.max().clamp_min(1e-12))
    adjusted = logits.clone()
    adjusted[torch.arange(labels.size(0), device=labels.device), labels] -= margins.to(logits.device)[labels]
    if class_weights is not None:
        class_weights = class_weights / class_weights[labels].mean().clamp_min(1e-12)
    return F.cross_entropy(scale * adjusted, labels, weight=class_weights)


def supervised_contrastive_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    similarity = torch.matmul(features, features.T) / temperature
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask.fill_diagonal_(False)
    logits_mask = ~torch.eye(features.size(0), dtype=torch.bool, device=features.device)
    similarity = similarity.masked_fill(~logits_mask, torch.finfo(similarity.dtype).min)
    exp_similarity = torch.exp(similarity)
    positives = (exp_similarity * mask).sum(dim=1)
    denominators = exp_similarity.sum(dim=1).clamp_min(1e-8)
    valid = positives > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=features.device)
    loss = -torch.log((positives[valid] / denominators[valid]).clamp_min(1e-8))
    return loss.mean()


def label_anchor_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    features = F.normalize(features, dim=-1)
    anchors = F.normalize(anchors, dim=-1)
    logits = features @ anchors.T / temperature
    return F.cross_entropy(logits, labels)


def initialize_class_priors(
    model: TransformerClassifier,
    train_frame: pd.DataFrame,
    label_field: str,
    num_labels: int,
    spec: ExperimentSpec,
    device: torch.device,
) -> None:
    count_based_losses = {
        LossType.PRIOR_ADJUSTED_LABEL_SMOOTHING,
        LossType.TAIL_WEIGHTED_LABEL_SMOOTHING,
        LossType.CLASS_BALANCED,
        LossType.BALANCED_SOFTMAX,
        LossType.LDAM,
        LossType.LDAM_DRW,
    }
    needs_aux_counts = (
        spec.loss == LossType.LABEL_SMOOTHING
        and float(spec.hyperparameters.get("balanced_aux_head_weight", 0.0)) > 0
    )
    needs_rdrop_counts = (
        spec.loss == LossType.RDROP
        and (
            float(spec.hyperparameters.get("rdrop_balanced_softmax_tau", 0.0)) > 0
            or float(spec.hyperparameters.get("rdrop_balanced_softmax_tau_final", 0.0)) > 0
        )
    )
    needs_rdrop_priors = (
        spec.loss == LossType.RDROP
        and (
            float(spec.hyperparameters.get("rdrop_prior_adjustment_tau", 0.0)) > 0
            or float(spec.hyperparameters.get("rdrop_prior_adjustment_tau_final", 0.0)) > 0
        )
    )
    if spec.loss not in count_based_losses and not needs_aux_counts and not needs_rdrop_counts and not needs_rdrop_priors:
        return
    smoothing = float(spec.hyperparameters.get("class_prior_smoothing", 1.0))
    counts = np.bincount(train_frame[label_field].to_numpy(dtype=np.int64), minlength=num_labels).astype(np.float64)
    model.register_buffer("class_counts_tensor", torch.tensor(counts, dtype=torch.float32, device=device))
    model.class_counts = counts.tolist()
    if (
        spec.loss in {LossType.CLASS_BALANCED, LossType.BALANCED_SOFTMAX, LossType.LDAM, LossType.LDAM_DRW}
        or needs_aux_counts
        or (needs_rdrop_counts and not needs_rdrop_priors)
    ):
        return
    if spec.loss == LossType.TAIL_WEIGHTED_LABEL_SMOOTHING:
        quantile = float(spec.hyperparameters.get("tail_weight_quantile", 0.5))
        alpha = float(spec.hyperparameters.get("tail_weight_alpha", 0.5))
        strength = float(spec.hyperparameters.get("tail_weight_strength", 0.5))
        max_weight = float(spec.hyperparameters.get("tail_weight_max", 2.0))
        positive_counts = counts[counts > 0]
        threshold = float(np.quantile(positive_counts, quantile))
        safe_counts = np.maximum(counts, 1.0)
        rarity = np.where(counts <= threshold, np.power(threshold / safe_counts, alpha), 1.0)
        weights = 1.0 + strength * (rarity - 1.0)
        weights = np.clip(weights, 1.0, max_weight)
        label_values = train_frame[label_field].to_numpy(dtype=np.int64)
        weights = weights / np.mean(weights[label_values])
        model.register_buffer("class_loss_weights", torch.tensor(weights, dtype=torch.float32, device=device))
        return
    priors = (counts + smoothing) / (counts.sum() + smoothing * num_labels)
    log_priors_np = np.log(priors)
    mode = str(spec.hyperparameters.get("prior_adjustment_mode", "all"))
    if mode == "tail_only":
        quantile = float(spec.hyperparameters.get("prior_tail_quantile", 0.5))
        threshold = float(np.quantile(counts[counts > 0], quantile))
        log_priors_np = np.where(counts <= threshold, log_priors_np, 0.0)
    elif mode == "auto_tail":
        probabilities = counts / counts.sum()
        normalized_entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12))) / float(np.log(num_labels))
        imbalance_index = 1.0 - normalized_entropy
        min_imbalance = float(spec.hyperparameters.get("prior_min_imbalance_index", 0.1))
        if imbalance_index < min_imbalance:
            log_priors_np = np.zeros_like(log_priors_np)
        else:
            quantile = float(spec.hyperparameters.get("prior_tail_quantile", 0.5))
            threshold = float(np.quantile(counts[counts > 0], quantile))
            log_priors_np = np.where(counts <= threshold, log_priors_np, 0.0)
    elif mode == "head_tail":
        quantile = float(spec.hyperparameters.get("prior_tail_quantile", 0.5))
        head_weight = float(spec.hyperparameters.get("prior_head_weight", 0.05))
        tail_weight = float(spec.hyperparameters.get("prior_tail_weight", 0.15))
        threshold = float(np.quantile(counts[counts > 0], quantile))
        weights = np.where(counts <= threshold, tail_weight, head_weight)
        log_priors_np = log_priors_np * weights
    elif mode == "adaptive_tail":
        quantile = float(spec.hyperparameters.get("prior_tail_quantile", 0.5))
        alpha = float(spec.hyperparameters.get("prior_tail_alpha", 0.5))
        min_weight = float(spec.hyperparameters.get("prior_tail_min_weight", 0.35))
        max_weight = float(spec.hyperparameters.get("prior_tail_max_weight", 1.15))
        positive_counts = counts[counts > 0]
        threshold = float(np.quantile(positive_counts, quantile))
        safe_counts = np.maximum(counts, 1.0)
        tail_mask = counts <= threshold
        rarity = np.zeros_like(counts, dtype=np.float64)
        rarity[tail_mask] = np.power(threshold / safe_counts[tail_mask], alpha)
        tail_rarity = rarity[tail_mask]
        if tail_rarity.size and float(tail_rarity.max()) > 1.0:
            normalized = (tail_rarity - 1.0) / (float(tail_rarity.max()) - 1.0)
        else:
            normalized = np.zeros_like(tail_rarity)
        weights = np.zeros_like(counts, dtype=np.float64)
        weights[tail_mask] = min_weight + (max_weight - min_weight) * normalized
        log_priors_np = log_priors_np * weights
    elif mode == "scaled_tail":
        quantile = float(spec.hyperparameters.get("prior_tail_quantile", 0.5))
        alpha = float(spec.hyperparameters.get("prior_tail_alpha", 1.0))
        min_weight = float(spec.hyperparameters.get("prior_tail_min_weight", 0.25))
        positive_counts = counts[counts > 0]
        threshold = float(np.quantile(positive_counts, quantile))
        min_count = float(positive_counts.min())
        span = max(threshold - min_count, 1.0)
        rarity = np.clip((threshold - counts) / span, 0.0, 1.0)
        weights = np.where(counts <= threshold, min_weight + (1.0 - min_weight) * np.power(rarity, alpha), 0.0)
        log_priors_np = log_priors_np * weights
    elif mode == "conservative_auto_tail":
        probabilities = counts / counts.sum()
        normalized_entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12))) / float(np.log(num_labels))
        imbalance_index = 1.0 - normalized_entropy
        min_imbalance = float(spec.hyperparameters.get("prior_min_imbalance_index", 0.1))
        if imbalance_index < min_imbalance:
            log_priors_np = np.zeros_like(log_priors_np)
        else:
            quantile = float(spec.hyperparameters.get("prior_tail_quantile", 0.5))
            alpha = float(spec.hyperparameters.get("prior_tail_alpha", 0.7))
            min_weight = float(spec.hyperparameters.get("prior_tail_min_weight", 0.15))
            max_weight = float(spec.hyperparameters.get("prior_tail_max_weight", 0.75))
            log_clip = float(spec.hyperparameters.get("prior_log_clip", 5.0))
            positive_counts = counts[counts > 0]
            threshold = float(np.quantile(positive_counts, quantile))
            tail_mask = counts <= threshold
            safe_counts = np.maximum(counts, 1.0)
            threshold_prior = (threshold + smoothing) / (counts.sum() + smoothing * num_labels)
            # Center at the tail threshold so borderline-tail classes receive little correction.
            centered_log_priors = np.minimum(log_priors_np - float(np.log(threshold_prior)), 0.0)
            rarity = np.zeros_like(counts, dtype=np.float64)
            rarity[tail_mask] = np.clip((threshold - safe_counts[tail_mask]) / max(threshold - float(positive_counts.min()), 1.0), 0.0, 1.0)
            weights = np.zeros_like(counts, dtype=np.float64)
            weights[tail_mask] = min_weight + (max_weight - min_weight) * np.power(rarity[tail_mask], alpha)
            log_priors_np = np.clip(centered_log_priors, -log_clip, 0.0) * weights
    elif mode != "all":
        raise ValueError(f"Unsupported prior_adjustment_mode {mode!r}")
    log_priors = torch.tensor(log_priors_np, dtype=torch.float32, device=device)
    model.register_buffer("class_log_priors", log_priors)


def label_descriptions(dataset: PreparedDataset, spec: ExperimentSpec) -> list[str]:
    override = spec.hyperparameters.get("label_descriptions")
    if isinstance(override, list) and len(override) == dataset.num_labels:
        return [str(value) for value in override]
    if dataset.name == "tnews" and dataset.num_labels == 15:
        return [
            "新闻类别：民生故事，社会生活和身边事件",
            "新闻类别：文化，文学历史艺术和传统文化",
            "新闻类别：娱乐，明星影视综艺和娱乐资讯",
            "新闻类别：体育，比赛运动员球队和竞技体育",
            "新闻类别：财经，经济金融消费和商业动态",
            "新闻类别：房产，楼市住房地产和物业",
            "新闻类别：汽车，车型驾驶车企和交通出行",
            "新闻类别：教育，学校考试升学和学习培训",
            "新闻类别：科技，互联网数码智能和科学技术",
            "新闻类别：军事，军队武器战争和国防",
            "新闻类别：旅游，景点酒店路线和旅行体验",
            "新闻类别：国际，海外国家外交和世界事件",
            "新闻类别：股票，股市证券基金和投资交易",
            "新闻类别：农业，农村种植养殖和农产品",
            "新闻类别：游戏，电子游戏玩家赛事和游戏产业",
        ]
    if dataset.name == "csl_cls_ctg" and dataset.num_labels == 13:
        return [
            "论文所属学科：军事学，国防安全、军队建设、武器装备、作战指挥与军事理论",
            "论文所属学科：农学，作物栽培、农业资源、畜牧兽医、林业、水产与食品农业",
            "论文所属学科：医学，临床医学、基础医学、公共卫生、药学、护理与中医药",
            "论文所属学科：历史学，历史文献、考古、世界史、中国史、文化史与史学理论",
            "论文所属学科：哲学，哲学理论、伦理学、逻辑学、美学、宗教学与思想研究",
            "论文所属学科：工学，计算机、机械、电子、材料、土木、能源、自动化与工程技术",
            "论文所属学科：教育学，教育理论、课程教学、学习评价、教师发展与教育管理",
            "论文所属学科：文学，语言文字、文学理论、中国文学、外国文学、新闻传播与写作",
            "论文所属学科：法学，法律制度、司法、政治学、社会学、民族学、马克思主义理论",
            "论文所属学科：理学，数学、物理、化学、生物、地理、天文与基础自然科学",
            "论文所属学科：管理学，工商管理、公共管理、信息管理、工程管理、会计与决策科学",
            "论文所属学科：经济学，宏观经济、金融、贸易、产业经济、财政税收与统计经济",
            "论文所属学科：艺术学，音乐、美术、设计、戏剧影视、舞蹈、艺术理论与艺术教育",
        ]
    return [f"新闻类别：{label}" for label in dataset.label_list]


def tnews_label_concepts() -> list[list[str]]:
    return [
        ["民生", "社会", "生活", "居民", "社区", "事件", "百姓", "救助", "就业"],
        ["文化", "文学", "历史", "艺术", "传统", "文物", "读书", "博物馆"],
        ["娱乐", "明星", "影视", "综艺", "演员", "导演", "电影", "音乐"],
        ["体育", "比赛", "球队", "球员", "联赛", "冠军", "教练", "赛季"],
        ["财经", "经济", "金融", "消费", "商业", "市场", "企业", "银行"],
        ["房产", "楼市", "住房", "楼盘", "物业", "租房", "买房", "房价"],
        ["汽车", "车型", "驾驶", "车企", "新能源", "销量", "发动机", "交通"],
        ["教育", "学校", "考试", "升学", "学生", "老师", "课程", "培训"],
        ["科技", "互联网", "数码", "智能", "手机", "芯片", "软件", "人工智能"],
        ["军事", "军队", "武器", "战争", "国防", "导弹", "演习", "航母"],
        ["旅游", "景点", "酒店", "路线", "旅行", "游客", "景区", "假期"],
        ["国际", "海外", "国家", "外交", "世界", "美国", "欧洲", "联合国"],
        ["股票", "股市", "证券", "基金", "投资", "A股", "涨停", "财报"],
        ["农业", "农村", "种植", "养殖", "农产品", "粮食", "农民", "乡村"],
        ["游戏", "电竞", "玩家", "手游", "网游", "赛事", "主机", "游戏"],
    ]


def label_concepts(dataset: PreparedDataset, spec: ExperimentSpec) -> list[list[str]]:
    override = spec.hyperparameters.get("label_concepts")
    if isinstance(override, list) and len(override) == dataset.num_labels:
        return [[str(item) for item in group] for group in override]
    if dataset.name == "tnews" and dataset.num_labels == 15:
        return tnews_label_concepts()
    if dataset.name == "csl_cls_ctg" and dataset.num_labels == 13:
        return [
            ["军事", "国防", "军队", "武器", "作战", "安全"],
            ["农业", "农学", "作物", "畜牧", "林业", "水产", "食品"],
            ["医学", "临床", "药学", "卫生", "护理", "中医"],
            ["历史", "考古", "文献", "中国史", "世界史", "史学"],
            ["哲学", "伦理", "逻辑", "美学", "宗教", "思想"],
            ["工学", "计算机", "机械", "电子", "材料", "工程", "自动化"],
            ["教育", "教学", "课程", "学习", "教师", "评价"],
            ["文学", "语言", "文字", "文学", "新闻", "传播"],
            ["法学", "法律", "司法", "政治", "社会", "民族"],
            ["理学", "数学", "物理", "化学", "生物", "地理", "天文"],
            ["管理", "工商", "公共管理", "信息管理", "会计", "决策"],
            ["经济", "金融", "贸易", "产业", "财政", "统计"],
            ["艺术", "音乐", "美术", "设计", "戏剧", "影视", "舞蹈"],
        ]
    return [[str(label)] for label in dataset.label_list]


def build_label_aware_mask_context(tokenizer: Any, dataset: PreparedDataset, spec: ExperimentSpec) -> dict[str, Any]:
    if not spec.hyperparameters.get("label_aware_token_masking", False):
        return {}
    special_ids = {
        token_id
        for token_id in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.unk_token_id]
        if token_id is not None
    }
    preserve_by_label: list[torch.Tensor] = []
    for concepts in label_concepts(dataset, spec):
        token_ids: set[int] = set()
        for concept in concepts:
            token_ids.update(int(token_id) for token_id in tokenizer.encode(concept, add_special_tokens=False))
        token_ids.difference_update(special_ids)
        preserve_by_label.append(torch.tensor(sorted(token_ids), dtype=torch.long))
    return {"preserve_token_ids_by_label": preserve_by_label}


def initialize_label_semantic_head(
    model: TransformerClassifier,
    tokenizer: Any,
    dataset: PreparedDataset,
    spec: ExperimentSpec,
    device: torch.device,
) -> None:
    if spec.head not in {HeadType.LABEL_SEMANTIC, HeadType.LABEL_ATTENTION, HeadType.RESIDUAL_LABEL_ATTENTION}:
        return
    descriptions = label_descriptions(dataset, spec)
    encoded = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model.encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        features = model._pool(outputs.last_hidden_state, encoded["attention_mask"])
        features = F.normalize(features, dim=-1)
        if model.classifier is not None:
            if spec.head == HeadType.LABEL_SEMANTIC:
                model.classifier.weight.copy_(features)
                model.classifier.bias.zero_()
        elif model.label_queries is not None:
            model.label_queries.copy_(features)
            if model.label_bias is not None:
                model.label_bias.zero_()
        if spec.head == HeadType.RESIDUAL_LABEL_ATTENTION and model.label_queries is not None:
            model.label_queries.copy_(features)
            if model.label_bias is not None:
                model.label_bias.zero_()
            if model.label_attention_gate is not None and "label_attention_gate_init" in spec.hyperparameters:
                model.label_attention_gate.data.fill_(float(spec.hyperparameters["label_attention_gate_init"]))
    if was_training:
        model.train()


def initialize_fixed_label_anchors(
    model: TransformerClassifier,
    tokenizer: Any,
    dataset: PreparedDataset,
    spec: ExperimentSpec,
    device: torch.device,
) -> None:
    if not spec.hyperparameters.get("fixed_label_anchors", False):
        return
    descriptions = label_descriptions(dataset, spec)
    encoded = tokenizer(descriptions, padding=True, truncation=True, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        outputs = model.encoder(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        anchors = model._pool(outputs.last_hidden_state, encoded["attention_mask"])
        anchors = F.normalize(anchors, dim=-1)
        model.register_buffer("fixed_label_anchors", anchors)
        model.label_anchors = model.fixed_label_anchors
    if was_training:
        model.train()


def initialize_auxiliary_balanced_head(
    model: TransformerClassifier,
    dataset: PreparedDataset,
    spec: ExperimentSpec,
    device: torch.device,
) -> None:
    if float(spec.hyperparameters.get("balanced_aux_head_weight", 0.0)) <= 0:
        return
    hidden_size = model.encoder.config.hidden_size
    model.aux_classifier = nn.Linear(hidden_size, dataset.num_labels).to(device)


def initialize_lexical_prior(
    model: TransformerClassifier,
    train_frame: pd.DataFrame,
    text_field: str,
    num_labels: int,
    spec: ExperimentSpec,
    device: torch.device,
) -> None:
    if float(spec.hyperparameters.get("lexical_prior_weight", 0.0)) <= 0:
        return
    raw_terms = spec.hyperparameters.get("lexical_prior_terms")
    if not isinstance(raw_terms, list) or len(raw_terms) != num_labels:
        raise ValueError("lexical_prior_weight requires lexical_prior_terms with one term list per label")
    terms_by_label = [[str(term) for term in terms if str(term)] for terms in raw_terms]
    max_sample_id = int(train_frame["sample_id"].max())
    logits = torch.zeros((max_sample_id + 1, num_labels), dtype=torch.float32, device=device)
    mask = torch.zeros(max_sample_id + 1, dtype=torch.bool, device=device)
    hit_strength = float(spec.hyperparameters.get("lexical_prior_hit_strength", 1.0))
    min_margin = float(spec.hyperparameters.get("lexical_prior_min_margin", 1.0))
    for row in train_frame.itertuples(index=False):
        sample_id = int(getattr(row, "sample_id"))
        text = str(getattr(row, text_field))
        scores = torch.zeros(num_labels, dtype=torch.float32, device=device)
        for label_idx, terms in enumerate(terms_by_label):
            hit_count = 0
            for term in terms:
                if term and term in text:
                    hit_count += 1
            if hit_count:
                scores[label_idx] = float(hit_count) * hit_strength
        top_values = scores.topk(k=min(2, num_labels)).values
        margin = top_values[0] - (top_values[1] if top_values.numel() > 1 else 0.0)
        if float(top_values[0].item()) > 0 and float(margin.item()) >= min_margin:
            logits[sample_id] = scores
            mask[sample_id] = True
    model.register_buffer("lexical_prior_logits", logits)
    model.register_buffer("lexical_prior_mask", mask)


def initialize_tfidf_teacher_prior(
    model: TransformerClassifier,
    train_frame: pd.DataFrame,
    text_field: str,
    label_field: str,
    num_labels: int,
    spec: ExperimentSpec,
    device: torch.device,
    seed: int,
) -> None:
    if float(spec.hyperparameters.get("tfidf_teacher_weight", 0.0)) <= 0:
        return
    texts = train_frame[text_field].astype(str).to_numpy()
    labels = train_frame[label_field].to_numpy(dtype=np.int64)
    sample_ids = train_frame["sample_id"].to_numpy(dtype=np.int64)
    max_sample_id = int(sample_ids.max())
    probabilities = np.zeros((len(train_frame), num_labels), dtype=np.float32)
    folds = int(spec.hyperparameters.get("tfidf_teacher_folds", 5))
    min_class_count = int(np.bincount(labels, minlength=num_labels).min())
    folds = max(2, min(folds, min_class_count))
    vectorizer_kwargs = {
        "analyzer": "char",
        "ngram_range": tuple(spec.hyperparameters.get("tfidf_teacher_ngram_range", [1, 3])),
        "min_df": int(spec.hyperparameters.get("tfidf_teacher_min_df", 2)),
        "max_features": int(spec.hyperparameters.get("tfidf_teacher_max_features", 80000)),
    }
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, holdout_idx in skf.split(texts, labels):
        vectorizer = TfidfVectorizer(**vectorizer_kwargs)
        train_x = vectorizer.fit_transform(texts[train_idx])
        holdout_x = vectorizer.transform(texts[holdout_idx])
        classifier = LogisticRegression(
            max_iter=int(spec.hyperparameters.get("tfidf_teacher_max_iter", 500)),
            class_weight=str(spec.hyperparameters.get("tfidf_teacher_class_weight", "balanced")),
            C=float(spec.hyperparameters.get("tfidf_teacher_c", 2.0)),
            random_state=seed,
        )
        classifier.fit(train_x, labels[train_idx])
        fold_probs = classifier.predict_proba(holdout_x)
        aligned = np.full((len(holdout_idx), num_labels), 1e-8, dtype=np.float32)
        for class_col, class_id in enumerate(classifier.classes_):
            aligned[:, int(class_id)] = fold_probs[:, class_col]
        probabilities[holdout_idx] = aligned
    prior_probs = torch.full((max_sample_id + 1, num_labels), 1.0 / num_labels, dtype=torch.float32, device=device)
    prior_mask = torch.zeros(max_sample_id + 1, dtype=torch.bool, device=device)
    confidence_threshold = float(spec.hyperparameters.get("tfidf_teacher_min_confidence", 0.0))
    for row_idx, sample_id in enumerate(sample_ids):
        probs = probabilities[row_idx]
        if float(probs.max()) >= confidence_threshold:
            prior_probs[int(sample_id)] = torch.tensor(probs, dtype=torch.float32, device=device)
            prior_mask[int(sample_id)] = True
    model.register_buffer("tfidf_teacher_probs", prior_probs)
    model.register_buffer("tfidf_teacher_mask", prior_mask)


def compute_loss(
    model: TransformerClassifier,
    batch: dict[str, torch.Tensor],
    spec: ExperimentSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, features = model(batch["input_ids"], batch["attention_mask"])
    labels = batch["labels"]

    if spec.loss == LossType.CE:
        loss = F.cross_entropy(logits, labels)
    elif spec.loss == LossType.FOCAL:
        gamma = float(spec.hyperparameters.get("focal_gamma", 2.0))
        loss = focal_loss(logits, labels, gamma=gamma)
    elif spec.loss == LossType.LABEL_SMOOTHING:
        smoothing = float(spec.hyperparameters.get("label_smoothing", 0.1))
        loss = F.cross_entropy(logits, labels, label_smoothing=smoothing)
        aux_weight = float(spec.hyperparameters.get("balanced_aux_head_weight", 0.0))
        if aux_weight > 0:
            class_counts = getattr(model, "class_counts_tensor", None)
            aux_classifier = getattr(model, "aux_classifier", None)
            if class_counts is None or aux_classifier is None:
                raise ValueError("balanced_aux_head_weight requires initialized class counts and aux head")
            aux_tau = float(spec.hyperparameters.get("balanced_aux_head_tau", 0.25))
            aux_smoothing = float(spec.hyperparameters.get("balanced_aux_head_label_smoothing", 0.0))
            aux_logits = aux_classifier(features)
            log_counts = torch.log(class_counts.clamp_min(1.0)).to(logits.device)
            aux_loss = F.cross_entropy(aux_logits + aux_tau * log_counts.unsqueeze(0), labels, label_smoothing=aux_smoothing)
            loss = loss + aux_weight * aux_loss
    elif spec.loss == LossType.CLASS_BALANCED:
        class_counts = getattr(model, "class_counts_tensor", None)
        if class_counts is None:
            raise ValueError("class_balanced requires initialized class counts")
        beta = float(spec.hyperparameters.get("class_balanced_beta", 0.9999))
        smoothing = float(spec.hyperparameters.get("label_smoothing", 0.0))
        loss = class_balanced_loss(logits, labels, class_counts, beta=beta, label_smoothing=smoothing)
    elif spec.loss == LossType.BALANCED_SOFTMAX:
        class_counts = getattr(model, "class_counts_tensor", None)
        if class_counts is None:
            raise ValueError("balanced_softmax requires initialized class counts")
        tau = float(spec.hyperparameters.get("balanced_softmax_tau", 1.0))
        loss = balanced_softmax_loss(logits, labels, class_counts, tau=tau)
    elif spec.loss == LossType.LDAM:
        class_counts = getattr(model, "class_counts_tensor", None)
        if class_counts is None:
            raise ValueError("ldam requires initialized class counts")
        max_margin = float(spec.hyperparameters.get("ldam_max_margin", 0.5))
        scale = float(spec.hyperparameters.get("ldam_scale", 1.0))
        loss = ldam_loss(logits, labels, class_counts, max_margin=max_margin, scale=scale)
    elif spec.loss == LossType.LDAM_DRW:
        class_counts = getattr(model, "class_counts_tensor", None)
        if class_counts is None:
            raise ValueError("ldam_drw requires initialized class counts")
        max_margin = float(spec.hyperparameters.get("ldam_max_margin", 0.5))
        scale = float(spec.hyperparameters.get("ldam_scale", 1.0))
        beta = float(spec.hyperparameters.get("ldam_drw_beta", 0.9999))
        start_epoch = int(spec.hyperparameters.get("ldam_drw_start_epoch", 1))
        current_epoch = int(getattr(model, "current_epoch", 0))
        weights = class_balanced_weights(class_counts, beta=beta) if current_epoch >= start_epoch else None
        loss = ldam_loss(logits, labels, class_counts, max_margin=max_margin, scale=scale, class_weights=weights)
    elif spec.loss == LossType.PRIOR_ADJUSTED_LABEL_SMOOTHING:
        smoothing = float(spec.hyperparameters.get("label_smoothing", 0.05))
        tau = float(getattr(model, "current_prior_adjustment_tau", spec.hyperparameters.get("prior_adjustment_tau", 0.25)))
        class_log_priors = getattr(model, "class_log_priors", None)
        if class_log_priors is None:
            raise ValueError("prior_adjusted_label_smoothing requires initialized class priors")
        adjusted_logits = logits + tau * class_log_priors.unsqueeze(0)
        sample_gate = str(spec.hyperparameters.get("prior_adjustment_sample_gate", "none"))
        if sample_gate == "none":
            loss = F.cross_entropy(adjusted_logits, labels, label_smoothing=smoothing)
        elif sample_gate in {"tail_label", "tail_or_uncertain"}:
            base_losses = F.cross_entropy(logits, labels, label_smoothing=smoothing, reduction="none")
            adjusted_losses = F.cross_entropy(adjusted_logits, labels, label_smoothing=smoothing, reduction="none")
            tail_labels = class_log_priors.ne(0.0)[labels]
            if sample_gate == "tail_label":
                gate = tail_labels
            else:
                probabilities = logits.detach().softmax(dim=-1)
                top_values = probabilities.topk(k=min(2, probabilities.size(-1)), dim=-1).values
                if top_values.size(-1) == 1:
                    margins = torch.zeros_like(top_values[:, 0])
                else:
                    margins = top_values[:, 0] - top_values[:, 1]
                confidence_threshold = float(spec.hyperparameters.get("prior_guard_confidence_threshold", 0.85))
                margin_threshold = float(spec.hyperparameters.get("prior_guard_margin_threshold", 0.30))
                uncertain = (top_values[:, 0] < confidence_threshold) & (margins < margin_threshold)
                gate = tail_labels | uncertain
            loss = torch.where(gate, adjusted_losses, base_losses).mean()
        else:
            raise ValueError(f"Unsupported prior_adjustment_sample_gate {sample_gate!r}")
        aux_weight = float(spec.hyperparameters.get("balanced_softmax_aux_weight", 0.0))
        if aux_weight > 0:
            class_counts = getattr(model, "class_counts_tensor", None)
            if class_counts is None:
                raise ValueError("balanced_softmax_aux_weight requires initialized class counts")
            aux_tau = float(spec.hyperparameters.get("balanced_softmax_aux_tau", 0.25))
            aux_smoothing = float(spec.hyperparameters.get("balanced_softmax_aux_label_smoothing", 0.0))
            log_counts = torch.log(class_counts.clamp_min(1.0)).to(logits.device)
            aux_loss = F.cross_entropy(logits + aux_tau * log_counts.unsqueeze(0), labels, label_smoothing=aux_smoothing)
            loss = (1.0 - aux_weight) * loss + aux_weight * aux_loss
    elif spec.loss == LossType.TAIL_WEIGHTED_LABEL_SMOOTHING:
        smoothing = float(spec.hyperparameters.get("label_smoothing", 0.05))
        class_loss_weights = getattr(model, "class_loss_weights", None)
        if class_loss_weights is None:
            raise ValueError("tail_weighted_label_smoothing requires initialized class weights")
        losses = F.cross_entropy(logits, labels, label_smoothing=smoothing, reduction="none")
        loss = (losses * class_loss_weights[labels]).mean()
    elif spec.loss == LossType.CE_SUPCON:
        ce = F.cross_entropy(logits, labels)
        scl = supervised_contrastive_loss(features, labels)
        supcon_weight = float(spec.hyperparameters.get("supcon_weight", 0.1))
        loss = ce + supcon_weight * scl
    elif spec.loss == LossType.LABEL_ANCHOR_CONTRASTIVE:
        smoothing = float(spec.hyperparameters.get("label_smoothing", 0.05))
        ce = F.cross_entropy(logits, labels, label_smoothing=smoothing)
        anchor_weight = float(spec.hyperparameters.get("label_anchor_weight", 0.2))
        temperature = float(spec.hyperparameters.get("label_anchor_temperature", 0.07))
        if spec.hyperparameters.get("fixed_label_anchors", False) and model.label_anchors is not None:
            anchors = model.label_anchors
        elif model.classifier is not None and hasattr(model.classifier, "weight"):
            anchors = model.classifier.weight
        elif model.prototypes is not None:
            anchors = model.prototypes
        else:
            raise ValueError("label_anchor_contrastive requires a linear/label_semantic head or prototype head")
        anchor_loss = label_anchor_contrastive_loss(features, labels, anchors, temperature=temperature)
        pair_weight = float(spec.hyperparameters.get("supcon_weight", 0.0))
        pair_loss = supervised_contrastive_loss(features, labels) if pair_weight > 0 else torch.tensor(0.0, device=features.device)
        loss = ce + anchor_weight * anchor_loss + pair_weight * pair_loss
    elif spec.loss == LossType.RDROP:
        logits_2, _ = model(batch["input_ids"], batch["attention_mask"])
        smoothing = float(spec.hyperparameters.get("label_smoothing", 0.0))
        prior_tau = float(
            getattr(
                model,
                "current_rdrop_prior_adjustment_tau",
                spec.hyperparameters.get("rdrop_prior_adjustment_tau", 0.0),
            )
        )
        balanced_tau = float(
            getattr(
                model,
                "current_rdrop_balanced_softmax_tau",
                spec.hyperparameters.get("rdrop_balanced_softmax_tau", 0.0),
            )
        )
        ce_logits = logits
        ce_logits_2 = logits_2
        if prior_tau > 0:
            class_log_priors = getattr(model, "class_log_priors", None)
            if class_log_priors is None:
                raise ValueError("rdrop_prior_adjustment_tau requires initialized class priors")
            prior_adjustment = prior_tau * class_log_priors.unsqueeze(0)
            ce_logits = ce_logits + prior_adjustment
            ce_logits_2 = ce_logits_2 + prior_adjustment
        if balanced_tau > 0:
            class_counts = getattr(model, "class_counts_tensor", None)
            if class_counts is None:
                raise ValueError("rdrop_balanced_softmax_tau requires initialized class counts")
            log_counts = torch.log(class_counts.clamp_min(1.0)).to(logits.device)
            adjustment_mode = str(spec.hyperparameters.get("rdrop_balanced_softmax_mode", "all"))
            if adjustment_mode == "all":
                adjustment = log_counts
            elif adjustment_mode == "frequency_band":
                positive_counts = class_counts[class_counts > 0]
                lower_quantile = float(spec.hyperparameters.get("rdrop_balanced_softmax_band_lower_quantile", 0.4))
                upper_quantile = float(spec.hyperparameters.get("rdrop_balanced_softmax_band_upper_quantile", 0.9))
                lower = torch.quantile(positive_counts, lower_quantile)
                upper = torch.quantile(positive_counts, upper_quantile)
                band = (class_counts >= lower) & (class_counts <= upper)
                adjustment = torch.where(band, log_counts, torch.zeros_like(log_counts))
            else:
                raise ValueError(f"Unsupported rdrop_balanced_softmax_mode {adjustment_mode!r}")
            balanced_adjustment = balanced_tau * adjustment.unsqueeze(0)
            ce_logits = ce_logits + balanced_adjustment
            ce_logits_2 = ce_logits_2 + balanced_adjustment
        ce = 0.5 * (
            F.cross_entropy(ce_logits, labels, label_smoothing=smoothing)
            + F.cross_entropy(ce_logits_2, labels, label_smoothing=smoothing)
        )
        p = F.log_softmax(logits, dim=-1)
        q = F.log_softmax(logits_2, dim=-1)
        kl = 0.5 * (
            F.kl_div(p, q.softmax(dim=-1), reduction="batchmean")
            + F.kl_div(q, p.softmax(dim=-1), reduction="batchmean")
        )
        rdrop_alpha = float(spec.hyperparameters.get("rdrop_alpha", 0.5))
        loss = ce + rdrop_alpha * kl
    else:
        raise ValueError(f"Unsupported loss {spec.loss}")
    semantic_anchor_weight = float(spec.hyperparameters.get("semantic_anchor_weight", 0.0))
    if semantic_anchor_weight > 0 and spec.loss != LossType.LABEL_ANCHOR_CONTRASTIVE:
        temperature = float(spec.hyperparameters.get("label_anchor_temperature", 0.07))
        if spec.hyperparameters.get("fixed_label_anchors", False) and model.label_anchors is not None:
            anchors = model.label_anchors
        elif model.classifier is not None and hasattr(model.classifier, "weight"):
            anchors = model.classifier.weight
        elif model.prototypes is not None:
            anchors = model.prototypes
        else:
            raise ValueError("semantic_anchor_weight requires a linear/prototype head or fixed_label_anchors")
        loss = loss + semantic_anchor_weight * label_anchor_contrastive_loss(features, labels, anchors, temperature=temperature)
    lexical_prior_weight = float(spec.hyperparameters.get("lexical_prior_weight", 0.0))
    if lexical_prior_weight > 0:
        prior_logits = getattr(model, "lexical_prior_logits", None)
        prior_mask = getattr(model, "lexical_prior_mask", None)
        if prior_logits is None or prior_mask is None:
            raise ValueError("lexical_prior_weight requires initialized lexical priors")
        sample_ids = batch["sample_ids"].to(prior_logits.device)
        valid = prior_mask[sample_ids]
        if valid.any():
            temperature = float(spec.hyperparameters.get("lexical_prior_temperature", 0.7))
            teacher = F.softmax(prior_logits[sample_ids[valid]] / temperature, dim=-1)
            student_log_probs = F.log_softmax(logits[valid], dim=-1)
            lexical_loss = F.kl_div(student_log_probs, teacher, reduction="batchmean")
            loss = loss + lexical_prior_weight * lexical_loss
    tfidf_teacher_weight = float(spec.hyperparameters.get("tfidf_teacher_weight", 0.0))
    if tfidf_teacher_weight > 0:
        teacher_probs = getattr(model, "tfidf_teacher_probs", None)
        teacher_mask = getattr(model, "tfidf_teacher_mask", None)
        if teacher_probs is None or teacher_mask is None:
            raise ValueError("tfidf_teacher_weight requires initialized tfidf teacher priors")
        sample_ids = batch["sample_ids"].to(teacher_probs.device)
        valid = teacher_mask[sample_ids]
        if valid.any():
            temperature = float(spec.hyperparameters.get("tfidf_teacher_temperature", 1.0))
            teacher = teacher_probs[sample_ids[valid]].clamp_min(1e-8)
            if temperature != 1.0:
                teacher = F.softmax(torch.log(teacher) / temperature, dim=-1)
            student_log_probs = F.log_softmax(logits[valid], dim=-1)
            teacher_loss = F.kl_div(student_log_probs, teacher, reduction="batchmean")
            loss = loss + tfidf_teacher_weight * teacher_loss
    return loss, logits


def apply_augmentation(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    spec: ExperimentSpec,
    tokenizer: Any,
    context: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spec.augmentation == AugmentationType.NONE:
        return input_ids, attention_mask
    augmented_ids = input_ids.clone()
    if spec.augmentation == AugmentationType.TOKEN_MASK and tokenizer.mask_token_id is not None:
        mask_prob = float(spec.hyperparameters.get("token_mask_prob", 0.1))
        mask = torch.rand_like(augmented_ids.float()) < mask_prob
        special = (augmented_ids == tokenizer.cls_token_id) | (augmented_ids == tokenizer.sep_token_id) | (augmented_ids == tokenizer.pad_token_id)
        preserve_by_label = (context or {}).get("preserve_token_ids_by_label")
        if preserve_by_label:
            preserve = torch.zeros_like(mask, dtype=torch.bool)
            for row_idx, label in enumerate(labels.tolist()):
                if 0 <= label < len(preserve_by_label) and preserve_by_label[label].numel() > 0:
                    preserve_ids = preserve_by_label[label].to(augmented_ids.device)
                    preserve[row_idx] = torch.isin(augmented_ids[row_idx], preserve_ids)
            mask = mask & ~preserve
        mask = mask & ~special
        augmented_ids[mask] = tokenizer.mask_token_id
        return augmented_ids, attention_mask
    if spec.augmentation == AugmentationType.SPAN_CUTOFF:
        for row_idx in range(augmented_ids.size(0)):
            valid_len = int(attention_mask[row_idx].sum().item())
            if valid_len <= 4:
                continue
            span = max(1, valid_len // 10)
            start = int(torch.randint(1, max(2, valid_len - span), (1,)).item())
            end = min(valid_len - 1, start + span)
            augmented_ids[row_idx, start:end] = tokenizer.mask_token_id or tokenizer.unk_token_id
        return augmented_ids, attention_mask
    return augmented_ids, attention_mask


def freeze_encoder(model: TransformerClassifier, frozen: bool) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad = not frozen


def reset_module_parameters(module: nn.Module | None) -> None:
    if module is None:
        return
    for child in module.modules():
        reset = getattr(child, "reset_parameters", None)
        if callable(reset):
            reset()


def prediction_head_parameters(model: TransformerClassifier) -> list[torch.nn.Parameter]:
    parameters: list[torch.nn.Parameter] = []
    for module in [model.classifier, model.attention_pool]:
        if module is not None:
            parameters.extend(parameter for parameter in module.parameters() if parameter.requires_grad)
    for parameter in [model.prototypes, model.scale, model.label_queries, model.label_bias, model.label_attention_gate]:
        if parameter is not None and parameter.requires_grad:
            parameters.append(parameter)
    return parameters


def build_crt_loader(
    train_frame: pd.DataFrame,
    dataset: PreparedDataset,
    task: ResearchTaskConfig,
    tokenizer: Any,
    spec: ExperimentSpec,
    text_template: str | None,
) -> DataLoader:
    labels = train_frame[dataset.label_field].to_numpy(dtype=np.int64)
    counts = np.bincount(labels, minlength=dataset.num_labels).astype(np.float64)
    sample_weights = 1.0 / np.maximum(counts[labels], 1.0)
    generator = torch.Generator()
    generator.manual_seed(int(spec.hyperparameters.get("crt_sampler_seed", 3407)))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        TextFrameDataset(train_frame, dataset.text_field, dataset.label_field, text_template=text_template),
        batch_size=int(spec.hyperparameters.get("crt_batch_size", task.training.train_batch_size)),
        sampler=sampler,
        shuffle=False,
        num_workers=task.training.num_workers,
        collate_fn=lambda rows: collate_batch(tokenizer, rows, dataset.max_length),
    )


def fgm_attack(model: TransformerClassifier, epsilon: float, backup: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if "word_embeddings" not in name:
            continue
        norm = torch.norm(parameter.grad)
        if torch.isfinite(norm) and norm > 0:
            backup[name] = parameter.data.clone()
            parameter.data.add_(epsilon * parameter.grad / norm)


def fgm_restore(model: TransformerClassifier, backup: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        if name in backup:
            parameter.data.copy_(backup[name])
    backup.clear()


def initialize_ema(model: TransformerClassifier) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def update_ema(model: TransformerClassifier, shadow: dict[str, torch.Tensor], decay: float) -> None:
    for name, parameter in model.named_parameters():
        if name in shadow:
            shadow[name].mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)


def apply_ema(model: TransformerClassifier, shadow: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    backup: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name in shadow:
            backup[name] = parameter.detach().clone()
            parameter.data.copy_(shadow[name])
    return backup


def restore_ema(model: TransformerClassifier, backup: dict[str, torch.Tensor]) -> None:
    for name, parameter in model.named_parameters():
        if name in backup:
            parameter.data.copy_(backup[name])


def build_optimizer(model: TransformerClassifier, task: ResearchTaskConfig, spec: ExperimentSpec) -> AdamW:
    lr = float(spec.hyperparameters.get("learning_rate", task.training.learning_rate))
    weight_decay = float(spec.hyperparameters.get("weight_decay", task.training.weight_decay))
    if spec.schedule != ScheduleType.LAYERWISE_LR_DECAY:
        return AdamW((param for param in model.parameters() if param.requires_grad), lr=lr, weight_decay=weight_decay)

    groups: list[dict[str, Any]] = []
    encoder = model.encoder
    layer_stack = None
    for attr in ("encoder", "bert", "roberta"):
        candidate = getattr(encoder, attr, None)
        if candidate is not None and hasattr(candidate, "layer"):
            layer_stack = candidate.layer
            break
    if layer_stack is None and hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        layer_stack = encoder.encoder.layer

    base_lr = lr
    decay = float(spec.hyperparameters.get("lrd_decay", 0.9))
    if hasattr(encoder, "embeddings"):
        groups.append({"params": encoder.embeddings.parameters(), "lr": base_lr * (decay ** 12), "weight_decay": weight_decay})
    if layer_stack is not None:
        for depth, layer in enumerate(layer_stack):
            groups.append({"params": layer.parameters(), "lr": base_lr * (decay ** (len(layer_stack) - depth - 1)), "weight_decay": weight_decay})
    if model.classifier is not None:
        groups.append({"params": model.classifier.parameters(), "lr": base_lr, "weight_decay": weight_decay})
    if model.attention_pool is not None:
        groups.append({"params": model.attention_pool.parameters(), "lr": base_lr, "weight_decay": weight_decay})
    if model.prototypes is not None:
        groups.append({"params": [model.prototypes], "lr": base_lr, "weight_decay": 0.0})
    if model.scale is not None:
        groups.append({"params": [model.scale], "lr": base_lr, "weight_decay": 0.0})
    if model.label_queries is not None:
        groups.append({"params": [model.label_queries], "lr": base_lr, "weight_decay": 0.0})
    if model.label_bias is not None:
        groups.append({"params": [model.label_bias], "lr": base_lr, "weight_decay": 0.0})
    if model.label_attention_gate is not None:
        groups.append({"params": [model.label_attention_gate], "lr": base_lr, "weight_decay": 0.0})
    return AdamW(groups)


def accuracy_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean() * 100.0)


def macro_f1_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(f1_score(labels, predictions, average="macro", zero_division=0) * 100.0)


def balanced_accuracy_metric(predictions: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float(balanced_accuracy_score(labels, predictions) * 100.0)


def write_eval_artifacts(
    artifact_dir: Path,
    split_name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    frame: pd.DataFrame,
    label_field: str,
) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eval_frame = frame.copy()
    eval_frame["prediction"] = predictions
    eval_frame["correct"] = eval_frame[label_field].to_numpy() == predictions
    write_frame(eval_frame, artifact_dir / f"{split_name}_predictions")
    errors = eval_frame.loc[~eval_frame["correct"]]
    write_frame(errors, artifact_dir / f"{split_name}_errors")

    cm = confusion_matrix(labels, predictions)
    (artifact_dir / f"{split_name}_confusion_matrix.json").write_text(json.dumps(cm.tolist()), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"{split_name} confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(artifact_dir / f"{split_name}_confusion_matrix.png")
    plt.close(fig)
    return {
        "accuracy": accuracy_score(predictions, labels),
        "macro_f1": macro_f1_score(predictions, labels),
        "balanced_accuracy": balanced_accuracy_metric(predictions, labels),
        "errors": int((labels != predictions).sum()),
    }


def run_traditional_baseline(
    dataset: PreparedDataset,
    task: ResearchTaskConfig,
    baseline_name: str,
    seed: int,
    artifact_root: Path,
) -> RunOutcome:
    set_seed(seed)
    train_df = dataset.train_search_df
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, max_features=50000)
    x_train = vectorizer.fit_transform(train_df[dataset.text_field])
    y_train = train_df[dataset.label_field].to_numpy()
    x_val = vectorizer.transform(dataset.validation_df[dataset.text_field])
    y_val = dataset.validation_df[dataset.label_field].to_numpy()

    start = perf_counter()
    if baseline_name == "tfidf_logreg":
        classifier = LogisticRegression(max_iter=500, n_jobs=None, random_state=seed)
        classifier.fit(x_train, y_train)
        val_predictions = classifier.predict(x_val)
        lockbox_predictions = classifier.predict(vectorizer.transform(dataset.lockbox_df[dataset.text_field])) if not dataset.lockbox_df.empty else np.array([])
    elif baseline_name == "tfidf_linear_svm":
        classifier = LinearSVC(random_state=seed)
        classifier.fit(x_train, y_train)
        val_predictions = classifier.predict(x_val)
        lockbox_predictions = classifier.predict(vectorizer.transform(dataset.lockbox_df[dataset.text_field])) if not dataset.lockbox_df.empty else np.array([])
    else:
        raise ValueError(f"Unsupported traditional baseline {baseline_name}")
    duration = perf_counter() - start

    run_id = stable_hash({"baseline": baseline_name, "dataset": dataset.name, "seed": seed}, prefix="run-")
    artifact_dir = ensure_dir(artifact_root / run_id)
    val_metrics = write_eval_artifacts(
        artifact_dir,
        "validation",
        y_val,
        val_predictions,
        dataset.validation_df,
        dataset.label_field,
    )
    lockbox_accuracy = None
    if not dataset.lockbox_df.empty:
        y_lockbox = dataset.lockbox_df[dataset.label_field].to_numpy()
        lockbox_metrics = write_eval_artifacts(
            artifact_dir,
            "lockbox",
            y_lockbox,
            lockbox_predictions,
            dataset.lockbox_df,
            dataset.label_field,
        )
        lockbox_accuracy = lockbox_metrics["accuracy"]

    (artifact_dir / "vectorizer.joblib.json").write_text(
        json.dumps({"kind": baseline_name, "vocabulary_size": len(vectorizer.vocabulary_)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return RunOutcome(
        run_id=run_id,
        experiment_id=baseline_name,
        dataset=dataset.name,
        model_name=baseline_name,
        model_id=baseline_name,
        budget="baseline",
        seed=seed,
        kind="baseline",
        signature=f"{dataset.name}|{baseline_name}",
        method_signature=baseline_name,
        train_seconds=duration,
        device="cpu",
        train_rows=len(train_df),
        validation_accuracy=val_metrics["accuracy"],
        lockbox_accuracy=lockbox_accuracy,
        metrics_json={"validation": val_metrics, "lockbox": {"accuracy": lockbox_accuracy}},
        params_json={"vectorizer": "tfidf-char-1-2", "seed": seed},
        artifact_dir=str(artifact_dir),
    )


def collate_batch(
    tokenizer: Any,
    batch: list[dict[str, Any]],
    max_length: int,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        [item["text"] for item in batch],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    encoded["sample_ids"] = torch.tensor([item["sample_id"] for item in batch], dtype=torch.long)
    return encoded


def evaluate_model(
    model: TransformerClassifier,
    tokenizer: Any,
    frame: pd.DataFrame,
    dataset: PreparedDataset,
    task: ResearchTaskConfig,
    device: torch.device,
    artifact_dir: Path,
    split_name: str,
    text_template: str | None = None,
) -> dict[str, Any]:
    loader = DataLoader(
        TextFrameDataset(frame, dataset.text_field, dataset.label_field, text_template=text_template),
        batch_size=task.training.eval_batch_size,
        shuffle=False,
        num_workers=task.training.num_workers,
        collate_fn=lambda rows: collate_batch(tokenizer, rows, dataset.max_length),
    )
    model.eval()
    all_predictions: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits, _ = model(batch["input_ids"], batch["attention_mask"])
            predictions = logits.argmax(dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(labels)

    predictions = np.concatenate(all_predictions) if all_predictions else np.array([], dtype=np.int64)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.int64)
    return write_eval_artifacts(artifact_dir, split_name, labels, predictions, frame, dataset.label_field)


def run_transformer_experiment(
    spec: ExperimentSpec,
    dataset: PreparedDataset,
    task: ResearchTaskConfig,
    seed: int,
    artifact_root: Path,
    kind: str = "proposal",
) -> RunOutcome:
    if torch is None or AutoTokenizer is None or get_linear_schedule_with_warmup is None:
        raise ImportError("torch and transformers are required for transformer experiments")
    set_seed(seed)
    budget = task.budget_profile(spec.budget)
    device, device_name = resolve_device(task.device_preference)

    sampled_train = sample_frame(dataset.train_search_df, dataset.label_field, budget.sample_cap, seed)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    text_template = spec.hyperparameters.get("text_template")
    text_template = str(text_template) if text_template else None
    augmentation_context = build_label_aware_mask_context(tokenizer, dataset, spec)
    train_loader = DataLoader(
        TextFrameDataset(sampled_train, dataset.text_field, dataset.label_field, text_template=text_template),
        batch_size=task.training.train_batch_size,
        shuffle=True,
        num_workers=task.training.num_workers,
        collate_fn=lambda rows: collate_batch(tokenizer, rows, dataset.max_length),
    )
    model = TransformerClassifier(
        model_id=spec.model_id,
        num_labels=dataset.num_labels,
        pooling=spec.pooling,
        head=spec.head,
        dropout=float(spec.hyperparameters.get("dropout", task.training.dropout)),
    ).to(device)
    initialize_class_priors(model, sampled_train, dataset.label_field, dataset.num_labels, spec, device)
    initialize_label_semantic_head(model, tokenizer, dataset, spec, device)
    initialize_fixed_label_anchors(model, tokenizer, dataset, spec, device)
    initialize_auxiliary_balanced_head(model, dataset, spec, device)
    initialize_lexical_prior(model, sampled_train, dataset.text_field, dataset.num_labels, spec, device)
    initialize_tfidf_teacher_prior(model, sampled_train, dataset.text_field, dataset.label_field, dataset.num_labels, spec, device, seed)
    if spec.schedule == ScheduleType.GRADUAL_UNFREEZE:
        freeze_encoder(model, True)

    optimizer = build_optimizer(model, task, spec)
    ema_decay = float(spec.hyperparameters.get("ema_decay", 0.0))
    ema_shadow = initialize_ema(model) if ema_decay > 0 else None
    total_steps = max(1, budget.epochs * len(train_loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * task.training.warmup_ratio)),
        num_training_steps=total_steps,
    )

    run_id = stable_hash({"experiment": spec.experiment_id, "dataset": dataset.name, "seed": seed}, prefix="run-")
    artifact_dir = ensure_dir(artifact_root / slugify(run_id))
    best_state = None
    best_validation = float("-inf")
    start = perf_counter()

    for epoch in range(budget.epochs):
        model.current_epoch = epoch
        if spec.schedule == ScheduleType.GRADUAL_UNFREEZE and epoch == 1:
            freeze_encoder(model, False)
            optimizer = build_optimizer(model, task, spec)

        model.train()
        progress_interval = max(1, len(train_loader) // 5)
        for step_index, batch in enumerate(train_loader, start=1):
            global_step = epoch * len(train_loader) + step_index - 1
            if spec.loss == LossType.PRIOR_ADJUSTED_LABEL_SMOOTHING:
                schedule = str(spec.hyperparameters.get("prior_adjustment_schedule", "constant"))
                start_tau = float(spec.hyperparameters.get("prior_adjustment_tau", 0.25))
                end_tau = float(spec.hyperparameters.get("prior_adjustment_tau_final", start_tau))
                progress = global_step / max(total_steps - 1, 1)
                if schedule == "linear_decay":
                    model.current_prior_adjustment_tau = start_tau + (end_tau - start_tau) * progress
                elif schedule == "linear_warmup":
                    model.current_prior_adjustment_tau = start_tau + (end_tau - start_tau) * progress
                elif schedule == "delayed_linear_warmup":
                    delay_ratio = float(spec.hyperparameters.get("prior_adjustment_delay_ratio", 0.5))
                    if progress <= delay_ratio:
                        model.current_prior_adjustment_tau = start_tau
                    else:
                        tail_progress = (progress - delay_ratio) / max(1.0 - delay_ratio, 1e-8)
                        model.current_prior_adjustment_tau = start_tau + (end_tau - start_tau) * tail_progress
                elif schedule == "constant":
                    model.current_prior_adjustment_tau = start_tau
                else:
                    raise ValueError(f"Unsupported prior_adjustment_schedule {schedule!r}")
            if spec.loss == LossType.RDROP:
                prior_schedule = str(spec.hyperparameters.get("rdrop_prior_adjustment_schedule", "constant"))
                prior_start_tau = float(spec.hyperparameters.get("rdrop_prior_adjustment_tau", 0.0))
                prior_end_tau = float(spec.hyperparameters.get("rdrop_prior_adjustment_tau_final", prior_start_tau))
                progress = global_step / max(total_steps - 1, 1)
                if prior_schedule == "constant":
                    model.current_rdrop_prior_adjustment_tau = prior_start_tau
                elif prior_schedule == "linear_decay":
                    model.current_rdrop_prior_adjustment_tau = prior_start_tau + (prior_end_tau - prior_start_tau) * progress
                elif prior_schedule == "linear_warmup":
                    model.current_rdrop_prior_adjustment_tau = prior_start_tau + (prior_end_tau - prior_start_tau) * progress
                elif prior_schedule == "delayed_linear_warmup":
                    delay_ratio = float(spec.hyperparameters.get("rdrop_prior_adjustment_delay_ratio", 0.5))
                    if progress <= delay_ratio:
                        model.current_rdrop_prior_adjustment_tau = prior_start_tau
                    else:
                        tail_progress = (progress - delay_ratio) / max(1.0 - delay_ratio, 1e-8)
                        model.current_rdrop_prior_adjustment_tau = prior_start_tau + (prior_end_tau - prior_start_tau) * tail_progress
                else:
                    raise ValueError(f"Unsupported rdrop_prior_adjustment_schedule {prior_schedule!r}")
                schedule = str(spec.hyperparameters.get("rdrop_balanced_softmax_schedule", "constant"))
                start_tau = float(spec.hyperparameters.get("rdrop_balanced_softmax_tau", 0.0))
                end_tau = float(spec.hyperparameters.get("rdrop_balanced_softmax_tau_final", start_tau))
                if schedule == "constant":
                    model.current_rdrop_balanced_softmax_tau = start_tau
                elif schedule == "linear_warmup":
                    model.current_rdrop_balanced_softmax_tau = start_tau + (end_tau - start_tau) * progress
                elif schedule == "delayed_linear_warmup":
                    delay_ratio = float(spec.hyperparameters.get("rdrop_balanced_softmax_delay_ratio", 0.5))
                    if progress <= delay_ratio:
                        model.current_rdrop_balanced_softmax_tau = start_tau
                    else:
                        tail_progress = (progress - delay_ratio) / max(1.0 - delay_ratio, 1e-8)
                        model.current_rdrop_balanced_softmax_tau = start_tau + (end_tau - start_tau) * tail_progress
                else:
                    raise ValueError(f"Unsupported rdrop_balanced_softmax_schedule {schedule!r}")
            batch = {key: value.to(device) for key, value in batch.items()}
            input_ids, attention_mask = apply_augmentation(
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
                spec,
                tokenizer,
                augmentation_context,
            )
            batch["input_ids"] = input_ids
            batch["attention_mask"] = attention_mask
            loss, _ = compute_loss(model, batch, spec)
            optimizer.zero_grad()
            loss.backward()
            fgm_epsilon = float(spec.hyperparameters.get("fgm_epsilon", 0.0))
            if fgm_epsilon > 0:
                backup: dict[str, torch.Tensor] = {}
                fgm_attack(model, fgm_epsilon, backup)
                adv_loss, _ = compute_loss(model, batch, spec)
                adv_weight = float(spec.hyperparameters.get("fgm_weight", 1.0))
                (adv_weight * adv_loss).backward()
                fgm_restore(model, backup)
            torch.nn.utils.clip_grad_norm_(model.parameters(), task.training.max_grad_norm)
            optimizer.step()
            if ema_shadow is not None:
                update_ema(model, ema_shadow, ema_decay)
            scheduler.step()
            if step_index == 1 or step_index == len(train_loader) or step_index % progress_interval == 0:
                elapsed = perf_counter() - start
                print(
                    "newalg-progress "
                    f"run_id={run_id} epoch={epoch + 1}/{budget.epochs} "
                    f"step={step_index}/{len(train_loader)} elapsed={elapsed:.1f}s "
                    f"loss={float(loss.detach().cpu().item()):.4f}",
                    flush=True,
                )

        ema_backup = apply_ema(model, ema_shadow) if ema_shadow is not None else None
        validation_metrics = evaluate_model(model, tokenizer, dataset.validation_df, dataset, task, device, artifact_dir, f"validation_epoch_{epoch}", text_template=text_template)
        if ema_backup is not None:
            restore_ema(model, ema_backup)
        if validation_metrics["accuracy"] > best_validation:
            best_validation = validation_metrics["accuracy"]
            if ema_shadow is not None:
                ema_backup = apply_ema(model, ema_shadow)
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                restore_ema(model, ema_backup)
            else:
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    crt_epochs = int(spec.hyperparameters.get("crt_retrain_epochs", 0))
    if crt_epochs > 0:
        if best_state is not None:
            model.load_state_dict(best_state)
        freeze_encoder(model, True)
        if bool(spec.hyperparameters.get("crt_reinitialize_head", True)):
            reset_module_parameters(model.classifier)
            reset_module_parameters(model.attention_pool)
            if model.prototypes is not None:
                nn.init.normal_(model.prototypes, std=0.02)
            if model.scale is not None:
                model.scale.data.fill_(10.0)
            if model.label_queries is not None:
                nn.init.normal_(model.label_queries, std=0.02)
            if model.label_bias is not None:
                model.label_bias.data.zero_()
            if model.label_attention_gate is not None:
                model.label_attention_gate.data.zero_()
        crt_loader = build_crt_loader(sampled_train, dataset, task, tokenizer, spec, text_template)
        head_parameters = prediction_head_parameters(model)
        if not head_parameters:
            raise ValueError("cRT requires trainable prediction-head parameters")
        crt_lr = float(spec.hyperparameters.get("crt_learning_rate", 5e-4))
        crt_weight_decay = float(spec.hyperparameters.get("crt_weight_decay", 0.0))
        optimizer = AdamW(head_parameters, lr=crt_lr, weight_decay=crt_weight_decay)
        total_crt_steps = max(1, crt_epochs * len(crt_loader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, int(total_crt_steps * float(spec.hyperparameters.get("crt_warmup_ratio", 0.05)))),
            num_training_steps=total_crt_steps,
        )
        for crt_epoch in range(crt_epochs):
            model.current_epoch = budget.epochs + crt_epoch
            model.train()
            progress_interval = max(1, len(crt_loader) // 5)
            for step_index, batch in enumerate(crt_loader, start=1):
                batch = {key: value.to(device) for key, value in batch.items()}
                loss, _ = compute_loss(model, batch, spec)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_parameters, task.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                if step_index == 1 or step_index == len(crt_loader) or step_index % progress_interval == 0:
                    elapsed = perf_counter() - start
                    print(
                        "newalg-progress "
                        f"run_id={run_id} phase=crt epoch={crt_epoch + 1}/{crt_epochs} "
                        f"step={step_index}/{len(crt_loader)} elapsed={elapsed:.1f}s "
                        f"loss={float(loss.detach().cpu().item()):.4f}",
                        flush=True,
                    )
            validation_metrics = evaluate_model(
                model,
                tokenizer,
                dataset.validation_df,
                dataset,
                task,
                device,
                artifact_dir,
                f"validation_crt_epoch_{crt_epoch}",
                text_template=text_template,
            )
            if validation_metrics["accuracy"] > best_validation:
                best_validation = validation_metrics["accuracy"]
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        freeze_encoder(model, False)

    duration = perf_counter() - start
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), artifact_dir / "model.pt")
    (artifact_dir / "experiment_spec.json").write_text(
        json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    validation_metrics = evaluate_model(model, tokenizer, dataset.validation_df, dataset, task, device, artifact_dir, "validation", text_template=text_template)
    lockbox_accuracy = None
    lockbox_metrics: dict[str, Any] = {"accuracy": None}
    if not dataset.lockbox_df.empty:
        lockbox_metrics = evaluate_model(model, tokenizer, dataset.lockbox_df, dataset, task, device, artifact_dir, "lockbox", text_template=text_template)
        lockbox_accuracy = lockbox_metrics["accuracy"]

    return RunOutcome(
        run_id=run_id,
        experiment_id=spec.experiment_id,
        dataset=dataset.name,
        model_name=spec.model_name,
        model_id=spec.model_id,
        budget=spec.budget.value,
        seed=seed,
        kind=kind,
        signature=spec.signature,
        method_signature=spec.method_signature,
        train_seconds=duration,
        device=device_name,
        train_rows=len(sampled_train),
        validation_accuracy=validation_metrics["accuracy"],
        lockbox_accuracy=lockbox_accuracy,
        metrics_json={"validation": validation_metrics, "lockbox": lockbox_metrics},
        params_json=spec.model_dump(mode="json"),
        artifact_dir=str(artifact_dir),
    )
