from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from datasets import Dataset, load_dataset
except ImportError:  # pragma: no cover - local dataset mode does not need huggingface datasets
    Dataset = Any
    load_dataset = None

from .config import DatasetConfig, ResearchTaskConfig
from .utils import ensure_dir, read_frame, write_frame


@dataclass
class PreparedDataset:
    name: str
    train_df: pd.DataFrame
    validation_df: pd.DataFrame
    lockbox_df: pd.DataFrame
    test_df: pd.DataFrame
    label_list: list[int]
    text_field: str
    label_field: str
    max_length: int

    @property
    def train_search_df(self) -> pd.DataFrame:
        return self.train_df

    @property
    def num_labels(self) -> int:
        return len(self.label_list)


class DatasetRepository:
    def __init__(self, task: ResearchTaskConfig) -> None:
        self.task = task
        self.cache_root = ensure_dir(task.output_root_path / "cache" / "datasets")

    def prepare(self, dataset_name: str) -> PreparedDataset:
        dataset_cache = ensure_dir(self.cache_root / dataset_name)
        manifest_path = dataset_cache / "manifest.json"
        if manifest_path.exists() and self._cache_matches_lockbox_policy(dataset_name, manifest_path):
            return self._load_cached(dataset_name, manifest_path)

        config = self.task.datasets[dataset_name]
        frame_bundle = self._build_frames(config)
        train_df = frame_bundle["train"].copy()
        validation_df = frame_bundle["validation"].copy()
        test_df = frame_bundle["test"].copy()

        lockbox_fraction = self._lockbox_fraction_for(dataset_name)
        if lockbox_fraction > 0:
            train_df, lockbox_df = self._split_lockbox(train_df, config.label_field, lockbox_fraction)
        else:
            lockbox_df = train_df.iloc[0:0].copy()

        train_df = train_df.reset_index(drop=True)
        validation_df = validation_df.reset_index(drop=True)
        lockbox_df = lockbox_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        label_values = sorted(int(value) for value in pd.concat([train_df, validation_df])[config.label_field].unique())
        manifest = {
            "dataset_name": dataset_name,
            "lockbox_fraction": lockbox_fraction,
            "lockbox_seed": self.task.lockbox_seed,
            "text_field": config.text_field,
            "label_field": config.label_field,
            "max_length": config.max_length,
            "label_list": label_values,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "lockbox_rows": len(lockbox_df),
            "test_rows": len(test_df),
        }

        write_frame(train_df, dataset_cache / "train")
        write_frame(validation_df, dataset_cache / "validation")
        write_frame(lockbox_df, dataset_cache / "lockbox")
        write_frame(test_df, dataset_cache / "test")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return self._load_cached(dataset_name, manifest_path)

    def _split_lockbox(
        self,
        train_df: pd.DataFrame,
        label_field: str,
        lockbox_fraction: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        counts = train_df[label_field].value_counts()
        rare_labels = set(counts[counts < 2].index.tolist())
        rare_train_df = train_df[train_df[label_field].isin(rare_labels)]
        splittable_df = train_df[~train_df[label_field].isin(rare_labels)]
        if splittable_df.empty:
            return train_df, train_df.iloc[0:0].copy()
        split_train_df, lockbox_df = train_test_split(
            splittable_df,
            test_size=lockbox_fraction,
            random_state=self.task.lockbox_seed,
            stratify=splittable_df[label_field],
        )
        train_with_rare_df = pd.concat([split_train_df, rare_train_df], ignore_index=True)
        return train_with_rare_df, lockbox_df

    def _lockbox_fraction_for(self, dataset_name: str) -> float:
        if dataset_name == self.task.primary_dataset:
            return self.task.lockbox_fraction
        leaderboard_datasets = {
            board.dataset
            for board in self.task.leaderboards
            if board.status.value in {"active", "promising", "ready_for_submission"}
        }
        return self.task.lockbox_fraction if dataset_name in leaderboard_datasets else 0.0

    def _cache_matches_lockbox_policy(self, dataset_name: str, manifest_path: Path) -> bool:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_fraction = self._lockbox_fraction_for(dataset_name)
        expected_seed = self.task.lockbox_seed
        cached_fraction = float(manifest.get("lockbox_fraction", 0.0))
        cached_seed = int(manifest.get("lockbox_seed", expected_seed))
        if abs(cached_fraction - expected_fraction) > 1e-12:
            return False
        if cached_seed != expected_seed:
            return False
        if expected_fraction > 0 and int(manifest.get("lockbox_rows", 0)) == 0:
            return False
        return True

    def _load_cached(self, dataset_name: str, manifest_path: Path) -> PreparedDataset:
        cache_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PreparedDataset(
            name=dataset_name,
            train_df=read_frame(cache_dir / "train"),
            validation_df=read_frame(cache_dir / "validation"),
            lockbox_df=read_frame(cache_dir / "lockbox"),
            test_df=read_frame(cache_dir / "test"),
            label_list=[int(value) for value in manifest["label_list"]],
            text_field=manifest["text_field"],
            label_field=manifest["label_field"],
            max_length=int(manifest["max_length"]),
        )

    def _build_frames(self, config: DatasetConfig) -> dict[str, pd.DataFrame]:
        if config.source.value == "local":
            return self._load_local(config)
        if load_dataset is None:
            raise ImportError("datasets is required for Hugging Face dataset sources")

        dataset_dict = load_dataset(path=config.hf_path, name=config.hf_name)
        return {
            "train": self._dataset_to_frame(dataset_dict[config.train_split], config),
            "validation": self._dataset_to_frame(dataset_dict[config.validation_split], config),
            "test": self._dataset_to_frame(dataset_dict[config.test_split], config, include_label=False),
        }

    def _dataset_to_frame(
        self,
        dataset: Dataset,
        config: DatasetConfig,
        include_label: bool = True,
    ) -> pd.DataFrame:
        payload: dict[str, Any] = {config.text_field: dataset[config.text_field]}
        if include_label and config.label_field in dataset.column_names:
            payload[config.label_field] = dataset[config.label_field]
        elif not include_label:
            payload[config.label_field] = [-1 for _ in range(len(dataset))]
        if "idx" in dataset.column_names:
            payload["sample_id"] = dataset["idx"]
        else:
            payload["sample_id"] = list(range(len(dataset)))
        frame = pd.DataFrame(payload)
        frame[config.text_field] = frame[config.text_field].astype(str)
        frame[config.label_field] = frame[config.label_field].astype(int)
        return frame

    def _load_local(self, config: DatasetConfig) -> dict[str, pd.DataFrame]:
        root = Path(config.local_path or "")
        if not root.exists():
            raise FileNotFoundError(f"Local dataset path not found: {root}")
        return {
            "train": self._load_local_split(root / f"{config.train_split}.jsonl", config, include_label=True),
            "validation": self._load_local_split(root / f"{config.validation_split}.jsonl", config, include_label=True),
            "test": self._load_local_split(root / f"{config.test_split}.jsonl", config, include_label=False),
        }

    def _load_local_split(
        self,
        path: Path,
        config: DatasetConfig,
        include_label: bool,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            rows.append(
                {
                    config.text_field: str(item[config.text_field]),
                    config.label_field: int(item[config.label_field]) if include_label else int(item.get(config.label_field, -1)),
                    "sample_id": item.get("sample_id", idx),
                }
            )
        return pd.DataFrame(rows)


def sample_frame(frame: pd.DataFrame, label_field: str, sample_cap: int | None, seed: int) -> pd.DataFrame:
    if sample_cap is None or len(frame) <= sample_cap:
        return frame
    sampled_parts: list[pd.DataFrame] = []
    remaining = sample_cap
    groups = list(frame.groupby(label_field, sort=False))
    for group_index, (_, chunk) in enumerate(groups):
        # Preserve every class while roughly keeping the original label distribution.
        proportional = int(sample_cap * len(chunk) / len(frame))
        min_required = max(1, proportional)
        groups_left = len(groups) - group_index - 1
        take = min(len(chunk), max(1, min(min_required, remaining - groups_left)))
        sampled_parts.append(chunk.sample(n=take, random_state=seed))
        remaining -= take
    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > sample_cap:
        sampled = sampled.sample(n=sample_cap, random_state=seed).reset_index(drop=True)
    return sampled.reset_index(drop=True)
