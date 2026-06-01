from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, get_linear_schedule_with_warmup

from .config import ResearchTaskConfig
from .datasets import DatasetRepository, PreparedDataset, sample_frame
from .training import resolve_device
from .utils import ensure_dir, set_seed, slugify


class TextOnlyDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )


@dataclass
class PretrainOutcome:
    output_dir: str
    base_model_id: str
    dataset: str
    train_rows: int
    epochs: int
    mlm_probability: float
    learning_rate: float
    train_seconds: float
    final_loss: float
    device: str

    def to_json(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "base_model_id": self.base_model_id,
            "dataset": self.dataset,
            "train_rows": self.train_rows,
            "epochs": self.epochs,
            "mlm_probability": self.mlm_probability,
            "learning_rate": self.learning_rate,
            "train_seconds": self.train_seconds,
            "final_loss": self.final_loss,
            "device": self.device,
        }


def continue_pretrain_mlm(
    task: ResearchTaskConfig,
    dataset_name: str,
    base_model_id: str,
    output_dir: str | Path,
    *,
    sample_cap: int = 4096,
    epochs: int = 1,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    mlm_probability: float = 0.15,
    seed: int = 13,
) -> PretrainOutcome:
    set_seed(seed)
    repository = DatasetRepository(task)
    dataset = repository.prepare(dataset_name)
    frame = sample_frame(dataset.train_search_df, dataset.label_field, sample_cap, seed)
    texts = frame[dataset.text_field].astype(str).tolist()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    model = AutoModelForMaskedLM.from_pretrained(base_model_id)
    device, device_name = resolve_device(task.device_preference)
    model.to(device)

    loader = _build_mlm_loader(dataset, texts, tokenizer, batch_size, mlm_probability)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=task.training.weight_decay)
    total_steps = max(1, epochs * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * task.training.warmup_ratio)),
        num_training_steps=total_steps,
    )

    start = perf_counter()
    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), task.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.detach().cpu().item())

    duration = perf_counter() - start
    target = ensure_dir(Path(output_dir))
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    outcome = PretrainOutcome(
        output_dir=str(target),
        base_model_id=base_model_id,
        dataset=dataset_name,
        train_rows=len(texts),
        epochs=epochs,
        mlm_probability=mlm_probability,
        learning_rate=learning_rate,
        train_seconds=duration,
        final_loss=last_loss,
        device=device_name,
    )
    (target / "newalg_tapt_metadata.json").write_text(json.dumps(outcome.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    return outcome


def default_tapt_output_dir(task: ResearchTaskConfig, dataset_name: str, base_model_id: str, sample_cap: int, seed: int) -> Path:
    safe_model = slugify(base_model_id.replace("/", "-"))
    return task.output_root_path / "pretrained" / f"{dataset_name}-tapt-{safe_model}-n{sample_cap}-seed{seed}"


def _build_mlm_loader(
    dataset: PreparedDataset,
    texts: list[str],
    tokenizer: Any,
    batch_size: int,
    mlm_probability: float,
) -> DataLoader:
    text_dataset = TextOnlyDataset(texts, tokenizer, dataset.max_length)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
        return_tensors="pt",
    )
    return DataLoader(text_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)

