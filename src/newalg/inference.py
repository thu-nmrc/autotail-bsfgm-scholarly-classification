from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import ExperimentSpec, ResearchTaskConfig
from .datasets import DatasetRepository, PreparedDataset
from .registry import RunRegistry
from .training import (
    AutoTokenizer,
    DataLoader,
    TextFrameDataset,
    TransformerClassifier,
    collate_batch,
    initialize_class_priors,
    initialize_fixed_label_anchors,
    initialize_label_semantic_head,
    resolve_device,
    torch,
)
from .utils import write_frame


TNEWS_LABEL_NAMES = ["100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112", "113", "114", "115", "116"]


def label_names_for_dataset(dataset: PreparedDataset) -> list[str]:
    if dataset.name == "tnews" and dataset.label_list == list(range(15)):
        return TNEWS_LABEL_NAMES
    return [str(label) for label in dataset.label_list]


def predict_test_for_run(task: ResearchTaskConfig, registry: RunRegistry, run_id: str) -> Path:
    if torch is None or AutoTokenizer is None:
        raise ImportError("torch and transformers are required for transformer test prediction")

    runs = registry.fetch_runs("run_id = ? AND status = 'completed'", [run_id])
    if runs.empty:
        raise ValueError(f"Completed run not found: {run_id}")
    run = runs.iloc[-1]
    if str(run["model_id"]) == "prediction-ensemble":
        raise ValueError("Use evaluate-ensemble after member test predictions exist to build ensemble test predictions")

    artifact_dir = Path(str(run["artifact_dir"]))
    spec_path = artifact_dir / "experiment_spec.json"
    model_path = artifact_dir / "model.pt"
    if not spec_path.exists() or not model_path.exists():
        raise ValueError(f"Run {run_id} does not contain a transformer checkpoint and experiment_spec.json")

    spec = ExperimentSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    dataset = DatasetRepository(task).prepare(str(run["dataset"]))
    device, _ = resolve_device(task.device_preference)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = TransformerClassifier(
        model_id=spec.model_id,
        num_labels=dataset.num_labels,
        pooling=spec.pooling,
        head=spec.head,
        dropout=float(spec.hyperparameters.get("dropout", task.training.dropout)),
    ).to(device)
    initialize_label_semantic_head(model, tokenizer, dataset, spec, device)
    initialize_fixed_label_anchors(model, tokenizer, dataset, spec, device)
    initialize_class_priors(model, dataset.train_df, dataset.label_field, dataset.num_labels, spec, device)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    text_template = spec.hyperparameters.get("text_template")
    predictions = predict_frame(
        model,
        tokenizer,
        dataset.test_df,
        dataset,
        task,
        device,
        text_template=str(text_template) if text_template else None,
    )
    output = dataset.test_df.copy()
    output["prediction"] = predictions
    names = label_names_for_dataset(dataset)
    output["label_name"] = [names[int(prediction)] for prediction in predictions]
    return write_frame(output, artifact_dir / "test_predictions")


def predict_frame(
    model: TransformerClassifier,
    tokenizer: Any,
    frame: pd.DataFrame,
    dataset: PreparedDataset,
    task: ResearchTaskConfig,
    device: Any,
    text_template: str | None = None,
) -> list[int]:
    loader = DataLoader(
        TextFrameDataset(frame, dataset.text_field, dataset.label_field, text_template=text_template),
        batch_size=task.training.eval_batch_size,
        shuffle=False,
        num_workers=task.training.num_workers,
        collate_fn=lambda rows: collate_batch(tokenizer, rows, dataset.max_length),
    )
    model.eval()
    predictions: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits, _ = model(batch["input_ids"], batch["attention_mask"])
            predictions.extend(int(value) for value in logits.argmax(dim=-1).cpu().tolist())
    return predictions
