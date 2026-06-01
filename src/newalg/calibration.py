from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from .config import ExperimentSpec, ResearchTaskConfig
from .datasets import DatasetRepository, PreparedDataset
from .registry import RunRegistry
from .training import AutoTokenizer, TextFrameDataset, TransformerClassifier, collate_batch, initialize_class_priors, resolve_device, torch
from .utils import ensure_dir, stable_hash, write_frame


DEFAULT_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def run_bias_calibration_oof(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    base_run_id: str,
    *,
    folds: int = 5,
    alphas: list[float] | None = None,
    l2: float = 0.02,
    seed: int = 20260511,
) -> dict[str, Any]:
    if torch is None or AutoTokenizer is None:
        raise ImportError("torch and transformers are required for calibration")

    started = perf_counter()
    base_run = _fetch_completed_run(registry, base_run_id)
    artifact_dir = Path(str(base_run["artifact_dir"]))
    spec = ExperimentSpec.model_validate_json((artifact_dir / "experiment_spec.json").read_text(encoding="utf-8"))
    dataset = DatasetRepository(task).prepare(str(base_run["dataset"]))
    model, tokenizer, device, device_name = _load_model(task, dataset, spec, artifact_dir)

    validation_logits, validation_labels = collect_logits(model, tokenizer, dataset.validation_df, dataset, task, device)
    lockbox_logits, lockbox_labels = collect_logits(model, tokenizer, dataset.lockbox_df, dataset, task, device)
    alpha_values = alphas or DEFAULT_ALPHAS
    baseline_validation = accuracy_from_logits(validation_logits, validation_labels)
    baseline_lockbox = accuracy_from_logits(lockbox_logits, lockbox_labels)

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    alpha_summaries: list[dict[str, Any]] = []
    fold_biases: dict[float, list[list[float]]] = {alpha: [] for alpha in alpha_values}
    for alpha in alpha_values:
        oof_predictions = np.empty_like(validation_labels)
        fold_scores: list[float] = []
        for train_index, test_index in skf.split(validation_logits, validation_labels):
            bias = learn_class_bias(validation_logits[train_index], validation_labels[train_index], l2=l2)
            scaled = alpha * bias
            predictions = (validation_logits[test_index] + scaled).argmax(axis=1)
            oof_predictions[test_index] = predictions
            fold_scores.append(accuracy_from_predictions(predictions, validation_labels[test_index]))
            fold_biases[alpha].append(scaled.tolist())
        oof_accuracy = accuracy_from_predictions(oof_predictions, validation_labels)
        alpha_summaries.append(
            {
                "alpha": alpha,
                "oof_validation_accuracy": oof_accuracy,
                "fold_scores": fold_scores,
                "oof_delta": oof_accuracy - baseline_validation,
            }
        )

    best = max(alpha_summaries, key=lambda item: item["oof_validation_accuracy"])
    final_bias = learn_class_bias(validation_logits, validation_labels, l2=l2)
    scaled_final_bias = float(best["alpha"]) * final_bias
    validation_predictions = (validation_logits + scaled_final_bias).argmax(axis=1)
    lockbox_predictions = (lockbox_logits + scaled_final_bias).argmax(axis=1)
    validation_accuracy = accuracy_from_predictions(validation_predictions, validation_labels)
    lockbox_accuracy = accuracy_from_predictions(lockbox_predictions, lockbox_labels)

    run_id = stable_hash(
        {"bias_calibration_oof": base_run_id, "folds": folds, "l2": l2, "seed": seed, "alpha": best["alpha"]},
        prefix="run-",
    )
    output_dir = ensure_dir(task.output_root_path / "artifacts" / run_id)
    _write_prediction_artifacts(output_dir, "validation", dataset.validation_df, validation_labels, validation_predictions, dataset.label_field)
    _write_prediction_artifacts(output_dir, "lockbox", dataset.lockbox_df, lockbox_labels, lockbox_predictions, dataset.label_field)
    calibration_payload = {
        "base_run_id": base_run_id,
        "folds": folds,
        "l2": l2,
        "seed": seed,
        "baseline_validation_accuracy": baseline_validation,
        "baseline_lockbox_accuracy": baseline_lockbox,
        "alpha_summaries": alpha_summaries,
        "selected_alpha": best["alpha"],
        "final_bias": scaled_final_bias.tolist(),
        "validation_accuracy": validation_accuracy,
        "lockbox_accuracy": lockbox_accuracy,
    }
    (output_dir / "calibration.json").write_text(json.dumps(calibration_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    row = {
        "run_id": run_id,
        "experiment_id": f"oof-class-bias-{base_run_id}",
        "dataset": dataset.name,
        "model_name": f"{base_run['model_name']}-oof-class-bias",
        "model_id": str(base_run["model_id"]),
        "budget": "calibration_oof",
        "seed": int(base_run["seed"]),
        "kind": "proposal",
        "status": "completed",
        "signature": f"oof_class_bias|{base_run_id}|{folds}|{l2}|{seed}",
        "method_signature": "oof_class_bias_calibration",
        "train_seconds": perf_counter() - started,
        "device": device_name,
        "train_rows": 0,
        "validation_accuracy": validation_accuracy,
        "lockbox_accuracy": lockbox_accuracy,
        "sanity_accuracy": None,
        "params_json": calibration_payload,
        "metrics_json": {
            "oof": {"best": best, "alpha_summaries": alpha_summaries},
            "validation": {"accuracy": validation_accuracy, "baseline_accuracy": baseline_validation},
            "lockbox": {"accuracy": lockbox_accuracy, "baseline_accuracy": baseline_lockbox},
        },
        "artifact_dir": str(output_dir),
    }
    registry.upsert_run(row)
    return row


def collect_logits(
    model: TransformerClassifier,
    tokenizer: Any,
    frame: pd.DataFrame,
    dataset: PreparedDataset,
    task: ResearchTaskConfig,
    device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        TextFrameDataset(frame, dataset.text_field, dataset.label_field),
        batch_size=task.training.eval_batch_size,
        shuffle=False,
        num_workers=task.training.num_workers,
        collate_fn=lambda rows: collate_batch(tokenizer, rows, dataset.max_length),
    )
    logits: list[np.ndarray] = []
    labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels.extend(int(value) for value in batch["labels"].cpu().tolist())
            batch = {key: value.to(device) for key, value in batch.items()}
            output, _ = model(batch["input_ids"], batch["attention_mask"])
            logits.append(output.cpu().numpy())
    return np.concatenate(logits), np.asarray(labels, dtype=int)


def learn_class_bias(logits: np.ndarray, labels: np.ndarray, *, l2: float = 0.02) -> np.ndarray:
    bias = np.zeros(int(logits.shape[1]), dtype=np.float32)
    for step in [0.2, 0.1, 0.05, 0.025, 0.0125]:
        improved = True
        while improved:
            improved = False
            for cls in range(len(bias)):
                current = float(bias[cls])
                best_score = accuracy_from_logits(logits, labels, bias) - l2 * float(np.square(bias).sum())
                best_value = current
                for value in [current - step, current + step]:
                    trial = bias.copy()
                    trial[cls] = value
                    score = accuracy_from_logits(logits, labels, trial) - l2 * float(np.square(trial).sum())
                    if score > best_score:
                        best_score = score
                        best_value = value
                if best_value != current:
                    bias[cls] = best_value
                    improved = True
    return bias


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray, bias: np.ndarray | None = None) -> float:
    adjusted = logits if bias is None else logits + bias
    return accuracy_from_predictions(adjusted.argmax(axis=1), labels)


def accuracy_from_predictions(predictions: np.ndarray, labels: np.ndarray) -> float:
    return float((predictions == labels).mean() * 100.0)


def _fetch_completed_run(registry: RunRegistry, run_id: str) -> pd.Series:
    runs = registry.fetch_runs("run_id = ? AND status = 'completed'", [run_id])
    if runs.empty:
        raise ValueError(f"Completed run not found: {run_id}")
    return runs.iloc[-1]


def _load_model(
    task: ResearchTaskConfig,
    dataset: PreparedDataset,
    spec: ExperimentSpec,
    artifact_dir: Path,
) -> tuple[TransformerClassifier, Any, Any, str]:
    device, device_name = resolve_device(task.device_preference)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    model = TransformerClassifier(
        spec.model_id,
        dataset.num_labels,
        spec.pooling,
        spec.head,
        float(spec.hyperparameters.get("dropout", task.training.dropout)),
    ).to(device)
    initialize_class_priors(model, dataset.train_search_df, dataset.label_field, dataset.num_labels, spec, device)
    model.load_state_dict(torch.load(artifact_dir / "model.pt", map_location="cpu"))
    model.to(device)
    return model, tokenizer, device, device_name


def _write_prediction_artifacts(
    artifact_dir: Path,
    split: str,
    frame: pd.DataFrame,
    labels: np.ndarray,
    predictions: np.ndarray,
    label_field: str,
) -> None:
    output = frame.copy()
    output["prediction"] = predictions
    output["correct"] = output[label_field].to_numpy() == predictions
    write_frame(output, artifact_dir / f"{split}_predictions")
    write_frame(output.loc[~output["correct"]], artifact_dir / f"{split}_errors")
    (artifact_dir / f"{split}_confusion_matrix.json").write_text(
        json.dumps(confusion_matrix(labels, predictions).tolist()),
        encoding="utf-8",
    )
