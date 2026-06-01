from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import ResearchTaskConfig
from .datasets import DatasetRepository
from .inference import label_names_for_dataset
from .registry import RunRegistry
from .utils import ensure_dir, read_frame, stable_hash, write_frame


def evaluate_prediction_ensemble(
    task: ResearchTaskConfig,
    registry: RunRegistry,
    run_ids: list[str],
    *,
    mode: str = "hard",
    name: str | None = None,
) -> dict[str, Any]:
    if len(run_ids) < 2:
        raise ValueError("At least two run ids are required for an ensemble")
    runs = registry.fetch_runs(
        f"run_id IN ({','.join(['?'] * len(run_ids))}) AND status = 'completed'",
        run_ids,
    )
    if len(runs) != len(run_ids):
        found = set(runs["run_id"].astype(str).tolist()) if not runs.empty else set()
        missing = [run_id for run_id in run_ids if run_id not in found]
        raise ValueError(f"Completed runs not found: {missing}")

    ordered = pd.DataFrame({"run_id": run_ids}).merge(runs, on="run_id", how="left")
    artifact_root = ensure_dir(task.resolve_path(task.artifacts_dir))
    experiment_id = name or f"ensemble-{stable_hash({'run_ids': run_ids, 'mode': mode}, prefix='')}"
    run_id = stable_hash({"ensemble": experiment_id, "run_ids": run_ids, "mode": mode}, prefix="run-")
    artifact_dir = ensure_dir(artifact_root / run_id)

    start = perf_counter()
    validation_metrics = _evaluate_split(ordered, artifact_dir, "validation", mode)
    lockbox_metrics = _evaluate_split(ordered, artifact_dir, "lockbox", mode)
    test_metrics = _write_test_predictions_if_available(task, ordered, artifact_dir, mode)
    duration = perf_counter() - start

    params = {
        "mode": mode,
        "run_ids": run_ids,
        "members": [
            {
                "run_id": str(row["run_id"]),
                "experiment_id": str(row["experiment_id"]),
                "model_name": str(row["model_name"]),
                "validation_accuracy": float(row["validation_accuracy"]),
                "lockbox_accuracy": None if pd.isna(row["lockbox_accuracy"]) else float(row["lockbox_accuracy"]),
            }
            for _, row in ordered.iterrows()
        ],
    }
    (artifact_dir / "ensemble_spec.json").write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")
    outcome = {
        "run_id": run_id,
        "experiment_id": experiment_id,
        "dataset": task.primary_dataset,
        "model_name": experiment_id,
        "model_id": "prediction-ensemble",
        "budget": "ensemble",
        "seed": -1,
        "kind": "proposal",
        "status": "completed",
        "signature": f"ensemble|{mode}|{'/'.join(run_ids)}",
        "method_signature": f"ensemble|{mode}",
        "train_seconds": duration,
        "device": "cpu",
        "train_rows": 0,
        "validation_accuracy": validation_metrics["accuracy"],
        "lockbox_accuracy": lockbox_metrics["accuracy"],
        "sanity_accuracy": None,
        "params_json": params,
        "metrics_json": {"validation": validation_metrics, "lockbox": lockbox_metrics, "test": test_metrics},
        "artifact_dir": str(artifact_dir),
    }
    registry.upsert_run(outcome)
    return outcome


def _evaluate_split(runs: pd.DataFrame, artifact_dir: Path, split: str, mode: str) -> dict[str, Any]:
    frames = []
    weights = []
    for _, row in runs.iterrows():
        frame = read_frame(Path(str(row["artifact_dir"])) / f"{split}_predictions")
        frames.append(frame[["sample_id", "label", "prediction"]].sort_values("sample_id").reset_index(drop=True))
        if mode == "validation_weight":
            weights.append(float(row["validation_accuracy"]))
        else:
            weights.append(1.0)
    base = frames[0][["sample_id", "label"]].copy()
    labels = base["label"].to_numpy()
    predictions = np.vstack([frame["prediction"].to_numpy(dtype=int) for frame in frames])
    voted = _weighted_vote(predictions, np.asarray(weights, dtype=float))
    output = base.copy()
    output["prediction"] = voted
    output["correct"] = output["label"].to_numpy() == output["prediction"].to_numpy()
    write_frame(output, artifact_dir / f"{split}_predictions")
    write_frame(output.loc[~output["correct"]], artifact_dir / f"{split}_errors")
    cm = confusion_matrix(labels, voted)
    (artifact_dir / f"{split}_confusion_matrix.json").write_text(json.dumps(cm.tolist()), encoding="utf-8")
    return {"accuracy": float((voted == labels).mean() * 100.0), "errors": int((voted != labels).sum())}


def _weighted_vote(predictions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    num_labels = int(predictions.max()) + 1
    scores = np.zeros((predictions.shape[1], num_labels), dtype=float)
    columns = np.arange(predictions.shape[1])
    for row, weight in zip(predictions, weights):
        scores[columns, row] += weight
    return scores.argmax(axis=1)


def _write_test_predictions_if_available(
    task: ResearchTaskConfig,
    runs: pd.DataFrame,
    artifact_dir: Path,
    mode: str,
) -> dict[str, Any]:
    frames = []
    weights = []
    for _, row in runs.iterrows():
        try:
            frame = read_frame(Path(str(row["artifact_dir"])) / "test_predictions")
        except FileNotFoundError:
            return {"available": False, "rows": 0}
        frames.append(frame[["sample_id", "prediction"]].sort_values("sample_id").reset_index(drop=True))
        weights.append(float(row["validation_accuracy"]) if mode == "validation_weight" else 1.0)

    base = frames[0][["sample_id"]].copy()
    predictions = np.vstack([frame["prediction"].to_numpy(dtype=int) for frame in frames])
    voted = _weighted_vote(predictions, np.asarray(weights, dtype=float))
    output = base.copy()
    output["prediction"] = voted

    dataset = DatasetRepository(task).prepare(str(runs.iloc[0]["dataset"]))
    names = label_names_for_dataset(dataset)
    output["label_name"] = [names[int(prediction)] for prediction in voted]
    write_frame(output, artifact_dir / "test_predictions")
    return {"available": True, "rows": int(len(output))}
