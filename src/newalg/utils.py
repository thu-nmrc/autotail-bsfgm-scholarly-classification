from __future__ import annotations

import hashlib
import json
import random
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def slugify(value: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    return "-".join(chunk for chunk in normalized.split("-") if chunk)


def stable_hash(payload: Any, prefix: str = "") -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"{prefix}{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def now_ts() -> float:
    return time.time()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


def write_frame(frame: pd.DataFrame, path_without_suffix: Path) -> Path:
    if parquet_available():
        target = path_without_suffix.with_suffix(".parquet")
        frame.to_parquet(target, index=False)
        return target
    target = path_without_suffix.with_suffix(".csv")
    frame.to_csv(target, index=False)
    return target


def read_frame(path_without_suffix: Path) -> pd.DataFrame:
    parquet_path = path_without_suffix.with_suffix(".parquet")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = path_without_suffix.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists")
