from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import duckdb
except ImportError:  # pragma: no cover - fallback path used in lightweight environments
    duckdb = None

from .utils import ensure_dir


class RunRegistry:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.backend = self._select_backend()
        self.conn = duckdb.connect(str(self.path)) if self.backend == "duckdb" else sqlite3.connect(str(self.path))
        self._init_schema()

    def _select_backend(self) -> str:
        if duckdb is None:
            return "sqlite"
        if not self.path.exists():
            return "duckdb"
        try:
            with self.path.open("rb") as handle:
                header = handle.read(16)
        except OSError:
            return "duckdb"
        if header.startswith(b"SQLite format 3"):
            return "sqlite"
        return "duckdb"

    def _init_schema(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id VARCHAR PRIMARY KEY,
                experiment_id VARCHAR,
                dataset VARCHAR,
                model_name VARCHAR,
                model_id VARCHAR,
                budget VARCHAR,
                seed INTEGER,
                kind VARCHAR,
                status VARCHAR,
                signature VARCHAR,
                method_signature VARCHAR,
                train_seconds DOUBLE,
                device VARCHAR,
                train_rows INTEGER,
                validation_accuracy DOUBLE,
                lockbox_accuracy DOUBLE,
                sanity_accuracy DOUBLE,
                params_json TEXT,
                metrics_json TEXT,
                artifact_dir VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

    def upsert_run(self, row: dict[str, Any]) -> None:
        payload = dict(row)
        payload["params_json"] = json.dumps(payload.get("params_json", {}), ensure_ascii=False)
        payload["metrics_json"] = json.dumps(payload.get("metrics_json", {}), ensure_ascii=False)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, experiment_id, dataset, model_name, model_id, budget, seed, kind, status,
                signature, method_signature, train_seconds, device, train_rows, validation_accuracy,
                lockbox_accuracy, sanity_accuracy, params_json, metrics_json, artifact_dir
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                payload.get("run_id"),
                payload.get("experiment_id"),
                payload.get("dataset"),
                payload.get("model_name"),
                payload.get("model_id"),
                payload.get("budget"),
                payload.get("seed"),
                payload.get("kind"),
                payload.get("status"),
                payload.get("signature"),
                payload.get("method_signature"),
                payload.get("train_seconds"),
                payload.get("device"),
                payload.get("train_rows"),
                payload.get("validation_accuracy"),
                payload.get("lockbox_accuracy"),
                payload.get("sanity_accuracy"),
                payload.get("params_json"),
                payload.get("metrics_json"),
                payload.get("artifact_dir"),
            ],
        )
        if self.backend == "sqlite":
            self.conn.commit()

    def fetch_runs(self, where: str = "TRUE", params: list[Any] | None = None) -> pd.DataFrame:
        query = f"SELECT * FROM runs WHERE {where} ORDER BY created_at"
        if self.backend == "duckdb":
            return self.conn.execute(query, params or []).df()
        return pd.read_sql_query(query, self.conn, params=params or [])

    def fetch_best_baseline(self, dataset: str) -> pd.DataFrame:
        query = """
        SELECT model_name, AVG(validation_accuracy) AS validation_accuracy,
               AVG(lockbox_accuracy) AS lockbox_accuracy, AVG(train_seconds) AS train_seconds
        FROM runs
        WHERE dataset = ? AND kind = 'baseline' AND status = 'completed'
        GROUP BY model_name
        ORDER BY validation_accuracy DESC
        LIMIT 1
        """
        if self.backend == "duckdb":
            return self.conn.execute(query, [dataset]).df()
        return pd.read_sql_query(query, self.conn, params=[dataset])

    def existing_signatures(self) -> set[str]:
        cursor = self.conn.execute("SELECT DISTINCT signature FROM runs WHERE signature IS NOT NULL")
        rows = cursor.fetchall()
        return {row[0] for row in rows if row[0]}
