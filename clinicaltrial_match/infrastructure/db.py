"""
SQLite database layer with WAL mode for concurrent reads.

Provides schema initialization, typed CRUD helpers, and connection management.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS trials (
    nct_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    brief_summary TEXT DEFAULT '',
    conditions TEXT DEFAULT '[]',
    interventions TEXT DEFAULT '[]',
    phase TEXT DEFAULT '',
    status TEXT DEFAULT 'UNKNOWN',
    eligibility_text TEXT DEFAULT '',
    eligibility_criteria TEXT,
    sponsor TEXT DEFAULT '',
    locations TEXT DEFAULT '[]',
    start_date TEXT,
    last_updated TEXT,
    cached_at REAL NOT NULL,
    embedding BLOB
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    raw_input TEXT NOT NULL,
    features TEXT,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS match_results (
    match_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    nct_id TEXT NOT NULL,
    composite_score REAL NOT NULL,
    confidence TEXT NOT NULL,
    explanation TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id),
    FOREIGN KEY (nct_id) REFERENCES trials(nct_id)
);

CREATE TABLE IF NOT EXISTS sync_jobs (
    job_id TEXT PRIMARY KEY,
    condition TEXT NOT NULL,
    status TEXT DEFAULT 'queued',
    trials_fetched INTEGER DEFAULT 0,
    trials_parsed INTEGER DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    completed_at REAL
);

CREATE TABLE IF NOT EXISTS diagnosis_equiv_cache (
    cache_key TEXT PRIMARY KEY,
    equivalent INTEGER NOT NULL,
    cached_at REAL NOT NULL,
    ttl_seconds INTEGER NOT NULL DEFAULT 604800
);

CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
CREATE INDEX IF NOT EXISTS idx_trials_cached_at ON trials(cached_at);
CREATE INDEX IF NOT EXISTS idx_match_results_patient ON match_results(patient_id);
CREATE INDEX IF NOT EXISTS idx_match_results_score ON match_results(composite_score DESC);
"""


class Database:
    def __init__(self, path: str) -> None:
        if path == ":memory:":
            self._path = ":memory:"
        else:
            resolved = str(Path(path).expanduser().resolve())
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
            self._path = resolved
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Database not connected — call connect() first")
        return self._conn

    # --- Trials ---

    def upsert_trial(self, trial_dict: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO trials
                (nct_id, title, brief_summary, conditions, interventions, phase, status,
                 eligibility_text, eligibility_criteria, sponsor, locations,
                 start_date, last_updated, cached_at, embedding)
            VALUES
                (:nct_id, :title, :brief_summary, :conditions, :interventions, :phase, :status,
                 :eligibility_text, :eligibility_criteria, :sponsor, :locations,
                 :start_date, :last_updated, :cached_at, :embedding)
            ON CONFLICT(nct_id) DO UPDATE SET
                title=excluded.title,
                brief_summary=excluded.brief_summary,
                conditions=excluded.conditions,
                interventions=excluded.interventions,
                phase=excluded.phase,
                status=excluded.status,
                eligibility_text=excluded.eligibility_text,
                eligibility_criteria=excluded.eligibility_criteria,
                sponsor=excluded.sponsor,
                locations=excluded.locations,
                start_date=excluded.start_date,
                last_updated=excluded.last_updated,
                cached_at=excluded.cached_at,
                embedding=COALESCE(excluded.embedding, trials.embedding)
            """,
            trial_dict,
        )
        self.conn.commit()

    def update_trial_embedding(self, nct_id: str, embedding_bytes: bytes) -> None:
        self.conn.execute(
            "UPDATE trials SET embedding=? WHERE nct_id=?", (embedding_bytes, nct_id)
        )
        self.conn.commit()

    def get_trial(self, nct_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM trials WHERE nct_id=?", (nct_id,)).fetchone()
        return dict(row) if row else None

    def list_trials(
        self,
        status: str | None = None,
        condition: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        conditions = []
        params: list[Any] = []
        if status:
            conditions.append("status=?")
            params.append(status)
        if condition:
            conditions.append("conditions LIKE ?")
            params.append(f"%{condition}%")
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        total = self.conn.execute(f"SELECT COUNT(*) FROM trials {where}", params).fetchone()[0]
        rows = self.conn.execute(
            f"SELECT * FROM trials {where} ORDER BY cached_at DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [dict(r) for r in rows], total

    def get_all_trial_embeddings(self) -> list[tuple[str, bytes]]:
        rows = self.conn.execute(
            "SELECT nct_id, embedding FROM trials WHERE embedding IS NOT NULL"
        ).fetchall()
        return [(r["nct_id"], r["embedding"]) for r in rows]

    def count_trials(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM trials").fetchone()[0]

    # --- Patients ---

    def upsert_patient(self, patient_dict: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT INTO patients (patient_id, source_type, raw_input, features, created_at)
            VALUES (:patient_id, :source_type, :raw_input, :features, :created_at)
            ON CONFLICT(patient_id) DO UPDATE SET
                source_type=excluded.source_type,
                raw_input=excluded.raw_input,
                features=excluded.features
            """,
            patient_dict,
        )
        self.conn.commit()

    def get_patient(self, patient_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM patients WHERE patient_id=?", (patient_id,)).fetchone()
        return dict(row) if row else None

    # --- Match Results ---

    def insert_match_result(self, result_dict: dict[str, Any]) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO match_results
                (match_id, patient_id, nct_id, composite_score, confidence, explanation, status, created_at)
            VALUES
                (:match_id, :patient_id, :nct_id, :composite_score, :confidence, :explanation, :status, :created_at)
            """,
            result_dict,
        )
        self.conn.commit()

    def get_match_result(self, match_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM match_results WHERE match_id=?", (match_id,)).fetchone()
        return dict(row) if row else None

    # --- Sync Jobs ---

    def create_sync_job(self, job_id: str, condition: str) -> None:
        self.conn.execute(
            "INSERT INTO sync_jobs (job_id, condition, status, created_at) VALUES (?,?,?,?)",
            (job_id, condition, "queued", time.time()),
        )
        self.conn.commit()

    def update_sync_job(self, job_id: str, **fields: Any) -> None:
        set_clause = ", ".join(f"{k}=?" for k in fields)
        values = list(fields.values()) + [job_id]
        self.conn.execute(f"UPDATE sync_jobs SET {set_clause} WHERE job_id=?", values)
        self.conn.commit()

    def get_sync_job(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM sync_jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def list_sync_jobs(self, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM sync_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Patients (list) ---

    def list_patients(
        self,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        total = self.conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        rows = self.conn.execute(
            "SELECT * FROM patients ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows], total

    # --- Match Results (by patient) ---

    def list_match_results_by_patient(
        self,
        patient_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM match_results WHERE patient_id=? ORDER BY created_at DESC LIMIT ?",
            (patient_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Conditions autocomplete ---

    def get_all_conditions(self) -> list[str]:
        rows = self.conn.execute("SELECT conditions FROM trials WHERE conditions IS NOT NULL").fetchall()
        seen: set[str] = set()
        result: list[str] = []
        for row in rows:
            try:
                import json as _json
                conds = _json.loads(row["conditions"])
                for c in conds:
                    if isinstance(c, str) and c and c not in seen:
                        seen.add(c)
                        result.append(c)
            except Exception:
                pass
        return result

    # --- Diagnosis Equivalence Cache ---

    def get_diagnosis_equiv(self, cache_key: str) -> bool | None:
        row = self.conn.execute(
            "SELECT equivalent, cached_at, ttl_seconds FROM diagnosis_equiv_cache WHERE cache_key=?",
            (cache_key,),
        ).fetchone()
        if row and time.time() - row["cached_at"] < row["ttl_seconds"]:
            return bool(row["equivalent"])
        return None

    def set_diagnosis_equiv(self, cache_key: str, equivalent: bool, ttl_seconds: int = 604800) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO diagnosis_equiv_cache (cache_key, equivalent, cached_at, ttl_seconds)
            VALUES (?,?,?,?)
            """,
            (cache_key, int(equivalent), time.time(), ttl_seconds),
        )
        self.conn.commit()

    def serialize_json(self, value: Any) -> str:
        return json.dumps(value)

    def deserialize_json(self, value: str | None) -> Any:
        if value is None:
            return None
        return json.loads(value)
