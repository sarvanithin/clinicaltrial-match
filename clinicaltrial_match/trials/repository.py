"""Trial repository: persistence layer for clinical trials."""
from __future__ import annotations

import json
import time
from typing import Any

import numpy as np

from clinicaltrial_match.infrastructure.db import Database
from clinicaltrial_match.infrastructure.embeddings import EmbeddingIndex
from clinicaltrial_match.trials.models import EligibilityCriteria, Trial, TrialStatus


class TrialRepository:
    def __init__(self, db: Database, embeddings: EmbeddingIndex) -> None:
        self._db = db
        self._embeddings = embeddings

    def save(self, trial: Trial) -> None:
        criteria_json = (
            trial.eligibility_criteria.model_dump_json()
            if trial.eligibility_criteria
            else None
        )
        embedding_bytes: bytes | None = None
        if trial.embedding:
            embedding_bytes = np.array(trial.embedding, dtype=np.float32).tobytes()
            self._embeddings.add(trial.nct_id, np.array(trial.embedding, dtype=np.float32))

        self._db.upsert_trial({
            "nct_id": trial.nct_id,
            "title": trial.title,
            "brief_summary": trial.brief_summary,
            "conditions": json.dumps(trial.conditions),
            "interventions": json.dumps(trial.interventions),
            "phase": trial.phase,
            "status": trial.status.value,
            "eligibility_text": trial.eligibility_text,
            "eligibility_criteria": criteria_json,
            "sponsor": trial.sponsor,
            "locations": json.dumps(trial.locations),
            "start_date": trial.start_date.isoformat() if trial.start_date else None,
            "last_updated": trial.last_updated.isoformat() if trial.last_updated else None,
            "cached_at": trial.cached_at or time.time(),
            "embedding": embedding_bytes,
        })

    def save_raw(self, raw: dict[str, Any], criteria: EligibilityCriteria | None, embedding: np.ndarray | None) -> None:
        """Save a raw normalized dict from the fetcher directly."""
        criteria_json = criteria.model_dump_json() if criteria else None
        embedding_bytes: bytes | None = None
        if embedding is not None:
            embedding_bytes = embedding.tobytes()
            self._embeddings.add(raw["nct_id"], embedding)

        row = dict(raw)
        row["conditions"] = json.dumps(raw.get("conditions", []))
        row["interventions"] = json.dumps(raw.get("interventions", []))
        row["locations"] = json.dumps(raw.get("locations", []))
        row["eligibility_criteria"] = criteria_json
        row["embedding"] = embedding_bytes
        row.pop("_gender_hint", None)
        row.pop("_min_age_years", None)
        row.pop("_max_age_years", None)
        self._db.upsert_trial(row)

    def get(self, nct_id: str) -> Trial | None:
        row = self._db.get_trial(nct_id)
        if not row:
            return None
        return self._row_to_trial(row)

    def list(
        self,
        status: str | None = None,
        condition: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[Trial], int]:
        rows, total = self._db.list_trials(status=status, condition=condition, limit=limit, offset=offset)
        return [self._row_to_trial(r) for r in rows], total

    def count(self) -> int:
        return self._db.count_trials()

    def _row_to_trial(self, row: dict[str, Any]) -> Trial:
        criteria: EligibilityCriteria | None = None
        if row.get("eligibility_criteria"):
            try:
                criteria = EligibilityCriteria.model_validate_json(row["eligibility_criteria"])
            except Exception:
                criteria = None

        from datetime import date
        def _parse_date(v: str | None) -> date | None:
            if not v:
                return None
            try:
                return date.fromisoformat(v[:10])
            except Exception:
                return None

        return Trial(
            nct_id=row["nct_id"],
            title=row["title"],
            brief_summary=row.get("brief_summary", ""),
            conditions=json.loads(row.get("conditions", "[]")),
            interventions=json.loads(row.get("interventions", "[]")),
            phase=row.get("phase", ""),
            status=TrialStatus(row.get("status", "UNKNOWN")),
            eligibility_criteria=criteria,
            eligibility_text=row.get("eligibility_text", ""),
            sponsor=row.get("sponsor", ""),
            locations=json.loads(row.get("locations", "[]")),
            start_date=_parse_date(row.get("start_date")),
            last_updated=_parse_date(row.get("last_updated")),
            cached_at=row.get("cached_at", 0.0),
        )
