"""Patient repository: persistence layer for patient records."""
from __future__ import annotations

import json
import time

from clinicaltrial_match.infrastructure.db import Database
from clinicaltrial_match.patients.models import Patient, PatientFeatures


class PatientRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def save(self, patient: Patient) -> None:
        features_json = patient.features.model_dump_json() if patient.features else None
        self._db.upsert_patient({
            "patient_id": patient.patient_id,
            "source_type": patient.source_type,
            "raw_input": patient.raw_input,
            "features": features_json,
            "created_at": patient.created_at or time.time(),
        })

    def get(self, patient_id: str) -> Patient | None:
        row = self._db.get_patient(patient_id)
        if not row:
            return None
        features: PatientFeatures | None = None
        if row.get("features"):
            try:
                features = PatientFeatures.model_validate_json(row["features"])
            except Exception:
                features = None
        return Patient(
            patient_id=row["patient_id"],
            source_type=row["source_type"],  # type: ignore[arg-type]
            raw_input=row["raw_input"],
            features=features,
            created_at=row["created_at"],
        )
