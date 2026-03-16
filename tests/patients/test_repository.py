"""Tests for patient repository."""

from __future__ import annotations

import time

from clinicaltrial_match.patients.models import Gender, Patient, PatientFeatures
from clinicaltrial_match.patients.repository import PatientRepository


def _make_patient(patient_id: str = "p-001") -> Patient:
    features = PatientFeatures(
        patient_id=patient_id,
        age_years=50.0,
        gender=Gender.MALE,
        clinical_summary="50-year-old male with diabetes",
    )
    return Patient(
        patient_id=patient_id,
        source_type="fhir",
        raw_input="{}",
        features=features,
        created_at=time.time(),
    )


def test_save_and_get_patient(in_memory_db):
    repo = PatientRepository(in_memory_db)
    patient = _make_patient("p-001")
    repo.save(patient)
    retrieved = repo.get("p-001")
    assert retrieved is not None
    assert retrieved.patient_id == "p-001"


def test_get_nonexistent_returns_none(in_memory_db):
    repo = PatientRepository(in_memory_db)
    assert repo.get("does-not-exist") is None


def test_features_round_trip(in_memory_db):
    repo = PatientRepository(in_memory_db)
    patient = _make_patient("p-002")
    repo.save(patient)
    retrieved = repo.get("p-002")
    assert retrieved is not None
    assert retrieved.features is not None
    assert retrieved.features.age_years == 50.0
    assert retrieved.features.gender == Gender.MALE


def test_save_updates_existing(in_memory_db):
    repo = PatientRepository(in_memory_db)
    p1 = _make_patient("p-003")
    repo.save(p1)
    p2 = Patient(
        patient_id="p-003",
        source_type="note",
        raw_input="updated note",
        features=PatientFeatures(patient_id="p-003", age_years=55.0, clinical_summary="Updated"),
        created_at=time.time(),
    )
    repo.save(p2)
    retrieved = repo.get("p-003")
    assert retrieved is not None
    assert retrieved.source_type == "note"
    assert retrieved.features is not None
    assert retrieved.features.age_years == 55.0
