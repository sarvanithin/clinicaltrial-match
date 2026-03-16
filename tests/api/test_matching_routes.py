"""Tests for matching API routes."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from clinicaltrial_match.api.app import create_app
from clinicaltrial_match.matching.models import MatchExplanation, MatchResult
from clinicaltrial_match.patients.models import Gender, Patient, PatientFeatures


def _make_app_with_mocks(patient: Patient, match_results: list[MatchResult]):
    app = create_app.__wrapped__() if hasattr(create_app, "__wrapped__") else None
    # Build app directly without lifespan
    from fastapi import FastAPI

    from clinicaltrial_match.api.routes.health import router as health_router
    from clinicaltrial_match.api.routes.matching import router as matching_router

    test_app = FastAPI()
    test_app.include_router(matching_router, prefix="/v1/match")
    test_app.include_router(health_router, prefix="/v1")

    patient_repo = MagicMock()
    patient_repo.get.return_value = patient

    engine = MagicMock()
    engine.match = AsyncMock(return_value=match_results)

    db = MagicMock()

    test_app.state.patient_repo = patient_repo
    test_app.state.matching_engine = engine
    test_app.state.db = db

    return test_app


def _make_patient() -> Patient:
    return Patient(
        patient_id="p-001",
        source_type="fhir",
        raw_input="{}",
        features=PatientFeatures(
            patient_id="p-001",
            age_years=50.0,
            gender=Gender.MALE,
            clinical_summary="50-year-old male",
        ),
        created_at=time.time(),
    )


def _make_match_result() -> MatchResult:
    return MatchResult(
        match_id="m-001",
        patient_id="p-001",
        nct_id="NCT99999999",
        trial_title="Test Trial",
        composite_score=0.85,
        confidence="high",
        explanation=MatchExplanation(
            semantic_score=0.9,
            constraint_score=0.8,
            composite_score=0.85,
        ),
        status="eligible",
        created_at=time.time(),
    )


class TestMatchRoute:
    def test_match_returns_200(self):
        patient = _make_patient()
        result = _make_match_result()
        app = _make_app_with_mocks(patient, [result])
        client = TestClient(app)
        resp = client.post("/v1/match", json={"patient_id": "p-001"})
        assert resp.status_code == 200

    def test_match_returns_results(self):
        patient = _make_patient()
        result = _make_match_result()
        app = _make_app_with_mocks(patient, [result])
        client = TestClient(app)
        resp = client.post("/v1/match", json={"patient_id": "p-001"})
        data = resp.json()
        assert len(data["matches"]) == 1
        assert data["matches"][0]["nct_id"] == "NCT99999999"

    def test_match_patient_not_found(self):
        app = _make_app_with_mocks.__wrapped__ if hasattr(_make_app_with_mocks, "__wrapped__") else None
        from fastapi import FastAPI

        from clinicaltrial_match.api.routes.matching import router as matching_router

        test_app = FastAPI()
        test_app.include_router(matching_router, prefix="/v1/match")
        patient_repo = MagicMock()
        patient_repo.get.return_value = None
        engine = MagicMock()
        db = MagicMock()
        test_app.state.patient_repo = patient_repo
        test_app.state.matching_engine = engine
        test_app.state.db = db
        client = TestClient(test_app, raise_server_exceptions=False)
        resp = client.post("/v1/match", json={"patient_id": "nonexistent"})
        assert resp.status_code == 404
