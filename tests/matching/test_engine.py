"""Tests for matching engine."""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from clinicaltrial_match.matching.engine import MatchingEngine
from clinicaltrial_match.matching.models import MatchRequest
from clinicaltrial_match.matching.semantic import SemanticSearcher
from clinicaltrial_match.patients.models import Diagnosis, Gender, LabValue, Medication, PatientFeatures
from clinicaltrial_match.trials.models import Trial, TrialStatus
from clinicaltrial_match.trials.repository import TrialRepository
from tests.fixtures.sample_criteria import make_sample_criteria


def _make_patient_features(patient_id: str = "p-001") -> PatientFeatures:
    return PatientFeatures(
        patient_id=patient_id,
        age_years=50.0,
        gender=Gender.MALE,
        diagnoses=[Diagnosis(name="Type 2 diabetes mellitus", code="E11")],
        lab_values=[LabValue(test_name="HbA1c", value=8.2, unit="%")],
        medications=[Medication(name="Metformin", active=True)],
        clinical_summary="50-year-old male with type 2 diabetes on Metformin",
    )


@pytest.fixture
def populated_db_and_repo(in_memory_db, mock_embeddings):
    """DB with one trial already stored."""
    criteria = make_sample_criteria()
    repo = TrialRepository(in_memory_db, mock_embeddings)
    trial = Trial(
        nct_id="NCT99999999",
        title="Study of Metformin in Type 2 Diabetes",
        status=TrialStatus.RECRUITING,
        conditions=["Type 2 Diabetes Mellitus"],
        eligibility_criteria=criteria,
        cached_at=time.time(),
    )
    repo.save(trial)
    return in_memory_db, repo


@pytest.fixture
def engine_with_mock_search(populated_db_and_repo, mock_embeddings, mock_claude):
    db, repo = populated_db_and_repo
    # Pre-insert patient so FK constraint is satisfied
    from clinicaltrial_match.patients.repository import PatientRepository
    PatientRepository(db).save(_make_patient_record())
    # Make semantic searcher return the stored trial as top hit
    searcher = MagicMock(spec=SemanticSearcher)
    searcher.search.return_value = [("NCT99999999", 0.82)]
    return MatchingEngine(db, repo, searcher, mock_claude)


def _make_patient_record():
    from clinicaltrial_match.patients.models import Patient
    return Patient(
        patient_id="p-001",
        source_type="fhir",
        raw_input="{}",
        features=_make_patient_features(),
        created_at=time.time(),
    )


@pytest.mark.asyncio
async def test_match_returns_results(engine_with_mock_search):
    features = _make_patient_features()
    request = MatchRequest(patient_id="p-001", max_results=5)
    results = await engine_with_mock_search.match(features, request)
    assert len(results) > 0
    assert results[0].nct_id == "NCT99999999"


@pytest.mark.asyncio
async def test_match_result_has_explanation(engine_with_mock_search):
    features = _make_patient_features()
    request = MatchRequest(patient_id="p-001", max_results=5)
    results = await engine_with_mock_search.match(features, request)
    assert results[0].explanation is not None
    assert results[0].explanation.semantic_score == pytest.approx(0.82)


@pytest.mark.asyncio
async def test_match_filters_by_status(engine_with_mock_search, populated_db_and_repo, mock_claude):
    db, repo = populated_db_and_repo
    searcher = MagicMock(spec=SemanticSearcher)
    searcher.search.return_value = [("NCT99999999", 0.82)]
    engine = MatchingEngine(db, repo, searcher, mock_claude)
    features = _make_patient_features()
    request = MatchRequest(patient_id="p-001", trial_status_filter=["COMPLETED"])
    results = await engine.match(features, request)
    assert len(results) == 0  # trial is RECRUITING, not COMPLETED


@pytest.mark.asyncio
async def test_match_respects_min_score(engine_with_mock_search):
    features = _make_patient_features()
    request = MatchRequest(patient_id="p-001", min_score=0.99)
    results = await engine_with_mock_search.match(features, request)
    assert all(r.composite_score >= 0.99 for r in results)


@pytest.mark.asyncio
async def test_match_no_candidates(populated_db_and_repo, mock_embeddings, mock_claude):
    db, repo = populated_db_and_repo
    from clinicaltrial_match.patients.repository import PatientRepository
    PatientRepository(db).save(_make_patient_record())
    searcher = MagicMock(spec=SemanticSearcher)
    searcher.search.return_value = []
    engine = MatchingEngine(db, repo, searcher, mock_claude)
    features = _make_patient_features()
    request = MatchRequest(patient_id="p-001")
    results = await engine.match(features, request)
    assert results == []


@pytest.mark.asyncio
async def test_match_persists_result(engine_with_mock_search, in_memory_db):
    features = _make_patient_features()
    request = MatchRequest(patient_id="p-001")
    results = await engine_with_mock_search.match(features, request)
    if results:
        stored = in_memory_db.get_match_result(results[0].match_id)
        assert stored is not None
        assert stored["patient_id"] == "p-001"
