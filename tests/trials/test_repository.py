"""Tests for the trial repository."""

from __future__ import annotations

import time

from clinicaltrial_match.trials.models import Trial, TrialStatus
from clinicaltrial_match.trials.repository import TrialRepository


def test_save_and_get_trial(in_memory_db, mock_embeddings, sample_criteria):
    repo = TrialRepository(in_memory_db, mock_embeddings)
    trial = Trial(
        nct_id="NCT99999999",
        title="Test Trial",
        brief_summary="A test trial",
        conditions=["Type 2 Diabetes"],
        status=TrialStatus.RECRUITING,
        eligibility_criteria=sample_criteria,
        cached_at=time.time(),
    )
    repo.save(trial)
    retrieved = repo.get("NCT99999999")
    assert retrieved is not None
    assert retrieved.nct_id == "NCT99999999"
    assert retrieved.title == "Test Trial"
    assert retrieved.status == TrialStatus.RECRUITING


def test_get_nonexistent_returns_none(in_memory_db, mock_embeddings):
    repo = TrialRepository(in_memory_db, mock_embeddings)
    assert repo.get("NCT00000000") is None


def test_list_trials_with_status_filter(in_memory_db, mock_embeddings):
    repo = TrialRepository(in_memory_db, mock_embeddings)
    for i in range(3):
        repo.save(
            Trial(
                nct_id=f"NCT0000000{i}",
                title=f"Trial {i}",
                status=TrialStatus.RECRUITING,
                cached_at=time.time(),
            )
        )
    repo.save(
        Trial(
            nct_id="NCT00000099",
            title="Completed Trial",
            status=TrialStatus.COMPLETED,
            cached_at=time.time(),
        )
    )
    trials, total = repo.list(status="RECRUITING")
    assert total == 3
    assert all(t.status == TrialStatus.RECRUITING for t in trials)


def test_count_trials(in_memory_db, mock_embeddings):
    repo = TrialRepository(in_memory_db, mock_embeddings)
    assert repo.count() == 0
    repo.save(Trial(nct_id="NCT00000001", title="T1", cached_at=time.time()))
    assert repo.count() == 1


def test_eligibility_criteria_round_trips(in_memory_db, mock_embeddings, sample_criteria):
    repo = TrialRepository(in_memory_db, mock_embeddings)
    repo.save(
        Trial(
            nct_id="NCT00000001",
            title="Trial",
            eligibility_criteria=sample_criteria,
            cached_at=time.time(),
        )
    )
    retrieved = repo.get("NCT00000001")
    assert retrieved is not None
    assert retrieved.eligibility_criteria is not None
    assert retrieved.eligibility_criteria.parse_confidence == 0.9
    assert retrieved.eligibility_criteria.age is not None
    assert retrieved.eligibility_criteria.age.minimum_age_years == 18
