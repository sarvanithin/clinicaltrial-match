"""Tests for the match ranker — pure function, no mocks."""

from __future__ import annotations

import time

from clinicaltrial_match.matching.models import ConstraintResult
from clinicaltrial_match.matching.ranker import build_match_result


def _passed(criterion: str) -> ConstraintResult:
    return ConstraintResult(criterion=criterion, satisfied=True, patient_value="val", reason="PASS")


def _failed(criterion: str) -> ConstraintResult:
    return ConstraintResult(criterion=criterion, satisfied=False, patient_value="val", reason="FAIL")


def test_high_confidence_with_all_passing():
    constraints = [_passed("age >= 18"), _passed("has diagnosis: T2DM")]
    result = build_match_result(
        match_id="m1",
        patient_id="p1",
        nct_id="NCT1",
        trial_title="T",
        semantic_score=0.9,
        constraint_results=constraints,
        created_at=time.time(),
    )
    assert result.confidence == "high"
    assert result.status == "eligible"
    assert result.composite_score >= 0.8


def test_medium_confidence_with_some_failures():
    constraints = [_passed("age >= 18"), _failed("has diagnosis: Cancer")]
    result = build_match_result(
        match_id="m2",
        patient_id="p1",
        nct_id="NCT2",
        trial_title="T",
        semantic_score=0.7,
        constraint_results=constraints,
        created_at=time.time(),
    )
    assert result.confidence in ("medium", "low")


def test_low_confidence_with_hard_failure():
    constraints = [_failed("age >= 65"), _failed("no diagnosis: T2DM")]
    result = build_match_result(
        match_id="m3",
        patient_id="p1",
        nct_id="NCT3",
        trial_title="T",
        semantic_score=0.5,
        constraint_results=constraints,
        created_at=time.time(),
    )
    # Two hard failures + only 0.5 semantic → score below high threshold
    assert result.status in ("likely_eligible", "potentially_eligible", "ineligible")
    assert len(result.explanation.disqualifying_factors) > 0


def test_no_constraints_gives_neutral_constraint_score():
    result = build_match_result(
        match_id="m4",
        patient_id="p1",
        nct_id="NCT4",
        trial_title="T",
        semantic_score=0.6,
        constraint_results=[],
        created_at=time.time(),
    )
    # constraint_score = 0.5 (neutral), composite = 0.4*0.6 + 0.6*0.5 = 0.54
    assert result.composite_score == pytest.approx(0.54, rel=0.01)


def test_composite_score_formula():
    # All passing constraints → constraint_score=1.0
    constraints = [_passed("age >= 18"), _passed("has diagnosis: T2DM")]
    result = build_match_result(
        match_id="m5",
        patient_id="p1",
        nct_id="NCT5",
        trial_title="T",
        semantic_score=0.8,
        constraint_results=constraints,
        created_at=time.time(),
    )
    expected = 0.40 * 0.8 + 0.60 * 1.0
    assert result.composite_score == pytest.approx(expected, rel=0.01)


def test_top_factors_populated():
    constraints = [_passed("age >= 18"), _passed("has diagnosis: T2DM"), _passed("HbA1c <= 10")]
    result = build_match_result(
        match_id="m6",
        patient_id="p1",
        nct_id="NCT6",
        trial_title="T",
        semantic_score=0.9,
        constraint_results=constraints,
        created_at=time.time(),
    )
    assert len(result.explanation.top_matching_factors) >= 1


import pytest  # noqa: E402
