"""Tests for trial fetcher normalization."""
from __future__ import annotations

from clinicaltrial_match.trials.fetcher import _parse_age_string, normalize_study
from tests.fixtures.sample_trials import SAMPLE_STUDY_RAW


def test_parse_age_string_years():
    assert _parse_age_string("18 Years") == 18.0


def test_parse_age_string_months():
    assert _parse_age_string("6 Months") == pytest.approx(0.5, rel=0.01)


def test_parse_age_string_none():
    assert _parse_age_string(None) is None
    assert _parse_age_string("N/A") is None


def test_normalize_study_basic_fields():
    result = normalize_study(SAMPLE_STUDY_RAW)
    assert result["nct_id"] == "NCT99999999"
    assert result["title"] == "Study of Metformin in Type 2 Diabetes"
    assert result["status"] == "RECRUITING"
    assert "Type 2 Diabetes Mellitus" in result["conditions"]


def test_normalize_study_interventions():
    result = normalize_study(SAMPLE_STUDY_RAW)
    assert "Metformin" in result["interventions"]


def test_normalize_study_age_hints():
    result = normalize_study(SAMPLE_STUDY_RAW)
    assert result["_min_age_years"] == 18.0
    assert result["_max_age_years"] == 75.0


def test_normalize_study_gender_hint():
    result = normalize_study(SAMPLE_STUDY_RAW)
    assert result["_gender_hint"] == "ALL"


def test_normalize_study_missing_fields():
    minimal = {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}}
    result = normalize_study(minimal)
    assert result["nct_id"] == "NCT00000001"
    assert result["title"] == ""
    assert result["conditions"] == []


import pytest  # noqa: E402 (needed for approx above)
