"""Tests for eligibility criteria parser."""
from __future__ import annotations

import pytest

from clinicaltrial_match.trials.parser import (
    EligibilityParser,
    _regex_parse_age,
    _regex_parse_gender,
    _split_inclusion_exclusion,
)


SAMPLE_CRITERIA_TEXT = (
    "Inclusion Criteria:\n"
    "- Adults 18 years or older\n"
    "- Type 2 diabetes mellitus diagnosis\n"
    "- HbA1c between 7 and 10 percent\n\n"
    "Exclusion Criteria:\n"
    "- Current insulin use\n"
    "- Renal impairment"
)


def test_split_inclusion_exclusion():
    incl, excl = _split_inclusion_exclusion(SAMPLE_CRITERIA_TEXT)
    assert "18 years or older" in incl
    assert "insulin" in excl


def test_regex_parse_age_minimum():
    age = _regex_parse_age("Adults 18 years or older")
    assert age is not None
    assert age.minimum_age_years == 18


def test_regex_parse_age_range():
    age = _regex_parse_age("between 18 and 65 years of age")
    assert age is not None
    assert age.minimum_age_years == 18
    assert age.maximum_age_years == 65


def test_regex_parse_age_none():
    age = _regex_parse_age("No specific age requirement")
    assert age is None


def test_regex_parse_gender_all():
    g = _regex_parse_gender("male or female patients")
    assert g is not None
    assert g.allowed == "ALL"


def test_regex_parse_gender_female_only():
    g = _regex_parse_gender("women only")
    assert g is not None
    assert g.allowed == "FEMALE"


def test_regex_parse_gender_no_match():
    g = _regex_parse_gender("No gender restriction mentioned")
    assert g is None


@pytest.mark.asyncio
async def test_parser_uses_regex_fallback_on_empty(mock_claude):
    """Empty text → regex-only result with no Claude call."""
    parser = EligibilityParser(mock_claude)
    criteria = await parser.parse("")
    assert criteria.parsed_by == "regex-fallback"
    mock_claude.tool_use.assert_not_called()


@pytest.mark.asyncio
async def test_parser_calls_claude_for_full_text(mock_claude):
    parser = EligibilityParser(mock_claude)
    criteria = await parser.parse(SAMPLE_CRITERIA_TEXT)
    assert criteria.parsed_by == "claude-haiku"
    assert criteria.parse_confidence == 0.9
    mock_claude.tool_use.assert_called_once()


@pytest.mark.asyncio
async def test_parser_falls_back_on_claude_error(mock_claude):
    mock_claude.tool_use.side_effect = RuntimeError("API error")
    parser = EligibilityParser(mock_claude)
    criteria = await parser.parse(SAMPLE_CRITERIA_TEXT)
    assert criteria.parsed_by == "regex-fallback"


@pytest.mark.asyncio
async def test_parser_extracts_diagnoses(mock_claude):
    parser = EligibilityParser(mock_claude)
    criteria = await parser.parse(SAMPLE_CRITERIA_TEXT)
    assert criteria.diagnoses is not None
    assert "Type 2 diabetes mellitus" in criteria.diagnoses.required_conditions


@pytest.mark.asyncio
async def test_parser_extracts_lab_constraints(mock_claude):
    parser = EligibilityParser(mock_claude)
    criteria = await parser.parse(SAMPLE_CRITERIA_TEXT)
    assert len(criteria.labs) == 1
    assert criteria.labs[0].test_name == "HbA1c"
    assert criteria.labs[0].operator == "between"
