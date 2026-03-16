"""Domain models for clinical trials."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrialStatus(str, Enum):
    RECRUITING = "RECRUITING"
    NOT_YET_RECRUITING = "NOT_YET_RECRUITING"
    ACTIVE_NOT_RECRUITING = "ACTIVE_NOT_RECRUITING"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"
    WITHDRAWN = "WITHDRAWN"
    SUSPENDED = "SUSPENDED"
    UNKNOWN = "UNKNOWN"


class AgeConstraint(BaseModel):
    minimum_age_years: float | None = None
    maximum_age_years: float | None = None


class GenderConstraint(BaseModel):
    allowed: Literal["ALL", "MALE", "FEMALE"] = "ALL"


class DiagnosisConstraint(BaseModel):
    required_conditions: list[str] = Field(default_factory=list)
    excluded_conditions: list[str] = Field(default_factory=list)


class LabConstraint(BaseModel):
    test_name: str
    operator: Literal["<", "<=", ">", ">=", "=", "between"]
    value: float
    value_upper: float | None = None
    unit: str = ""


class MedicationConstraint(BaseModel):
    required: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)


class EligibilityCriteria(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw_inclusion_text: str = ""
    raw_exclusion_text: str = ""
    age: AgeConstraint | None = None
    gender: GenderConstraint | None = None
    diagnoses: DiagnosisConstraint | None = None
    labs: list[LabConstraint] = Field(default_factory=list)
    medications: MedicationConstraint | None = None
    other_inclusion: list[str] = Field(default_factory=list)
    other_exclusion: list[str] = Field(default_factory=list)
    parse_confidence: float = 0.0
    parsed_by: Literal["claude-haiku", "regex-fallback"] = "regex-fallback"


class Trial(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    nct_id: str
    title: str
    brief_summary: str = ""
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    phase: str = ""
    status: TrialStatus = TrialStatus.UNKNOWN
    eligibility_criteria: EligibilityCriteria | None = None
    eligibility_text: str = ""
    sponsor: str = ""
    locations: list[str] = Field(default_factory=list)
    start_date: date | None = None
    primary_completion_date: date | None = None
    last_updated: date | None = None
    embedding: list[float] | None = None
    cached_at: float = 0.0
