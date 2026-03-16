"""Domain models for matching results."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConstraintResult(BaseModel):
    criterion: str
    satisfied: bool
    patient_value: str
    reason: str


class MatchExplanation(BaseModel):
    semantic_score: float
    constraint_score: float
    composite_score: float
    passed_constraints: list[ConstraintResult] = Field(default_factory=list)
    failed_constraints: list[ConstraintResult] = Field(default_factory=list)
    top_matching_factors: list[str] = Field(default_factory=list)
    disqualifying_factors: list[str] = Field(default_factory=list)


class MatchResult(BaseModel):
    match_id: str
    patient_id: str
    nct_id: str
    trial_title: str
    composite_score: float
    confidence: Literal["high", "medium", "low"]
    explanation: MatchExplanation
    status: Literal["eligible", "likely_eligible", "potentially_eligible", "ineligible"]
    created_at: float = 0.0


class MatchRequest(BaseModel):
    patient_id: str
    max_results: int = 10
    min_score: float = 0.3
    trial_status_filter: list[str] = Field(default_factory=lambda: ["RECRUITING"])
    condition_filter: list[str] = Field(default_factory=list)
