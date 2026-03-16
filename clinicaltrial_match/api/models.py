"""Pydantic v2 request/response schemas for the API layer."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrialSummary(BaseModel):
    nct_id: str
    title: str
    brief_summary: str
    conditions: list[str]
    phase: str
    status: str
    sponsor: str
    locations: list[str]


class PaginatedTrialsResponse(BaseModel):
    trials: list[TrialSummary]
    total: int
    limit: int
    offset: int


class SyncRequest(BaseModel):
    condition: str
    max_trials: int = 100
    force_reparse: bool = False
    status_filter: list[str] = Field(default_factory=lambda: ["RECRUITING"])


class SyncResponse(BaseModel):
    job_id: str
    status: str
    condition: str


class SyncStatusResponse(BaseModel):
    job_id: str
    status: str
    condition: str
    trials_fetched: int
    trials_parsed: int
    error: str | None


class SyncJobListResponse(BaseModel):
    jobs: list[SyncStatusResponse]


class PatientResponse(BaseModel):
    patient_id: str
    source_type: str
    extraction_method: str
    extraction_confidence: float
    clinical_summary: str
    age_years: float | None
    gender: str
    diagnoses_count: int
    medications_count: int
    lab_values_count: int
    warnings: list[str] = Field(default_factory=list)


class PatientSummary(BaseModel):
    patient_id: str
    source_type: str
    extraction_method: str
    extraction_confidence: float
    clinical_summary: str
    age_years: float | None
    gender: str
    created_at: float


class PatientListResponse(BaseModel):
    patients: list[PatientSummary]
    total: int
    limit: int
    offset: int


class NoteIngestRequest(BaseModel):
    note_text: str
    patient_id: str | None = None


class MatchResponse(BaseModel):
    patient_id: str
    matches: list[Any]  # list[MatchResult] — avoid circular import
    total_evaluated: int
    processing_time_ms: float


class BatchMatchRequest(BaseModel):
    patient_ids: list[str]
    max_results: int = 5
    min_score: float = 0.3


class BatchMatchResponse(BaseModel):
    results: dict[str, list[Any]]
    total_patients: int
    processing_time_ms: float


class CompareRequest(BaseModel):
    nct_ids: list[str]
    patient_id: str | None = None


class CompareResponse(BaseModel):
    trials: list[dict[str, Any]]


class AutocompleteResponse(BaseModel):
    suggestions: list[str]


class HealthResponse(BaseModel):
    status: str
    dependencies: dict[str, bool]
    trials_cached: int
    version: str
    uptime_seconds: float
