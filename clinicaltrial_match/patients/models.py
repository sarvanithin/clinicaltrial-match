"""Domain models for patients."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class LabValue(BaseModel):
    test_name: str
    value: float
    unit: str = ""
    reference_range_low: float | None = None
    reference_range_high: float | None = None
    collected_date: date | None = None


class Medication(BaseModel):
    name: str
    dose: str = ""
    frequency: str = ""
    active: bool = True


class Procedure(BaseModel):
    name: str
    code: str = ""
    performed_date: date | None = None


class Diagnosis(BaseModel):
    name: str
    code: str = ""
    code_system: Literal["ICD-10", "SNOMED", "other"] = "ICD-10"
    onset_date: date | None = None
    status: Literal["active", "resolved", "unknown"] = "active"


class PatientFeatures(BaseModel):
    patient_id: str
    age_years: float | None = None
    gender: Gender = Gender.UNKNOWN
    diagnoses: list[Diagnosis] = Field(default_factory=list)
    lab_values: list[LabValue] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    clinical_summary: str = ""
    extraction_method: Literal["fhir", "nlp", "hybrid"] = "fhir"
    extraction_confidence: float = 1.0


class Patient(BaseModel):
    patient_id: str
    source_type: Literal["fhir", "note"]
    raw_input: str
    features: PatientFeatures | None = None
    created_at: float = 0.0
