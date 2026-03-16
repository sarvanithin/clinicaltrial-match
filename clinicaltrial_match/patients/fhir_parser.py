"""
FHIR R4 resource parser using manual Pydantic models.

Handles FHIR Bundle and individual resource types:
- Patient → age, gender
- Condition → diagnoses
- Observation → lab values
- MedicationStatement / MedicationRequest → medications

No fhirclient dependency — all parsing via Pydantic with extra="ignore".
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from clinicaltrial_match.patients.models import (
    Diagnosis,
    Gender,
    LabValue,
    Medication,
    Patient,
    PatientFeatures,
)


class _FHIRBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _FHIRCoding(_FHIRBase):
    system: str = ""
    code: str = ""
    display: str = ""


class _FHIRCodeableConcept(_FHIRBase):
    coding: list[_FHIRCoding] = Field(default_factory=list)
    text: str = ""


class _FHIRPatient(_FHIRBase):
    resourceType: str = "Patient"
    id: str = ""
    birthDate: str = ""
    gender: str = "unknown"


class _FHIRCondition(_FHIRBase):
    resourceType: str = "Condition"
    code: _FHIRCodeableConcept | None = None
    clinicalStatus: _FHIRCodeableConcept | None = None
    onsetDateTime: str = ""
    subject: dict[str, Any] = Field(default_factory=dict)


class _FHIRQuantity(_FHIRBase):
    value: float | None = None
    unit: str = ""


class _FHIRObservation(_FHIRBase):
    resourceType: str = "Observation"
    code: _FHIRCodeableConcept | None = None
    valueQuantity: _FHIRQuantity | None = None
    effectiveDateTime: str = ""
    referenceRange: list[dict[str, Any]] = Field(default_factory=list)


class _FHIRMedicationStatement(_FHIRBase):
    resourceType: str = "MedicationStatement"
    medicationCodeableConcept: _FHIRCodeableConcept | None = None
    medicationReference: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    dosage: list[dict[str, Any]] = Field(default_factory=list)


class _FHIRBundleEntry(_FHIRBase):
    resource: dict[str, Any] = Field(default_factory=dict)


class _FHIRBundle(_FHIRBase):
    resourceType: str = "Bundle"
    entry: list[_FHIRBundleEntry] = Field(default_factory=list)


def _calc_age(birth_date_str: str) -> float | None:
    if not birth_date_str:
        return None
    try:
        bd = date.fromisoformat(birth_date_str[:10])
        today = date.today()
        return float((today - bd).days) / 365.25
    except Exception:
        return None


def _gender_from_fhir(raw: str) -> Gender:
    mapping = {
        "male": Gender.MALE,
        "female": Gender.FEMALE,
        "other": Gender.OTHER,
        "unknown": Gender.UNKNOWN,
    }
    return mapping.get(raw.lower(), Gender.UNKNOWN)


def _concept_name(concept: _FHIRCodeableConcept | None) -> str:
    if not concept:
        return ""
    if concept.text:
        return concept.text
    for coding in concept.coding:
        if coding.display:
            return coding.display
    return ""


def _concept_code(concept: _FHIRCodeableConcept | None) -> tuple[str, str]:
    if not concept:
        return "", "other"
    for coding in concept.coding:
        system = coding.system or ""
        if "icd" in system.lower():
            return coding.code, "ICD-10"
        if "snomed" in system.lower():
            return coding.code, "SNOMED"
    return "", "other"


def parse_fhir_bundle(raw_json: str | dict[str, Any]) -> Patient:
    if isinstance(raw_json, str):
        data = json.loads(raw_json)
    else:
        data = raw_json

    patient_id: str | None = None
    age: float | None = None
    gender = Gender.UNKNOWN
    diagnoses: list[Diagnosis] = []
    lab_values: list[LabValue] = []
    medications: list[Medication] = []

    resource_type = data.get("resourceType", "")
    resources: list[dict[str, Any]] = []

    if resource_type == "Bundle":
        bundle = _FHIRBundle.model_validate(data)
        resources = [e.resource for e in bundle.entry if e.resource]
    else:
        resources = [data]

    for resource in resources:
        rtype = resource.get("resourceType", "")
        if rtype == "Patient":
            p = _FHIRPatient.model_validate(resource)
            patient_id = p.id or patient_id
            age = _calc_age(p.birthDate)
            gender = _gender_from_fhir(p.gender)

        elif rtype == "Condition":
            c = _FHIRCondition.model_validate(resource)
            name = _concept_name(c.code)
            code, code_system = _concept_code(c.code)
            clin_status = _concept_name(c.clinicalStatus)
            status_val: Any = (
                "active" if "active" in clin_status.lower() else ("resolved" if "resolved" in clin_status.lower() else "unknown")
            )
            onset: date | None = None
            if c.onsetDateTime:
                try:
                    onset = date.fromisoformat(c.onsetDateTime[:10])
                except Exception:
                    pass
            if name:
                diagnoses.append(
                    Diagnosis(
                        name=name,
                        code=code,
                        code_system=code_system,  # type: ignore[arg-type]
                        onset_date=onset,
                        status=status_val,
                    )
                )

        elif rtype == "Observation":
            obs = _FHIRObservation.model_validate(resource)
            name = _concept_name(obs.code)
            if obs.valueQuantity and obs.valueQuantity.value is not None and name:
                ref = obs.referenceRange[0] if obs.referenceRange else {}
                lab_values.append(
                    LabValue(
                        test_name=name,
                        value=obs.valueQuantity.value,
                        unit=obs.valueQuantity.unit,
                        reference_range_low=ref.get("low", {}).get("value") if ref else None,
                        reference_range_high=ref.get("high", {}).get("value") if ref else None,
                        collected_date=date.fromisoformat(obs.effectiveDateTime[:10]) if obs.effectiveDateTime else None,
                    )
                )

        elif rtype in ("MedicationStatement", "MedicationRequest"):
            med = _FHIRMedicationStatement.model_validate(resource)
            name = _concept_name(med.medicationCodeableConcept)
            if name:
                active = med.status in ("active", "intended", "taking")
                medications.append(Medication(name=name, active=active))

    if not patient_id:
        patient_id = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    active_diagnoses = [d for d in diagnoses if d.status != "resolved"]
    summary_parts = []
    if age is not None:
        summary_parts.append(f"{int(age)}-year-old {gender.value}")
    if active_diagnoses:
        summary_parts.append("with " + ", ".join(d.name for d in active_diagnoses[:5]))
    if medications:
        active_meds = [m.name for m in medications if m.active]
        if active_meds:
            summary_parts.append("on " + ", ".join(active_meds[:5]))
    clinical_summary = " ".join(summary_parts) if summary_parts else "Patient record from FHIR"

    features = PatientFeatures(
        patient_id=patient_id,
        age_years=age,
        gender=gender,
        diagnoses=diagnoses,
        lab_values=lab_values,
        medications=medications,
        clinical_summary=clinical_summary,
        extraction_method="fhir",
        extraction_confidence=1.0,
    )
    return Patient(
        patient_id=patient_id,
        source_type="fhir",
        raw_input=json.dumps(data) if isinstance(data, dict) else str(data),
        features=features,
        created_at=time.time(),
    )
