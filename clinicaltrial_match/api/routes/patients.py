"""Patients routes: POST /v1/patients/fhir, POST /v1/patients/note"""
from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from clinicaltrial_match.api.models import NoteIngestRequest, PatientResponse
from clinicaltrial_match.patients.fhir_parser import parse_fhir_bundle
from clinicaltrial_match.patients.note_extractor import NoteExtractor
from clinicaltrial_match.patients.repository import PatientRepository

router = APIRouter()
logger = structlog.get_logger()


@router.post("/fhir", response_model=PatientResponse)
async def ingest_fhir(request: Request, body: dict[str, Any]) -> PatientResponse:
    repo: PatientRepository = request.app.state.patient_repo
    warnings: list[str] = []

    try:
        patient = parse_fhir_bundle(body)
    except Exception as exc:
        logger.error("fhir_parse_error", error=str(exc))
        return JSONResponse(status_code=422, content={"detail": f"FHIR parse error: {exc}"})  # type: ignore[return-value]

    repo.save(patient)
    f = patient.features
    if not f:
        return JSONResponse(status_code=422, content={"detail": "No features extracted"})  # type: ignore[return-value]

    if not f.diagnoses:
        warnings.append("No diagnoses extracted from FHIR bundle")

    return PatientResponse(
        patient_id=patient.patient_id,
        source_type="fhir",
        extraction_method=f.extraction_method,
        extraction_confidence=f.extraction_confidence,
        clinical_summary=f.clinical_summary,
        age_years=f.age_years,
        gender=f.gender.value,
        diagnoses_count=len(f.diagnoses),
        medications_count=len(f.medications),
        lab_values_count=len(f.lab_values),
        warnings=warnings,
    )


@router.post("/note", response_model=PatientResponse)
async def ingest_note(request: Request, body: NoteIngestRequest) -> PatientResponse:
    repo: PatientRepository = request.app.state.patient_repo
    extractor: NoteExtractor = request.app.state.note_extractor

    if not body.note_text.strip():
        return JSONResponse(status_code=422, content={"detail": "note_text is empty"})  # type: ignore[return-value]

    patient = await extractor.extract(body.note_text, body.patient_id)
    repo.save(patient)
    f = patient.features
    if not f:
        return JSONResponse(status_code=422, content={"detail": "Extraction failed"})  # type: ignore[return-value]

    warnings: list[str] = []
    if f.extraction_confidence < 0.5:
        warnings.append("Low extraction confidence — review extracted features")

    return PatientResponse(
        patient_id=patient.patient_id,
        source_type="note",
        extraction_method=f.extraction_method,
        extraction_confidence=f.extraction_confidence,
        clinical_summary=f.clinical_summary,
        age_years=f.age_years,
        gender=f.gender.value,
        diagnoses_count=len(f.diagnoses),
        medications_count=len(f.medications),
        lab_values_count=len(f.lab_values),
        warnings=warnings,
    )
