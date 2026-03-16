"""Patients routes: GET /v1/patients, POST /v1/patients/fhir, POST /v1/patients/note"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from clinicaltrial_match.api.models import (
    NoteIngestRequest,
    PatientListResponse,
    PatientResponse,
    PatientSummary,
)
from clinicaltrial_match.patients.fhir_parser import parse_fhir_bundle
from clinicaltrial_match.patients.note_extractor import NoteExtractor
from clinicaltrial_match.patients.repository import PatientRepository

router = APIRouter()
logger = structlog.get_logger()


@router.get("", response_model=PatientListResponse)
async def list_patients(
    request: Request,
    limit: int = 20,
    offset: int = 0,
) -> PatientListResponse:
    if limit < 1 or limit > 100:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "limit must be between 1 and 100", "field": "limit"},
        )
    try:
        db = request.app.state.db
        rows, total = db.list_patients(limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list_patients_db_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error fetching patients"},
        )

    patients: list[PatientSummary] = []
    for row in rows:
        features_data: dict = {}
        if row.get("features"):
            try:
                features_data = json.loads(row["features"])
            except Exception:
                pass
        patients.append(
            PatientSummary(
                patient_id=row["patient_id"],
                source_type=row["source_type"],
                extraction_method=features_data.get("extraction_method", ""),
                extraction_confidence=features_data.get("extraction_confidence", 0.0),
                clinical_summary=features_data.get("clinical_summary", ""),
                age_years=features_data.get("age_years"),
                gender=features_data.get("gender", "UNKNOWN"),
                created_at=row["created_at"],
            )
        )

    return PatientListResponse(
        patients=patients,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("/fhir", response_model=PatientResponse)
async def ingest_fhir(request: Request, body: dict[str, Any]) -> PatientResponse:
    repo: PatientRepository = request.app.state.patient_repo
    warnings: list[str] = []

    try:
        patient = parse_fhir_bundle(body)
    except Exception as exc:
        logger.error("fhir_parse_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": f"FHIR parse error: {exc}"},
        )

    try:
        repo.save(patient)
    except Exception as exc:
        logger.error("fhir_save_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error saving patient"},
        )

    f = patient.features
    if not f:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "No features extracted"},
        )

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
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "note_text is empty", "field": "note_text"},
        )
    if body.patient_id and len(body.patient_id) > 128:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "patient_id must be at most 128 characters", "field": "patient_id"},
        )

    try:
        patient = await extractor.extract(body.note_text, body.patient_id)
    except Exception as exc:
        logger.error("note_extract_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error during note extraction"},
        )

    try:
        repo.save(patient)
    except Exception as exc:
        logger.error("note_save_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error saving patient"},
        )

    f = patient.features
    if not f:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "Extraction failed"},
        )

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
