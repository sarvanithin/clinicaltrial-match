"""Matching routes: POST /v1/match, GET /v1/match/{match_id}"""
from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from clinicaltrial_match.api.models import MatchResponse
from clinicaltrial_match.matching.engine import MatchingEngine
from clinicaltrial_match.matching.models import MatchRequest, MatchResult
from clinicaltrial_match.patients.repository import PatientRepository

router = APIRouter()
logger = structlog.get_logger()


@router.post("", response_model=MatchResponse)
async def match_patient(request: Request, body: MatchRequest) -> MatchResponse:
    patient_repo: PatientRepository = request.app.state.patient_repo
    engine: MatchingEngine = request.app.state.matching_engine
    db = request.app.state.db

    patient = patient_repo.get(body.patient_id)
    if not patient:
        return JSONResponse(status_code=404, content={"detail": f"Patient {body.patient_id!r} not found"})  # type: ignore[return-value]
    if not patient.features:
        return JSONResponse(status_code=422, content={"detail": "Patient has no extracted features"})  # type: ignore[return-value]

    start = time.time()
    results = await engine.match(patient.features, body)
    elapsed_ms = (time.time() - start) * 1000

    logger.info(
        "match_completed",
        patient_id=body.patient_id,
        results=len(results),
        elapsed_ms=round(elapsed_ms, 1),
    )

    return MatchResponse(
        patient_id=body.patient_id,
        matches=[r.model_dump() for r in results],
        total_evaluated=len(results),
        processing_time_ms=round(elapsed_ms, 1),
    )


@router.get("/{match_id}", response_model=MatchResult)
async def get_match(request: Request, match_id: str) -> MatchResult:
    db = request.app.state.db
    row = db.get_match_result(match_id)
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Match result not found"})  # type: ignore[return-value]

    import json
    from clinicaltrial_match.matching.models import MatchExplanation

    explanation = MatchExplanation.model_validate_json(row["explanation"])
    return MatchResult(
        match_id=row["match_id"],
        patient_id=row["patient_id"],
        nct_id=row["nct_id"],
        trial_title=row.get("trial_title", ""),
        composite_score=row["composite_score"],
        confidence=row["confidence"],  # type: ignore[arg-type]
        explanation=explanation,
        status=row["status"],  # type: ignore[arg-type]
        created_at=row["created_at"],
    )
