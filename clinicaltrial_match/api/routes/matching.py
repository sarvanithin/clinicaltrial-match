"""Matching routes: POST /v1/match, GET /v1/match/{match_id}, history, batch."""

from __future__ import annotations

import asyncio
import time

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from clinicaltrial_match.api.models import (
    BatchMatchRequest,
    BatchMatchResponse,
    LiveMatchRequest,
    LiveMatchResponse,
    MatchResponse,
)
from clinicaltrial_match.matching.engine import MatchingEngine
from clinicaltrial_match.matching.models import MatchExplanation, MatchRequest, MatchResult
from clinicaltrial_match.patients.repository import PatientRepository

router = APIRouter()
logger = structlog.get_logger()


@router.post("/live", response_model=LiveMatchResponse)
async def live_match(request: Request, body: LiveMatchRequest) -> LiveMatchResponse:
    # MPP payment gate — only active when CTM_MPP__ENABLED=true + recipient set
    mpp = getattr(request.app.state, "mpp", None)
    if mpp is not None:
        from mpp import Challenge

        from clinicaltrial_match.config import get_config

        price = get_config().mpp.price_per_query
        result = await mpp.charge(
            authorization=request.headers.get("Authorization"),
            amount=price,
        )
        if isinstance(result, Challenge):
            return JSONResponse(  # type: ignore[return-value]
                status_code=402,
                content={"error": "Payment required", "price_usd": price},
                headers={"WWW-Authenticate": result.to_www_authenticate(mpp.realm)},
            )
    """Ephemeral match — patient data is never written to disk or database."""
    if body.source not in ("fhir", "note"):
        return JSONResponse(status_code=422, content={"detail": "source must be 'fhir' or 'note'"})  # type: ignore[return-value]
    if body.max_results < 1 or body.max_results > 50:
        return JSONResponse(status_code=422, content={"detail": "max_results must be between 1 and 50"})  # type: ignore[return-value]
    if body.min_score < 0.0 or body.min_score > 1.0:
        return JSONResponse(status_code=422, content={"detail": "min_score must be between 0.0 and 1.0"})  # type: ignore[return-value]

    engine: MatchingEngine = request.app.state.matching_engine
    start = time.time()

    # Extract features in-memory — no DB write
    try:
        if body.source == "fhir":
            if not body.fhir_data:
                return JSONResponse(status_code=422, content={"detail": "fhir_data is required when source='fhir'"})  # type: ignore[return-value]
            from clinicaltrial_match.patients.fhir_parser import parse_fhir_bundle

            patient = parse_fhir_bundle(body.fhir_data)
        else:
            if not body.note_text or not body.note_text.strip():
                return JSONResponse(status_code=422, content={"detail": "note_text is required when source='note'"})  # type: ignore[return-value]
            extractor = request.app.state.note_extractor
            patient = await extractor.extract(body.note_text, patient_id="_ephemeral")
    except Exception as exc:
        logger.error("live_match_extract_error", source=body.source, error=str(exc))
        return JSONResponse(status_code=422, content={"detail": f"Feature extraction failed: {exc}"})  # type: ignore[return-value]

    if not patient.features:
        return JSONResponse(status_code=422, content={"detail": "Could not extract patient features"})  # type: ignore[return-value]

    f = patient.features

    match_req = MatchRequest(
        patient_id="_ephemeral",
        max_results=body.max_results,
        min_score=body.min_score,
        trial_status_filter=body.trial_status_filter,
    )

    try:
        # persist=False — nothing written to DB
        results = await engine.match(f, match_req, persist=False)
    except Exception as exc:
        logger.error("live_match_engine_error", error=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Matching failed — please try again"})  # type: ignore[return-value]

    elapsed_ms = (time.time() - start) * 1000
    label = body.patient_label or f"{int(f.age_years or 0)}y {f.gender.value}"

    logger.info("live_match_completed", results=len(results), elapsed_ms=round(elapsed_ms, 1))

    return LiveMatchResponse(
        patient_label=label,
        age_years=f.age_years,
        gender=f.gender.value,
        diagnoses_count=len(f.diagnoses),
        medications_count=len(f.medications),
        lab_values_count=len(f.lab_values),
        clinical_summary=f.clinical_summary,
        extraction_confidence=f.extraction_confidence,
        matches=[r.model_dump() for r in results],
        total_evaluated=len(results),
        processing_time_ms=round(elapsed_ms, 1),
    )


@router.post("", response_model=MatchResponse)
async def match_patient(request: Request, body: MatchRequest) -> MatchResponse:
    if not body.patient_id or len(body.patient_id) > 128:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "patient_id must be non-empty and at most 128 characters", "field": "patient_id"},
        )
    if body.max_results < 1 or body.max_results > 100:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "max_results must be between 1 and 100", "field": "max_results"},
        )
    if body.min_score < 0.0 or body.min_score > 1.0:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "min_score must be between 0.0 and 1.0", "field": "min_score"},
        )

    patient_repo: PatientRepository = request.app.state.patient_repo
    engine: MatchingEngine = request.app.state.matching_engine

    try:
        patient = patient_repo.get(body.patient_id)
    except Exception as exc:
        logger.error("match_patient_lookup_error", patient_id=body.patient_id, error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error looking up patient"},
        )

    if not patient:
        return JSONResponse(  # type: ignore[return-value]
            status_code=404,
            content={"detail": f"Patient {body.patient_id!r} not found"},
        )
    if not patient.features:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "Patient has no extracted features"},
        )

    start = time.time()
    try:
        results = await engine.match(patient.features, body)
    except Exception as exc:
        logger.error("match_engine_error", patient_id=body.patient_id, error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error during matching"},
        )
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


@router.post("/batch", response_model=BatchMatchResponse)
async def batch_match(request: Request, body: BatchMatchRequest) -> BatchMatchResponse:
    if not body.patient_ids:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "patient_ids must be non-empty", "field": "patient_ids"},
        )
    if len(body.patient_ids) > 50:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "patient_ids may contain at most 50 IDs", "field": "patient_ids"},
        )
    if body.max_results < 1 or body.max_results > 100:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "max_results must be between 1 and 100", "field": "max_results"},
        )
    if body.min_score < 0.0 or body.min_score > 1.0:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "min_score must be between 0.0 and 1.0", "field": "min_score"},
        )
    for pid in body.patient_ids:
        if len(pid) > 128:
            return JSONResponse(  # type: ignore[return-value]
                status_code=422,
                content={"detail": f"patient_id {pid!r} exceeds 128 characters", "field": "patient_ids"},
            )

    patient_repo: PatientRepository = request.app.state.patient_repo
    engine: MatchingEngine = request.app.state.matching_engine

    start = time.time()

    async def _match_one(patient_id: str) -> tuple[str, list]:
        try:
            patient = patient_repo.get(patient_id)
        except Exception as exc:
            logger.error("batch_match_lookup_error", patient_id=patient_id, error=str(exc))
            return patient_id, []
        if not patient or not patient.features:
            return patient_id, []
        req = MatchRequest(
            patient_id=patient_id,
            max_results=body.max_results,
            min_score=body.min_score,
        )
        try:
            results = await engine.match(patient.features, req)
            return patient_id, [r.model_dump() for r in results]
        except Exception as exc:
            logger.error("batch_match_engine_error", patient_id=patient_id, error=str(exc))
            return patient_id, []

    gathered = await asyncio.gather(*[_match_one(pid) for pid in body.patient_ids])
    elapsed_ms = (time.time() - start) * 1000

    return BatchMatchResponse(
        results=dict(gathered),
        total_patients=len(body.patient_ids),
        processing_time_ms=round(elapsed_ms, 1),
    )


@router.get("/history/{patient_id}")
async def match_history(request: Request, patient_id: str) -> dict:
    if len(patient_id) > 128:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "patient_id must be at most 128 characters", "field": "patient_id"},
        )
    try:
        db = request.app.state.db
        rows = db.list_match_results_by_patient(patient_id, limit=20)
    except Exception as exc:
        logger.error("match_history_error", patient_id=patient_id, error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error fetching match history"},
        )

    results = []
    for row in rows:
        try:
            explanation = MatchExplanation.model_validate_json(row["explanation"])
            results.append(
                MatchResult(
                    match_id=row["match_id"],
                    patient_id=row["patient_id"],
                    nct_id=row["nct_id"],
                    trial_title=row.get("trial_title", ""),
                    composite_score=row["composite_score"],
                    confidence=row["confidence"],  # type: ignore[arg-type]
                    explanation=explanation,
                    status=row["status"],  # type: ignore[arg-type]
                    created_at=row["created_at"],
                ).model_dump()
            )
        except Exception as exc:
            logger.warning("match_history_row_error", match_id=row.get("match_id"), error=str(exc))

    return {"patient_id": patient_id, "results": results, "total": len(results)}


@router.get("/{match_id}", response_model=MatchResult)
async def get_match(request: Request, match_id: str) -> MatchResult:
    try:
        db = request.app.state.db
        row = db.get_match_result(match_id)
    except Exception as exc:
        logger.error("get_match_error", match_id=match_id, error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error fetching match result"},
        )
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Match result not found"})  # type: ignore[return-value]

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
