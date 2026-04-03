"""Trials routes: GET/POST /v1/trials, sync, compare, autocomplete."""

from __future__ import annotations

import asyncio
import json
import uuid

import structlog
from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from clinicaltrial_match.api.models import (
    AutocompleteResponse,
    CompareRequest,
    CompareResponse,
    PaginatedTrialsResponse,
    SyncJobListResponse,
    SyncRequest,
    SyncResponse,
    SyncStatusResponse,
    TrialSummary,
)
from clinicaltrial_match.trials.fetcher import TrialFetcher
from clinicaltrial_match.trials.parser import EligibilityParser
from clinicaltrial_match.trials.repository import TrialRepository

router = APIRouter()
logger = structlog.get_logger()


@router.get("", response_model=PaginatedTrialsResponse)
async def list_trials(
    request: Request,
    condition: str | None = None,
    status: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> PaginatedTrialsResponse:
    if limit < 1 or limit > 100:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "limit must be between 1 and 100", "field": "limit"},
        )
    if condition and len(condition) > 200:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "condition must be at most 200 characters", "field": "condition"},
        )
    try:
        repo: TrialRepository = request.app.state.trial_repo
        trials, total = repo.list(status=status, condition=condition, limit=limit, offset=offset)
    except Exception as exc:
        logger.error("list_trials_db_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error fetching trials"},
        )
    return PaginatedTrialsResponse(
        trials=[
            TrialSummary(
                nct_id=t.nct_id,
                title=t.title,
                brief_summary=t.brief_summary[:500],
                conditions=t.conditions,
                phase=t.phase,
                status=t.status.value,
                sponsor=t.sponsor,
                locations=t.locations[:5],
            )
            for t in trials
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/autocomplete", response_model=AutocompleteResponse)
async def autocomplete_conditions(
    request: Request,
    q: str = Query(default="", min_length=0),
) -> AutocompleteResponse:
    if len(q) > 200:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "q must be at most 200 characters", "field": "q"},
        )
    try:
        db = request.app.state.db
        all_conditions = db.get_all_conditions()
    except Exception as exc:
        logger.error("autocomplete_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error fetching conditions"},
        )

    q_lower = q.lower().strip()
    suggestions = [c for c in all_conditions if not q_lower or q_lower in c.lower()][:10]

    return AutocompleteResponse(suggestions=suggestions)


@router.get("/{nct_id}")
async def get_trial(request: Request, nct_id: str) -> dict:
    try:
        db = request.app.state.db
        row = db.get_trial(nct_id)
    except Exception as exc:
        logger.error("get_trial_error", nct_id=nct_id, error=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})  # type: ignore[return-value]
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Trial not found"})  # type: ignore[return-value]
    return {
        "nct_id": row["nct_id"],
        "title": row["title"],
        "brief_summary": row.get("brief_summary", ""),
        "conditions": _parse_json_col(row.get("conditions")),
        "interventions": _parse_json_col(row.get("interventions")),
        "phase": row.get("phase", ""),
        "status": row.get("status", ""),
        "sponsor": row.get("sponsor", ""),
        "locations": _parse_json_col(row.get("locations")),
        "eligibility_text": row.get("eligibility_text", ""),
        "start_date": row.get("start_date"),
        "last_updated": row.get("last_updated"),
    }


@router.post("/sync", response_model=SyncResponse, status_code=202)
async def sync_trials(request: Request, body: SyncRequest) -> SyncResponse:
    if not body.condition or len(body.condition) > 200:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "condition must be non-empty and at most 200 characters", "field": "condition"},
        )
    db = request.app.state.db
    fetcher: TrialFetcher = request.app.state.trial_fetcher
    parser: EligibilityParser = request.app.state.eligibility_parser
    repo: TrialRepository = request.app.state.trial_repo
    embeddings = request.app.state.embeddings

    job_id = str(uuid.uuid4())
    try:
        db.create_sync_job(job_id, body.condition)
    except Exception as exc:
        logger.error("sync_job_create_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Failed to create sync job"},
        )

    try:
        asyncio.create_task(_run_sync(job_id, body, db, fetcher, parser, repo, embeddings))
    except Exception as exc:
        logger.error("sync_task_create_error", job_id=job_id, error=str(exc))
        db.update_sync_job(job_id, status="failed", error=str(exc)[:500])

    return SyncResponse(job_id=job_id, status="queued", condition=body.condition)


@router.get("/sync", response_model=SyncJobListResponse)
async def list_sync_jobs(request: Request, limit: int = 10) -> SyncJobListResponse:
    if limit < 1 or limit > 100:
        limit = 10
    try:
        db = request.app.state.db
        rows = db.list_sync_jobs(limit=limit)
    except Exception as exc:
        logger.error("list_sync_jobs_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error fetching sync jobs"},
        )
    jobs = [
        SyncStatusResponse(
            job_id=r["job_id"],
            status=r["status"],
            condition=r["condition"],
            trials_fetched=r["trials_fetched"] or 0,
            trials_parsed=r["trials_parsed"] or 0,
            error=r["error"],
        )
        for r in rows
    ]
    return SyncJobListResponse(jobs=jobs)


@router.get("/sync/{job_id}", response_model=SyncStatusResponse)
async def get_sync_status(request: Request, job_id: str) -> SyncStatusResponse:
    try:
        db = request.app.state.db
        row = db.get_sync_job(job_id)
    except Exception as exc:
        logger.error("get_sync_status_error", job_id=job_id, error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error fetching sync job"},
        )
    if not row:
        return JSONResponse(status_code=404, content={"detail": "Job not found"})  # type: ignore[return-value]
    return SyncStatusResponse(
        job_id=row["job_id"],
        status=row["status"],
        condition=row["condition"],
        trials_fetched=row["trials_fetched"] or 0,
        trials_parsed=row["trials_parsed"] or 0,
        error=row["error"],
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_trials(request: Request, body: CompareRequest) -> CompareResponse:
    if not body.nct_ids:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "nct_ids must be non-empty", "field": "nct_ids"},
        )
    if len(body.nct_ids) > 10:
        return JSONResponse(  # type: ignore[return-value]
            status_code=422,
            content={"detail": "nct_ids may contain at most 10 IDs", "field": "nct_ids"},
        )
    try:
        db = request.app.state.db
    except Exception as exc:
        logger.error("compare_trials_error", error=str(exc))
        return JSONResponse(  # type: ignore[return-value]
            status_code=500,
            content={"detail": "Internal server error"},
        )

    # Optionally resolve constraint results for a patient
    patient_features = None
    if body.patient_id:
        if len(body.patient_id) > 128:
            return JSONResponse(  # type: ignore[return-value]
                status_code=422,
                content={"detail": "patient_id must be at most 128 characters", "field": "patient_id"},
            )
        try:
            patient_repo = request.app.state.patient_repo
            patient = patient_repo.get(body.patient_id)
            if patient:
                patient_features = patient.features
        except Exception as exc:
            logger.warning("compare_patient_lookup_error", error=str(exc))

    trials_out: list[dict] = []
    for nct_id in body.nct_ids:
        try:
            row = db.get_trial(nct_id)
        except Exception as exc:
            logger.error("compare_get_trial_error", nct_id=nct_id, error=str(exc))
            trials_out.append({"nct_id": nct_id, "error": "Failed to fetch trial"})
            continue

        if not row:
            trials_out.append({"nct_id": nct_id, "error": "Trial not found"})
            continue

        entry: dict = {
            "nct_id": row["nct_id"],
            "title": row["title"],
            "conditions": _parse_json_col(row.get("conditions")),
            "eligibility_summary": (row.get("eligibility_text") or "")[:500],
        }

        if patient_features and row.get("eligibility_criteria"):
            try:
                from clinicaltrial_match.matching.constraints import ConstraintEvaluator
                from clinicaltrial_match.trials.models import EligibilityCriteria

                criteria = EligibilityCriteria.model_validate_json(row["eligibility_criteria"])
                evaluator = ConstraintEvaluator()
                constraint_results, _ = evaluator.evaluate(criteria, patient_features)
                entry["constraint_results"] = [r.model_dump() for r in constraint_results]
            except Exception as exc:
                logger.warning("compare_constraint_eval_error", nct_id=nct_id, error=str(exc))

        trials_out.append(entry)

    return CompareResponse(trials=trials_out)


def _parse_json_col(value: str | None) -> list:
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


async def _run_sync(
    job_id: str,
    body: SyncRequest,
    db,
    fetcher: TrialFetcher,
    parser: EligibilityParser,
    repo: TrialRepository,
    embeddings,
) -> None:
    import time

    db.update_sync_job(job_id, status="running")
    fetched = 0
    parsed = 0
    try:
        async for raw in fetcher.fetch_by_condition(
            body.condition,
            status_filter=body.status_filter,
            max_trials=body.max_trials,
        ):
            fetched += 1
            db.update_sync_job(job_id, trials_fetched=fetched)

            try:
                criteria = None
                if raw.get("eligibility_text"):
                    gender_hint = raw.get("_gender_hint", "ALL")
                    criteria = await parser.parse(raw["eligibility_text"], gender_hint)

                embedding_vec = None
                try:
                    title = raw.get("title", "")
                    summary = raw.get("brief_summary", "")
                    conds = ", ".join(raw.get("conditions", []))
                    elig_snippet = (raw.get("eligibility_text") or "")[:500]
                    embed_text = f"{title}. {summary}. Conditions: {conds}. {elig_snippet}"
                    embedding_vec = embeddings.encode_one(embed_text)
                except Exception as exc:
                    logger.warning("embedding_failed", nct_id=raw.get("nct_id"), error=str(exc))

                repo.save_raw(raw, criteria, embedding_vec)
                parsed += 1
                db.update_sync_job(job_id, trials_parsed=parsed)
            except Exception as exc:
                logger.error(
                    "sync_trial_error",
                    job_id=job_id,
                    nct_id=raw.get("nct_id"),
                    error=str(exc),
                )
                # Continue with next trial — one failure doesn't abort the sync

        db.update_sync_job(
            job_id,
            status="completed",
            trials_fetched=fetched,
            trials_parsed=parsed,
            completed_at=time.time(),
        )
        logger.info("sync_completed", job_id=job_id, fetched=fetched, parsed=parsed)
    except Exception as exc:
        logger.error("sync_failed", job_id=job_id, error=str(exc))
        db.update_sync_job(job_id, status="failed", error=str(exc)[:500])
