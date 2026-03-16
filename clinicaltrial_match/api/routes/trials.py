"""Trials routes: GET /v1/trials, POST /v1/trials/sync, GET /v1/trials/sync/{job_id}"""
from __future__ import annotations

import asyncio
import uuid

import structlog
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from clinicaltrial_match.api.models import (
    PaginatedTrialsResponse,
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
    repo: TrialRepository = request.app.state.trial_repo
    trials, total = repo.list(status=status, condition=condition, limit=limit, offset=offset)
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


@router.post("/sync", response_model=SyncResponse, status_code=202)
async def sync_trials(request: Request, body: SyncRequest) -> SyncResponse:
    db = request.app.state.db
    fetcher: TrialFetcher = request.app.state.trial_fetcher
    parser: EligibilityParser = request.app.state.eligibility_parser
    repo: TrialRepository = request.app.state.trial_repo
    embeddings = request.app.state.embeddings

    job_id = str(uuid.uuid4())
    db.create_sync_job(job_id, body.condition)

    asyncio.create_task(
        _run_sync(job_id, body, db, fetcher, parser, repo, embeddings)
    )

    return SyncResponse(job_id=job_id, status="queued", condition=body.condition)


@router.get("/sync/{job_id}", response_model=SyncStatusResponse)
async def get_sync_status(request: Request, job_id: str) -> SyncStatusResponse:
    db = request.app.state.db
    row = db.get_sync_job(job_id)
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

            criteria = None
            if raw.get("eligibility_text"):
                gender_hint = raw.get("_gender_hint", "ALL")
                criteria = await parser.parse(raw["eligibility_text"], gender_hint)

            # Build embedding text
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
