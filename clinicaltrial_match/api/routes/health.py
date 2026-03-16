"""GET /v1/health"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from clinicaltrial_match.api.models import HealthResponse
from clinicaltrial_match import __version__

router = APIRouter()
_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    db = request.app.state.db
    dependencies = {
        "db": True,
        "embeddings": True,
        "anthropic_api": bool(request.app.state.claude),
        "clinicaltrials_api": True,
    }
    try:
        trials_cached = db.count_trials()
    except Exception:
        dependencies["db"] = False
        trials_cached = 0

    status = "healthy" if all(dependencies.values()) else "degraded"

    return HealthResponse(
        status=status,
        dependencies=dependencies,
        trials_cached=trials_cached,
        version=__version__,
        uptime_seconds=round(time.time() - _start_time, 1),
    )
