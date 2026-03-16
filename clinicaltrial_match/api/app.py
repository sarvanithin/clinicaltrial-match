"""
FastAPI application factory for clinicaltrial-match.

Mirrors medguard's create_app() pattern with lifespan for embedding warmup.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from clinicaltrial_match.api.routes.health import router as health_router
from clinicaltrial_match.api.routes.matching import router as matching_router
from clinicaltrial_match.api.routes.patients import router as patients_router
from clinicaltrial_match.api.routes.trials import router as trials_router

_DESCRIPTION = """
**clinicaltrial-match** — AI-powered clinical trial matching from patient records.

- Ingest patient records as FHIR R4 bundles or unstructured clinical notes
- Sync clinical trials from ClinicalTrials.gov
- Match patients to eligible trials with NLP-based eligibility parsing
- Get ranked results with confidence scores and plain-English explanations
"""


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    from clinicaltrial_match.config import get_config
    from clinicaltrial_match.infrastructure.claude_client import ClaudeClient
    from clinicaltrial_match.infrastructure.db import Database
    from clinicaltrial_match.infrastructure.embeddings import EmbeddingIndex
    from clinicaltrial_match.matching.engine import MatchingEngine
    from clinicaltrial_match.matching.semantic import SemanticSearcher
    from clinicaltrial_match.patients.note_extractor import NoteExtractor
    from clinicaltrial_match.patients.repository import PatientRepository
    from clinicaltrial_match.trials.fetcher import TrialFetcher
    from clinicaltrial_match.trials.parser import EligibilityParser
    from clinicaltrial_match.trials.repository import TrialRepository

    config = get_config()

    db = Database(config.db.path)
    db.connect()

    embeddings = EmbeddingIndex(config.embedding)
    embeddings.load_from_db(db)

    claude = ClaudeClient(config.claude)
    trial_repo = TrialRepository(db, embeddings)
    patient_repo = PatientRepository(db)
    eligibility_parser = EligibilityParser(claude, concurrency=config.trials.parse_concurrency)
    note_extractor = NoteExtractor(claude)
    trial_fetcher = TrialFetcher(config.trials)
    searcher = SemanticSearcher(embeddings)
    matching_engine = MatchingEngine(db, trial_repo, searcher, claude)

    app.state.db = db
    app.state.embeddings = embeddings
    app.state.claude = claude
    app.state.trial_repo = trial_repo
    app.state.patient_repo = patient_repo
    app.state.eligibility_parser = eligibility_parser
    app.state.note_extractor = note_extractor
    app.state.trial_fetcher = trial_fetcher
    app.state.matching_engine = matching_engine

    yield

    db.close()


def create_app() -> FastAPI:
    from clinicaltrial_match.config import get_config

    config = get_config()

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, config.api.log_level.upper(), logging.INFO)),
    )

    app = FastAPI(
        title="clinicaltrial-match",
        description=_DESCRIPTION,
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API key auth — only active when CTM_AUTH__API_KEY is set
    if config.auth.api_key:
        from clinicaltrial_match.api.middleware import APIKeyMiddleware

        app.add_middleware(APIKeyMiddleware, api_key=config.auth.api_key)

    app.include_router(trials_router, prefix="/v1/trials", tags=["trials"])
    app.include_router(patients_router, prefix="/v1/patients", tags=["patients"])
    app.include_router(matching_router, prefix="/v1/match", tags=["matching"])
    app.include_router(health_router, prefix="/v1", tags=["health"])

    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/ui", StaticFiles(directory=static_dir, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui")

    return app
