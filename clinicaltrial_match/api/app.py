"""
FastAPI application factory for clinicaltrial-match.

Mirrors medguard's create_app() pattern with lifespan for embedding warmup.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from clinicaltrial_match.api.routes.health import router as health_router
from clinicaltrial_match.api.routes.matching import router as matching_router
from clinicaltrial_match.api.routes.patients import router as patients_router
from clinicaltrial_match.api.routes.trials import router as trials_router
from clinicaltrial_match.trials.fetcher import normalize_study

_DESCRIPTION = """
**clinicaltrial-match** — AI-powered clinical trial matching from patient records.

- Ingest patient records as FHIR R4 bundles or unstructured clinical notes
- Sync clinical trials from ClinicalTrials.gov
- Match patients to eligible trials with NLP-based eligibility parsing
- Get ranked results with confidence scores and plain-English explanations
"""

_X_GUIDANCE = (
    "Use POST /v1/match/live to match a patient to recruiting clinical trials. "
    "Send a JSON body with 'source' ('note' or 'fhir'), 'note_text' (free-text clinical note) "
    "or 'fhir_data' (FHIR R4 bundle), and optional 'max_results' (default 10) and "
    "'min_score' (0–1, default 0.2). Returns ranked trial matches with confidence scores "
    "and plain-English eligibility explanations. "
    "This endpoint requires an MPP payment of $0.05 USDC per call — "
    "send 'Authorization: Payment <credential>' after receiving the 402 challenge."
)


def _inject_mpp_schema(schema: dict[str, Any], ownership_proofs: list[str]) -> dict[str, Any]:
    """Inject MPP-required fields into an existing OpenAPI schema dict."""
    schema = copy.deepcopy(schema)

    # Agent-readable guidance at the top level
    schema.setdefault("info", {})["x-guidance"] = _X_GUIDANCE

    # Ownership proofs for mppscan.com registration
    if ownership_proofs:
        schema["x-discovery"] = {"ownershipProofs": ownership_proofs}

    # Annotate POST /v1/match/live as a payable operation
    live_op = schema.get("paths", {}).get("/v1/match/live", {}).get("post")
    if live_op:
        live_op["x-payment-info"] = {
            "pricingMode": "fixed",
            "price": "0.050000",
            "protocols": ["mpp"],
        }
        live_op.setdefault("responses", {})["402"] = {
            "description": "Payment Required — send MPP credential in Authorization header",
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error": {"type": "string"},
                            "price_usd": {"type": "string"},
                        },
                    }
                }
            },
        }

    return schema


def _seed_if_empty(db, trial_repo, embeddings) -> None:
    """Load bundled seed trials into the DB if it is empty."""
    import json

    log = structlog.get_logger()
    try:
        _, total = trial_repo.list(limit=1, offset=0)
        if total > 0:
            log.info("seed_skip", existing_trials=total)
            return
    except Exception:
        return

    seed_path = Path(__file__).parent.parent / "data" / "seed_trials.json"
    if not seed_path.exists():
        log.warning("seed_file_missing", path=str(seed_path))
        return

    try:
        studies = json.loads(seed_path.read_text())
        loaded = 0
        for raw_study in studies:
            try:
                normalized = normalize_study(raw_study)
                vec = embeddings.encode_one(
                    f"{normalized['title']}. {normalized['brief_summary'][:300]}. "
                    f"Conditions: {', '.join(normalized['conditions'])}."
                )
                trial_repo.save_raw(normalized, None, vec)
                loaded += 1
            except Exception as exc:
                log.warning(
                    "seed_trial_error",
                    nct_id=raw_study.get("protocolSection", {}).get("identificationModule", {}).get("nctId"),
                    error=str(exc),
                )
        log.info("seed_complete", loaded=loaded, total=len(studies))
    except Exception as exc:
        log.error("seed_failed", error=str(exc))


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    from clinicaltrial_match.api.mpp_payment import create_mpp
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

    claude = ClaudeClient(config.claude, martian_config=config.martian)
    structlog.get_logger().info("llm_backend", backend=claude.backend)
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
    app.state.mpp = create_mpp(config.mpp)

    # Seed DB from bundled data if empty (bypasses ClinicalTrials.gov IP blocks on cloud hosts)
    import asyncio

    def _seed_and_warmup():
        _seed_if_empty(db, trial_repo, embeddings)
        embeddings.warmup()

    asyncio.get_event_loop().run_in_executor(None, _seed_and_warmup)
    structlog.get_logger().info("seed_and_warmup_started_in_background")

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

    # credentials=False allows allow_origins=["*"] to work for browser fetch() from
    # GitHub Pages → ngrok / Render. credentials=True + wildcard origin is invalid per spec
    # and browsers block with "No Access-Control-Allow-Origin".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=False,
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

    # Custom OpenAPI schema — injects MPP x-payment-info, x-guidance, x-discovery
    ownership_proofs = config.mpp.ownership_proofs

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        base = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        app.openapi_schema = _inject_mpp_schema(base, ownership_proofs)
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui")

    return app
