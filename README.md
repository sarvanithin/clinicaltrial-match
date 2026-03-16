# ClinicalTrial Match

[![CI](https://github.com/sarvanithin/clinicaltrial-match/actions/workflows/ci.yml/badge.svg)](https://github.com/sarvanithin/clinicaltrial-match/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

AI-powered clinical trial matching from patient records. Connects eligible patients to open trials using semantic search and structured eligibility constraint evaluation.

---

## Features

- **Trial sync** — Fetch from ClinicalTrials.gov API v2 (400K+ trials) by condition
- **Eligibility parsing** — Two-stage: regex pre-parse + Claude Haiku tool-use → structured JSON constraints
- **Patient ingestion** — FHIR R4 bundles or unstructured clinical notes (Claude extraction)
- **Semantic matching** — `all-MiniLM-L6-v2` embeddings, numpy cosine (FAISS at >5k trials)
- **Constraint evaluation** — Age, gender, diagnosis (fuzzy + ICD-10), labs, medications
- **Composite ranking** — 40% semantic + 60% constraint satisfaction with confidence tiers
- **Plain-English explanations** — Per-constraint pass/fail with disqualifying factors
- **Web UI** — Single-page app with trial browser, patient ingestion, match results
- **REST API** — FastAPI with OpenAPI docs at `/docs`
- **Docker** — Production-ready container with health checks

---

## Architecture

```
clinicaltrial_match/
├── trials/          # ClinicalTrials.gov fetcher, eligibility parser, trial repo
├── patients/        # FHIR R4 parser, clinical note extractor, patient repo
├── matching/        # Semantic searcher, constraint evaluator, ranker, engine
├── infrastructure/  # SQLite DB, embedding index, Claude client
└── api/             # FastAPI app, routes, middleware, models
```

**Matching pipeline:**
1. `POST /v1/patients/fhir` or `/v1/patients/note` → extract `PatientFeatures`
2. `POST /v1/trials/sync` → fetch & parse trials from ClinicalTrials.gov
3. `POST /v1/match` → semantic search (top K×3) → constraint filter → composite rank

**Scoring:**
```
composite = 0.40 × semantic_score + 0.60 × constraint_score
high (eligible)          ≥ 0.80, no hard constraint failures
medium (likely eligible) ≥ 0.60
low (potentially)        ≥ 0.30
ineligible               < 0.30
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
git clone https://github.com/sarvanithin/clinicaltrial-match.git
cd clinicaltrial-match

# Core install
pip install -e .

# With NLP support (clinical note extraction)
pip install -e ".[nlp]"
python -m spacy download en_core_web_sm
```

### Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m clinicaltrial_match serve
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
# → http://localhost:8000/ui    (Web UI)
```

### Docker

```bash
docker-compose -f docker/docker-compose.yml up
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/trials` | List cached trials (filter: condition, status) |
| `POST` | `/v1/trials/sync` | Trigger background trial sync from ClinicalTrials.gov |
| `GET` | `/v1/trials/sync` | List recent sync jobs |
| `GET` | `/v1/trials/sync/{job_id}` | Poll sync job status |
| `POST` | `/v1/trials/compare` | Side-by-side trial comparison (with optional patient constraints) |
| `GET` | `/v1/trials/autocomplete` | Condition autocomplete from cached trials |
| `POST` | `/v1/patients/fhir` | Ingest FHIR R4 Bundle, extract features |
| `POST` | `/v1/patients/note` | Ingest clinical note text, extract features |
| `GET` | `/v1/patients` | List ingested patients |
| `POST` | `/v1/match` | Match patient to trials |
| `POST` | `/v1/match/batch` | Batch match multiple patients |
| `GET` | `/v1/match/history/{patient_id}` | Past match results for a patient |
| `GET` | `/v1/match/{match_id}` | Retrieve specific match result |
| `GET` | `/v1/health` | Health check + dependency status |

### Example: ingest patient + match

```bash
# Sync trials
curl -X POST http://localhost:8000/v1/trials/sync \
  -H 'Content-Type: application/json' \
  -d '{"condition": "type 2 diabetes", "max_trials": 50}'

# Ingest patient
curl -X POST http://localhost:8000/v1/patients/fhir \
  -H 'Content-Type: application/json' \
  -d @- <<'EOF'
{
  "resourceType": "Bundle", "type": "collection",
  "entry": [
    {"resource": {"resourceType": "Patient", "id": "p1", "birthDate": "1972-06-10", "gender": "male"}},
    {"resource": {"resourceType": "Condition", "code": {"coding": [{"code": "E11", "display": "Type 2 diabetes mellitus"}]}}}
  ]
}
EOF

# Match
curl -X POST http://localhost:8000/v1/match \
  -H 'Content-Type: application/json' \
  -d '{"patient_id": "p1", "max_results": 10}'
```

---

## Configuration

Config file: `~/.clinicaltrial_match/config.json` — overridable via env vars (prefix `CTM_`, delimiter `__`):

```bash
export CTM_CLAUDE__FAST_MODEL=claude-haiku-4-5-20251001
export CTM_CLAUDE__REASONING_MODEL=claude-sonnet-4-6-20251101
export CTM_TRIALS__PAGE_SIZE=100
export CTM_TRIALS__MAX_TRIALS_PER_SYNC=1000
export CTM_EMBEDDING__USE_FAISS=true          # for >5k trials
export CTM_DB__PATH=/data/ctm.db
export CTM_AUTH__API_KEY=my-secret-key        # optional API key auth
export CTM_API__LOG_LEVEL=INFO
```

---

## Development

```bash
pip install -e ".[dev]"

# Run all unit tests
pytest tests/ -m "not integration" -v --cov=clinicaltrial_match

# Lint + format
ruff check clinicaltrial_match/ tests/
ruff format clinicaltrial_match/ tests/

# Type check
mypy clinicaltrial_match/ --ignore-missing-imports

# Integration tests (requires ANTHROPIC_API_KEY + network)
pytest tests/ -m integration -v
```

---

## Security

- **API key auth** — Set `CTM_AUTH__API_KEY` to require `X-API-Key` header on all `/v1/*` routes
- **No secrets in code** — API keys loaded from env vars only
- **Input validation** — All endpoints validate bounds on numeric params and string lengths

---

## License

MIT
