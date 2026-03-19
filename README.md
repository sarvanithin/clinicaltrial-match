# ClinicalTrial Match

[![CI](https://github.com/sarvanithin/clinicaltrial-match/actions/workflows/ci.yml/badge.svg)](https://github.com/sarvanithin/clinicaltrial-match/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/sarvanithin/clinicaltrial-match)

**AI-powered clinical trial matching — privacy-first, open source.**

Upload a patient record (FHIR R4 bundle or clinical note), get back ranked matching trials from ClinicalTrials.gov in seconds. **Patient data is never stored** — processed in memory and discarded after matching.

---

## How it works

```
Your patient data (FHIR / note)
        │
        ▼  extracted in-memory, never written to disk
  PatientFeatures
        │
        ├── Semantic search — all-MiniLM-L6-v2 embeddings against 400K+ trials
        │
        └── Constraint evaluation — age · gender · diagnosis · labs · medications
                │
                ▼
        Ranked matches with plain-English explanations
        (patient data discarded immediately after)
```

**Scoring:**
```
composite = 0.40 × semantic_score + 0.60 × constraint_score

≥ 0.80  →  High confidence / Eligible
≥ 0.60  →  Medium / Likely eligible
≥ 0.30  →  Low / Potentially eligible
< 0.30  →  Filtered out
```

---

## Privacy model

- **No patient storage** — `POST /v1/match/live` processes everything in-memory with `persist=False`
- **No patient database** — only ClinicalTrials.gov public data is cached in SQLite
- **Open source** — full audit trail; you can run it on your own infrastructure
- **API key auth** — optional `CTM_AUTH__API_KEY` to restrict access to your instance

---

## Quick start

### Prerequisites
- Python 3.11+
- [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
git clone https://github.com/sarvanithin/clinicaltrial-match.git
cd clinicaltrial-match

pip install -e .                      # core
pip install -e ".[nlp]"              # + clinical note extraction
python -m spacy download en_core_web_sm
```

### Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -m clinicaltrial_match serve
# Web UI  →  http://localhost:8000
# API docs →  http://localhost:8000/docs
```

### Docker

```bash
ANTHROPIC_API_KEY=sk-ant-... docker-compose -f docker/docker-compose.yml up
```

### Deploy publicly (Render.com — free)

1. Click **Deploy to Render** badge above
2. In the Render dashboard, set the `ANTHROPIC_API_KEY` environment variable (marked secret — never stored in git)
3. Click **Deploy** — you get a live URL like `https://clinicaltrial-match.onrender.com`

The API key is stored securely in Render's environment secrets, not in the codebase.

---

## Try it — ephemeral match (no data stored)

```bash
# 1. Sync some trials first
curl -X POST http://localhost:8000/v1/trials/sync \
  -H 'Content-Type: application/json' \
  -d '{"condition": "type 2 diabetes", "max_trials": 100}'

# 2. Match from a clinical note — nothing written to DB
curl -X POST http://localhost:8000/v1/match/live \
  -H 'Content-Type: application/json' \
  -d '{
    "source": "note",
    "note_text": "65-year-old male with type 2 diabetes (HbA1c 8.2%) and hypertension.",
    "max_results": 5
  }'
```

Response includes:
```json
{
  "patient_label": "65y male",
  "privacy_notice": "Patient data processed in memory only — nothing stored",
  "matches": [ ... ],
  "processing_time_ms": 420.1
}
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/match/live` | **Ephemeral match** — FHIR or note → ranked trials, no storage |
| `GET` | `/v1/trials` | List cached trials (filter: condition, status) |
| `POST` | `/v1/trials/sync` | Fetch trials from ClinicalTrials.gov (background job) |
| `GET` | `/v1/trials/sync/{job_id}` | Poll sync job status |
| `GET` | `/v1/trials/{nct_id}` | Get single trial details |
| `POST` | `/v1/trials/compare` | Side-by-side trial comparison |
| `GET` | `/v1/trials/autocomplete` | Condition autocomplete |
| `POST` | `/v1/match` | Match a stored patient (advanced use) |
| `POST` | `/v1/match/batch` | Batch match multiple stored patients |
| `GET` | `/v1/health` | Health check |

Full interactive docs at `/docs` (Swagger UI).

---

## Project structure

```
clinicaltrial_match/
├── trials/          # ClinicalTrials.gov fetcher · eligibility parser · trial repo
├── patients/        # FHIR R4 parser · clinical note extractor
├── matching/        # Semantic searcher · constraint evaluator · ranker · engine
├── infrastructure/  # SQLite · embedding index · Claude client
└── api/             # FastAPI app · routes · middleware · models
tests/               # 81 unit tests, no external API calls
docker/              # Dockerfile + docker-compose.yml
```

---

## Configuration

All settings via environment variables (prefix `CTM_`, delimiter `__`):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | required | Anthropic API key |
| `CTM_DB__PATH` | `~/.clinicaltrial_match/data.db` | SQLite path |
| `CTM_CLAUDE__FAST_MODEL` | `claude-haiku-4-5-20251001` | Model for parsing |
| `CTM_CLAUDE__REASONING_MODEL` | `claude-sonnet-4-6-20251101` | Model for reasoning |
| `CTM_EMBEDDING__USE_FAISS` | `false` | Use FAISS for >5k trials |
| `CTM_AUTH__API_KEY` | unset | Optional API key for all `/v1/*` routes |
| `CTM_API__LOG_LEVEL` | `INFO` | Log level |

See `.env.example` for a full list.

---

## Development

```bash
pip install -e ".[dev]"

pytest tests/ -m "not integration" -v --cov=clinicaltrial_match
ruff check clinicaltrial_match/ tests/
mypy clinicaltrial_match/ --ignore-missing-imports
```

Integration tests (hit real APIs):
```bash
ANTHROPIC_API_KEY=... pytest tests/ -m integration -v
```

---

## Disclaimer

This tool is for **research and informational purposes only**. It is not a substitute for clinical judgment. Always verify trial eligibility directly with the trial site and a qualified clinician before making any medical decisions.

---

## License

MIT © 2026 Nithin Sarva
