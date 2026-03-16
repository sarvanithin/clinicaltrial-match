"""Shared test fixtures: in-memory SQLite, mock Claude, deterministic embeddings."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from clinicaltrial_match.infrastructure.db import Database
from clinicaltrial_match.infrastructure.embeddings import EmbeddingIndex
from tests.fixtures.sample_criteria import make_sample_criteria
from tests.fixtures.sample_fhir import SAMPLE_FHIR_BUNDLE


@pytest.fixture
def in_memory_db() -> Database:
    db = Database(":memory:")
    db.connect()
    return db


@pytest.fixture
def mock_claude():
    claude = MagicMock()
    claude.fast_model = "claude-haiku-4-5-20251001"
    claude.reasoning_model = "claude-sonnet-4-6-20251101"
    # Default tool_use returns a high-confidence extraction
    claude.tool_use = AsyncMock(return_value={
        "required_diagnoses": ["Type 2 diabetes mellitus"],
        "excluded_diagnoses": [],
        "required_medications": [],
        "excluded_medications": ["insulin"],
        "lab_constraints": [
            {"test_name": "HbA1c", "operator": "between", "value": 7.0, "value_upper": 10.0, "unit": "%"}
        ],
        "other_inclusion": [],
        "other_exclusion": [],
        "parse_confidence": 0.9,
    })
    claude.complete = AsyncMock(return_value="yes")
    return claude


@pytest.fixture
def mock_embeddings(in_memory_db: Database) -> EmbeddingIndex:
    rng = np.random.default_rng(42)
    idx = EmbeddingIndex.__new__(EmbeddingIndex)
    idx._config = MagicMock()
    idx._config.batch_size = 64
    idx._config.use_faiss = False
    idx._config.faiss_rebuild_threshold = 5000
    idx._model = None
    idx._nct_ids = []
    idx._matrix = None
    idx._faiss_index = None

    def _fake_encode(texts):
        vecs = rng.standard_normal((len(texts), 384)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    idx.encode = _fake_encode
    idx.encode_one = lambda t: _fake_encode([t])[0]
    return idx


@pytest.fixture
def sample_criteria():
    return make_sample_criteria()


@pytest.fixture
def sample_fhir_bundle():
    return SAMPLE_FHIR_BUNDLE
