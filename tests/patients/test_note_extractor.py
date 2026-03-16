"""Tests for clinical note extractor."""
from __future__ import annotations

import pytest

from clinicaltrial_match.patients.note_extractor import NoteExtractor, _chunk_text, _merge_results


SAMPLE_NOTE = (
    "Patient is a 52-year-old male with a history of type 2 diabetes mellitus "
    "diagnosed 10 years ago. Current medications include Metformin 500mg twice daily. "
    "Most recent HbA1c: 8.4%. No insulin use. eGFR 65 mL/min."
)


def test_chunk_text_no_split():
    chunks = _chunk_text("Short text", max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0] == "Short text"


def test_chunk_text_splits_long():
    long_text = "Word " * 5000
    chunks = _chunk_text(long_text, max_chars=1000)
    assert len(chunks) > 1
    assert all(len(c) <= 1100 for c in chunks)  # some slack for boundary


def test_merge_results_deduplicates():
    r1 = {"age_years": 52, "gender": "male", "diagnoses": [{"name": "Diabetes", "status": "active"}], "medications": [], "lab_values": [], "procedures": [], "clinical_summary": "A"}
    r2 = {"age_years": None, "gender": "unknown", "diagnoses": [{"name": "Diabetes", "status": "active"}], "medications": [{"name": "Metformin", "active": True}], "lab_values": [], "procedures": [], "clinical_summary": "B"}
    merged = _merge_results([r1, r2])
    assert merged["age_years"] == 52
    assert merged["gender"] == "male"
    assert len(merged["diagnoses"]) == 1  # deduplicated
    assert len(merged["medications"]) == 1


@pytest.mark.asyncio
async def test_extractor_calls_claude(mock_claude):
    mock_claude.tool_use.return_value = {
        "age_years": 52,
        "gender": "male",
        "diagnoses": [{"name": "Type 2 diabetes mellitus", "status": "active"}],
        "medications": [{"name": "Metformin", "active": True}],
        "lab_values": [{"test_name": "HbA1c", "value": 8.4, "unit": "%"}],
        "procedures": [],
        "clinical_summary": "52-year-old male with type 2 diabetes on Metformin",
    }
    extractor = NoteExtractor(mock_claude)
    patient = await extractor.extract(SAMPLE_NOTE)
    mock_claude.tool_use.assert_called_once()
    assert patient.features is not None
    assert patient.features.age_years == 52


@pytest.mark.asyncio
async def test_extractor_fallback_on_error(mock_claude):
    mock_claude.tool_use.side_effect = RuntimeError("API error")
    extractor = NoteExtractor(mock_claude)
    patient = await extractor.extract(SAMPLE_NOTE)
    assert patient.features is not None
    assert patient.features.extraction_confidence < 0.5


@pytest.mark.asyncio
async def test_extractor_uses_provided_patient_id(mock_claude):
    mock_claude.tool_use.return_value = {"clinical_summary": "Test", "parse_confidence": 0.8}
    extractor = NoteExtractor(mock_claude)
    patient = await extractor.extract("Some note", patient_id="custom-id-123")
    assert patient.patient_id == "custom-id-123"
