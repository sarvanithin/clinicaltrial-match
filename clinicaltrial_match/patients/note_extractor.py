"""
Clinical note extractor using spaCy NER + Claude haiku-4-5 tool-use.

For unstructured clinical text → PatientFeatures.
Notes >4000 tokens are chunked and merged.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import structlog

from clinicaltrial_match.infrastructure.claude_client import ClaudeClient
from clinicaltrial_match.patients.models import (
    Diagnosis,
    Gender,
    LabValue,
    Medication,
    Patient,
    PatientFeatures,
    Procedure,
)

logger = structlog.get_logger()

_EXTRACTION_TOOL: dict[str, Any] = {
    "name": "extract_patient_features",
    "description": "Extract structured patient features from a clinical note",
    "input_schema": {
        "type": "object",
        "properties": {
            "age_years": {"type": "number", "description": "Patient age in years"},
            "gender": {"type": "string", "enum": ["male", "female", "other", "unknown"]},
            "diagnoses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "code": {"type": "string"},
                        "status": {"type": "string", "enum": ["active", "resolved", "unknown"]},
                    },
                    "required": ["name"],
                },
            },
            "medications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "dose": {"type": "string"},
                        "active": {"type": "boolean"},
                    },
                    "required": ["name"],
                },
            },
            "lab_values": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_name": {"type": "string"},
                        "value": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                    "required": ["test_name", "value"],
                },
            },
            "procedures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            "clinical_summary": {
                "type": "string",
                "description": "2-3 sentence plain-English summary of the patient's key features",
            },
        },
        "required": ["clinical_summary"],
    },
}

_SYSTEM_PROMPT = (
    "You are a clinical informatics specialist. Extract structured patient features "
    "from the clinical note below. Focus on diagnoses, medications, lab values, procedures, "
    "age, and gender. Be conservative — only extract what is clearly stated. "
    "Use the extract_patient_features tool to return structured data."
)

_MAX_CHUNK_CHARS = 12000  # roughly 4000 tokens


def _chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # Try to break at sentence boundary
        boundary = text.rfind(". ", start, end)
        if boundary > start:
            end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end
    return chunks


def _merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {}
    merged: dict[str, Any] = {}
    for r in results:
        if r.get("age_years") and not merged.get("age_years"):
            merged["age_years"] = r["age_years"]
        if r.get("gender") and r["gender"] != "unknown" and not merged.get("gender"):
            merged["gender"] = r["gender"]
        for key in ("diagnoses", "medications", "lab_values", "procedures"):
            merged.setdefault(key, [])
            # Deduplicate by name
            existing_names = {i.get("name", "").lower() for i in merged[key]}
            for item in r.get(key, []):
                if item.get("name", "").lower() not in existing_names:
                    merged[key].append(item)
                    existing_names.add(item.get("name", "").lower())
    merged["clinical_summary"] = results[0].get("clinical_summary", "")
    return merged


def _build_features(patient_id: str, result: dict[str, Any]) -> PatientFeatures:
    diagnoses = [
        Diagnosis(
            name=d["name"],
            code=d.get("code", ""),
            status=d.get("status", "unknown"),  # type: ignore[arg-type]
        )
        for d in result.get("diagnoses", [])
    ]
    medications = [
        Medication(
            name=m["name"],
            dose=m.get("dose", ""),
            active=m.get("active", True),
        )
        for m in result.get("medications", [])
    ]
    lab_values = [
        LabValue(
            test_name=lv["test_name"],
            value=lv["value"],
            unit=lv.get("unit", ""),
        )
        for lv in result.get("lab_values", [])
    ]
    procedures = [Procedure(name=p["name"]) for p in result.get("procedures", [])]

    gender_raw = result.get("gender", "unknown")
    gender = Gender(gender_raw) if gender_raw in Gender._value2member_map_ else Gender.UNKNOWN

    return PatientFeatures(
        patient_id=patient_id,
        age_years=result.get("age_years"),
        gender=gender,
        diagnoses=diagnoses,
        lab_values=lab_values,
        medications=medications,
        procedures=procedures,
        clinical_summary=result.get("clinical_summary", "Patient from clinical note"),
        extraction_method="nlp",
        extraction_confidence=0.85,
    )


class NoteExtractor:
    def __init__(self, claude: ClaudeClient) -> None:
        self._claude = claude
        self._nlp = None

    def _load_spacy(self) -> Any:
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = None
        return self._nlp

    async def extract(self, note_text: str, patient_id: str | None = None) -> Patient:
        if not patient_id:
            patient_id = hashlib.md5(note_text.encode()).hexdigest()[:12]

        chunks = _chunk_text(note_text)
        chunk_results: list[dict[str, Any]] = []

        for chunk in chunks:
            try:
                result = await self._claude.tool_use(
                    model=self._claude.fast_model,
                    system=_SYSTEM_PROMPT,
                    user_message=chunk,
                    tools=[_EXTRACTION_TOOL],
                    tool_name="extract_patient_features",
                )
                if result:
                    chunk_results.append(result)
            except Exception as exc:
                logger.warning("note_extraction_failed", error=str(exc), chunk_len=len(chunk))

        if chunk_results:
            merged = _merge_results(chunk_results)
            features = _build_features(patient_id, merged)
        else:
            # Minimal fallback
            features = PatientFeatures(
                patient_id=patient_id,
                clinical_summary=note_text[:200],
                extraction_method="nlp",
                extraction_confidence=0.1,
            )

        return Patient(
            patient_id=patient_id,
            source_type="note",
            raw_input=note_text,
            features=features,
            created_at=time.time(),
        )
