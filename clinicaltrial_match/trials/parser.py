"""
Eligibility criteria parser using Claude haiku-4-5 tool-use.

Two-stage approach:
1. Regex pre-parse for age, gender, and text splitting
2. Claude tool-use for structured extraction of diagnoses, labs, medications

Falls back to regex-only results if Claude call fails or returns low confidence.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from clinicaltrial_match.infrastructure.claude_client import ClaudeClient
from clinicaltrial_match.trials.models import (
    AgeConstraint,
    DiagnosisConstraint,
    EligibilityCriteria,
    GenderConstraint,
    LabConstraint,
    MedicationConstraint,
)

logger = structlog.get_logger()

_EXTRACTION_TOOL: dict[str, Any] = {
    "name": "extract_eligibility_criteria",
    "description": "Extract structured eligibility criteria from clinical trial text",
    "input_schema": {
        "type": "object",
        "properties": {
            "required_diagnoses": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions/diagnoses a patient MUST have to be eligible",
            },
            "excluded_diagnoses": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Conditions/diagnoses that DISQUALIFY a patient",
            },
            "required_medications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Medications a patient must currently be taking",
            },
            "excluded_medications": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Medications that disqualify a patient",
            },
            "lab_constraints": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "test_name": {"type": "string"},
                        "operator": {"type": "string", "enum": ["<", "<=", ">", ">=", "=", "between"]},
                        "value": {"type": "number"},
                        "value_upper": {"type": "number"},
                        "unit": {"type": "string"},
                    },
                    "required": ["test_name", "operator", "value"],
                },
            },
            "other_inclusion": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Other inclusion criteria not captured above",
            },
            "other_exclusion": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Other exclusion criteria not captured above",
            },
            "parse_confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence that criteria were accurately extracted (0-1)",
            },
        },
        "required": ["parse_confidence"],
    },
}

_SYSTEM_PROMPT = (
    "You are a clinical informatics specialist. Extract structured eligibility criteria "
    "from clinical trial text. Be precise and conservative — only extract criteria that "
    "are clearly stated. Use the extract_eligibility_criteria tool to return structured data."
)

_INCLUSION_HEADER = re.compile(r"inclusion\s+criteria\s*:?", re.IGNORECASE)
_EXCLUSION_HEADER = re.compile(r"exclusion\s+criteria\s*:?", re.IGNORECASE)
_AGE_PATTERN = re.compile(
    r"(?:age[ds]?\s*(?:between\s*)?)?(\d+(?:\.\d+)?)\s*(?:years?|yrs?)?"
    r"(?:\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)\s*(?:years?|yrs?))?"
    r"(?:\s*(?:or\s+)?(?:older|younger|above|below|greater|less))?",
    re.IGNORECASE,
)


def _split_inclusion_exclusion(text: str) -> tuple[str, str]:
    excl_match = _EXCLUSION_HEADER.search(text)
    incl_match = _INCLUSION_HEADER.search(text)
    if excl_match:
        inclusion = text[: excl_match.start()].strip()
        exclusion = text[excl_match.end() :].strip()
        if incl_match and incl_match.start() < excl_match.start():
            inclusion = text[incl_match.end() : excl_match.start()].strip()
        return inclusion, exclusion
    if incl_match:
        return text[incl_match.end() :].strip(), ""
    return text, ""


def _regex_parse_age(text: str) -> AgeConstraint | None:
    min_age: float | None = None
    max_age: float | None = None

    range_m = re.search(r"(?:between|aged?)\s+(\d+)\s*(?:and|to|-)\s*(\d+)\s*(?:years?|yrs?)", text, re.IGNORECASE)
    if range_m:
        min_age = float(range_m.group(1))
        max_age = float(range_m.group(2))
    else:
        min_m = re.search(r"(?:minimum\s+age|at\s+least|>=?)\s*:?\s*(\d+)\s*(?:years?|yrs?)", text, re.IGNORECASE)
        if min_m:
            min_age = float(min_m.group(1))
        older_m = re.search(r"(\d+)\s*(?:years?|yrs?)\s+(?:of\s+age\s+)?(?:or\s+)?older", text, re.IGNORECASE)
        if older_m:
            min_age = float(older_m.group(1))
        max_m = re.search(r"(?:maximum\s+age|no\s+more\s+than)\s*:?\s*(\d+)\s*(?:years?|yrs?)", text, re.IGNORECASE)
        if max_m:
            max_age = float(max_m.group(1))

    if min_age is None and max_age is None:
        return None
    return AgeConstraint(minimum_age_years=min_age, maximum_age_years=max_age)


def _regex_parse_gender(text: str) -> GenderConstraint | None:
    text_lower = text.lower()
    if "male or female" in text_lower or "all genders" in text_lower or "both sexes" in text_lower:
        return GenderConstraint(allowed="ALL")
    if "female only" in text_lower or "women only" in text_lower:
        return GenderConstraint(allowed="FEMALE")
    if "male only" in text_lower or "men only" in text_lower:
        return GenderConstraint(allowed="MALE")
    return None


class EligibilityParser:
    def __init__(self, claude: ClaudeClient, concurrency: int = 5) -> None:
        self._claude = claude
        self._semaphore = asyncio.Semaphore(concurrency)

    async def parse(self, eligibility_text: str, gender_hint: str = "ALL") -> EligibilityCriteria:
        inclusion_text, exclusion_text = _split_inclusion_exclusion(eligibility_text)
        combined = inclusion_text + "\n" + exclusion_text

        age = _regex_parse_age(combined)
        gender_from_text = _regex_parse_gender(combined)
        if gender_hint != "ALL" and gender_from_text is None:
            gender_from_text = GenderConstraint(allowed=gender_hint)  # type: ignore[arg-type]

        criteria = EligibilityCriteria(
            raw_inclusion_text=inclusion_text,
            raw_exclusion_text=exclusion_text,
            age=age,
            gender=gender_from_text,
            parse_confidence=0.0,
            parsed_by="regex-fallback",
        )

        if not eligibility_text.strip():
            return criteria

        try:
            async with self._semaphore:
                result = await self._claude.tool_use(
                    model=self._claude.fast_model,
                    system=_SYSTEM_PROMPT,
                    user_message=(f"Inclusion Criteria:\n{inclusion_text}\n\nExclusion Criteria:\n{exclusion_text}"),
                    tools=[_EXTRACTION_TOOL],
                    tool_name="extract_eligibility_criteria",
                )
            if result and result.get("parse_confidence", 0) >= 0.4:
                criteria = _apply_claude_result(criteria, result)
        except Exception as exc:
            logger.warning("eligibility_parse_failed", error=str(exc))

        return criteria


def _apply_claude_result(criteria: EligibilityCriteria, result: dict[str, Any]) -> EligibilityCriteria:
    diagnoses = DiagnosisConstraint(
        required_conditions=result.get("required_diagnoses", []),
        excluded_conditions=result.get("excluded_diagnoses", []),
    )
    medications = MedicationConstraint(
        required=result.get("required_medications", []),
        excluded=result.get("excluded_medications", []),
    )
    labs = [
        LabConstraint(
            test_name=lab["test_name"],
            operator=lab["operator"],
            value=lab["value"],
            value_upper=lab.get("value_upper"),
            unit=lab.get("unit", ""),
        )
        for lab in result.get("lab_constraints", [])
        if "test_name" in lab and "value" in lab
    ]
    return EligibilityCriteria(
        raw_inclusion_text=criteria.raw_inclusion_text,
        raw_exclusion_text=criteria.raw_exclusion_text,
        age=criteria.age,
        gender=criteria.gender,
        diagnoses=diagnoses if (diagnoses.required_conditions or diagnoses.excluded_conditions) else None,
        labs=labs,
        medications=medications if (medications.required or medications.excluded) else None,
        other_inclusion=result.get("other_inclusion", []),
        other_exclusion=result.get("other_exclusion", []),
        parse_confidence=float(result.get("parse_confidence", 0.7)),
        parsed_by="claude-haiku",
    )
