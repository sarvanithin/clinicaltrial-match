"""
Composite scorer and ranker for trial matches.

composite = 0.40 * semantic_score + 0.60 * constraint_score

Confidence tiers:
  - high/eligible:              composite >= 0.8 AND no failed hard constraints
  - medium/likely_eligible:     composite >= 0.6
  - low/potentially_eligible:   composite >= 0.3
  - ineligible:                 composite < 0.3
"""
from __future__ import annotations

from typing import Literal

from clinicaltrial_match.matching.models import ConstraintResult, MatchExplanation, MatchResult

SEMANTIC_WEIGHT = 0.40
CONSTRAINT_WEIGHT = 0.60


def _is_hard_constraint(result: ConstraintResult) -> bool:
    """Age, gender, and explicit exclusions are hard constraints."""
    c = result.criterion.lower()
    return (
        c.startswith("age")
        or c.startswith("gender")
        or c.startswith("no diagnosis")
        or c.startswith("not on medication")
    )


def _assign_tier(
    composite: float,
    failed: list[ConstraintResult],
) -> tuple[Literal["high", "medium", "low"], Literal["eligible", "likely_eligible", "potentially_eligible", "ineligible"]]:
    hard_failures = [r for r in failed if _is_hard_constraint(r)]
    if composite >= 0.8 and not hard_failures:
        return "high", "eligible"
    if composite >= 0.6:
        return "medium", "likely_eligible"
    if composite >= 0.3:
        return "low", "potentially_eligible"
    return "low", "ineligible"


def build_match_result(
    match_id: str,
    patient_id: str,
    nct_id: str,
    trial_title: str,
    semantic_score: float,
    constraint_results: list[ConstraintResult],
    created_at: float,
) -> MatchResult:
    total = len(constraint_results)
    passed = [r for r in constraint_results if r.satisfied]
    failed = [r for r in constraint_results if not r.satisfied]

    constraint_score = len(passed) / total if total > 0 else 0.5  # no constraints → neutral
    composite = SEMANTIC_WEIGHT * semantic_score + CONSTRAINT_WEIGHT * constraint_score

    confidence, status = _assign_tier(composite, failed)

    top_factors: list[str] = []
    if semantic_score >= 0.6:
        top_factors.append(f"strong semantic match (score={semantic_score:.2f})")
    top_factors.extend(r.criterion for r in passed[:3])

    disqualifying = [r.reason for r in failed if _is_hard_constraint(r)]
    disqualifying += [r.reason for r in failed if not _is_hard_constraint(r)]

    explanation = MatchExplanation(
        semantic_score=round(semantic_score, 4),
        constraint_score=round(constraint_score, 4),
        composite_score=round(composite, 4),
        passed_constraints=passed,
        failed_constraints=failed,
        top_matching_factors=top_factors[:5],
        disqualifying_factors=disqualifying[:5],
    )

    return MatchResult(
        match_id=match_id,
        patient_id=patient_id,
        nct_id=nct_id,
        trial_title=trial_title,
        composite_score=round(composite, 4),
        confidence=confidence,
        explanation=explanation,
        status=status,
        created_at=created_at,
    )
