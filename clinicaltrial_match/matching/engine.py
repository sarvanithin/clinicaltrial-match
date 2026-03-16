"""
Matching engine: orchestrates semantic search + constraint evaluation + ranking.

For each patient, the engine:
1. Runs semantic search to retrieve top-K candidate trials
2. For each candidate, evaluates eligibility constraints
3. Handles borderline diagnosis checks via Claude (with DB cache)
4. Ranks and returns top-N results
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import structlog

from clinicaltrial_match.infrastructure.claude_client import ClaudeClient
from clinicaltrial_match.infrastructure.db import Database
from clinicaltrial_match.matching.constraints import ConstraintEvaluator
from clinicaltrial_match.matching.models import MatchRequest, MatchResult
from clinicaltrial_match.matching.ranker import build_match_result
from clinicaltrial_match.matching.semantic import SemanticSearcher
from clinicaltrial_match.patients.models import PatientFeatures
from clinicaltrial_match.trials.models import EligibilityCriteria, Trial
from clinicaltrial_match.trials.repository import TrialRepository

logger = structlog.get_logger()

_DIAG_EQUIV_SYSTEM = (
    "You are a clinical terminologist. Determine if two medical conditions refer to the same "
    "or clinically equivalent condition. Answer only 'yes' or 'no'."
)


class MatchingEngine:
    def __init__(
        self,
        db: Database,
        trial_repo: TrialRepository,
        searcher: SemanticSearcher,
        claude: ClaudeClient,
    ) -> None:
        self._db = db
        self._trial_repo = trial_repo
        self._searcher = searcher
        self._claude = claude
        self._evaluator = ConstraintEvaluator()

    async def match(self, features: PatientFeatures, request: MatchRequest) -> list[MatchResult]:
        candidate_k = request.max_results * 3
        candidates = self._searcher.search(features, top_k=candidate_k)

        if not candidates:
            return []

        results: list[MatchResult] = []

        for nct_id, semantic_score in candidates:
            trial = self._trial_repo.get(nct_id)
            if not trial:
                continue

            # Apply status filter
            if request.trial_status_filter and trial.status.value not in request.trial_status_filter:
                continue

            # Apply condition filter
            if request.condition_filter:
                cond_match = any(fc.lower() in c.lower() for fc in request.condition_filter for c in trial.conditions)
                if not cond_match:
                    continue

            constraint_results, borderline = self._evaluate_constraints(trial, features)
            constraint_results = await self._resolve_borderline(
                constraint_results, borderline, features, trial.eligibility_criteria
            )

            match_id = hashlib.md5(f"{features.patient_id}:{nct_id}:{time.time()}".encode()).hexdigest()[:16]
            result = build_match_result(
                match_id=match_id,
                patient_id=features.patient_id,
                nct_id=nct_id,
                trial_title=trial.title,
                semantic_score=semantic_score,
                constraint_results=constraint_results,
                created_at=time.time(),
            )

            if result.composite_score >= request.min_score:
                results.append(result)
                # Persist
                self._db.insert_match_result(
                    {
                        "match_id": match_id,
                        "patient_id": features.patient_id,
                        "nct_id": nct_id,
                        "composite_score": result.composite_score,
                        "confidence": result.confidence,
                        "explanation": result.explanation.model_dump_json(),
                        "status": result.status,
                        "created_at": result.created_at,
                    }
                )

        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results[: request.max_results]

    def _evaluate_constraints(
        self,
        trial: Trial,
        features: PatientFeatures,
    ) -> tuple[list[Any], list[str]]:
        if not trial.eligibility_criteria:
            return [], []
        return self._evaluator.evaluate(trial.eligibility_criteria, features)

    async def _resolve_borderline(
        self,
        constraint_results: list[Any],
        borderline: list[str],
        features: PatientFeatures,
        criteria: EligibilityCriteria | None,
    ) -> list[Any]:
        if not borderline or criteria is None:
            return constraint_results

        patient_diag_names = [d.name for d in features.diagnoses]

        for item in borderline:
            is_required = item.startswith("required:")
            condition = item.split(":", 1)[1] if ":" in item else item

            # Check all patient diagnoses against this condition
            for patient_diag in patient_diag_names:
                cache_key = hashlib.md5(f"{patient_diag.lower()}|{condition.lower()}".encode()).hexdigest()
                cached = self._db.get_diagnosis_equiv(cache_key)
                if cached is not None:
                    equiv = cached
                else:
                    try:
                        answer = await self._claude.complete(
                            model=self._claude.reasoning_model,
                            system=_DIAG_EQUIV_SYSTEM,
                            user_message=f"Are '{patient_diag}' and '{condition}' the same or equivalent condition?",
                        )
                        equiv = answer.strip().lower().startswith("yes")
                        self._db.set_diagnosis_equiv(cache_key, equiv)
                    except Exception as exc:
                        logger.warning("diagnosis_equiv_check_failed", error=str(exc))
                        equiv = False

                if equiv:
                    # Update the constraint result
                    for r in constraint_results:
                        if condition in r.criterion and "Borderline" in r.reason:
                            object.__setattr__(r, "satisfied", is_required)
                            object.__setattr__(r, "reason", f"Semantic equivalence confirmed: '{patient_diag}' ≈ '{condition}'")
                    break

        return constraint_results
