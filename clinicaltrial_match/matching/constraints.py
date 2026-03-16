"""
Rule-based eligibility constraint satisfaction.

ConstraintEvaluator.evaluate() is a pure function — no I/O, no side effects.
Diagnosis semantic equivalence checks (borderline fuzzy matches) are delegated
to the engine layer which has access to the Claude client + cache.
"""
from __future__ import annotations

import difflib
from typing import Callable

from clinicaltrial_match.matching.models import ConstraintResult
from clinicaltrial_match.patients.models import PatientFeatures
from clinicaltrial_match.trials.models import EligibilityCriteria, LabConstraint


def _fuzzy_match(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _diagnoses_match(patient_names: list[str], patient_codes: list[str], target: str) -> bool:
    target_lower = target.lower()
    # Exact code match
    for code in patient_codes:
        if code and code.lower() == target_lower:
            return True
    # Exact name match
    for name in patient_names:
        if name.lower() == target_lower:
            return True
    # Fuzzy name match (threshold 0.8)
    for name in patient_names:
        if _fuzzy_match(name, target) >= 0.8:
            return True
    return False


def _med_match(patient_meds: list[str], target: str) -> bool:
    target_n = target.lower().strip()
    for med in patient_meds:
        if med.lower().strip() == target_n:
            return True
        if _fuzzy_match(med, target) >= 0.85:
            return True
    return False


def _eval_lab(constraint: LabConstraint, patient_value: float) -> bool:
    op = constraint.operator
    v = constraint.value
    v2 = constraint.value_upper
    if op == "<":
        return patient_value < v
    if op == "<=":
        return patient_value <= v
    if op == ">":
        return patient_value > v
    if op == ">=":
        return patient_value >= v
    if op == "=":
        return abs(patient_value - v) < 1e-6
    if op == "between" and v2 is not None:
        return v <= patient_value <= v2
    return False


class ConstraintEvaluator:
    """
    Pure constraint evaluator.

    For borderline diagnosis matches (0.6-0.8 fuzzy), a callback is provided
    that the engine will invoke using Claude (with caching). These are marked
    with satisfied=None semantics by using a special "borderline" reason.
    """

    def evaluate(
        self,
        criteria: EligibilityCriteria,
        features: PatientFeatures,
    ) -> tuple[list[ConstraintResult], list[str]]:
        """
        Returns (constraint_results, borderline_diagnosis_pairs).

        borderline_diagnosis_pairs are (patient_diagnosis_name, required_or_excluded_condition)
        tuples that need semantic equivalence check via Claude.
        """
        results: list[ConstraintResult] = []
        borderline: list[str] = []

        patient_diag_names = [d.name for d in features.diagnoses]
        patient_diag_codes = [d.code for d in features.diagnoses]
        patient_active_meds = [m.name for m in features.medications if m.active]
        patient_lab_map = {lv.test_name.lower(): lv.value for lv in features.lab_values}

        # Age constraint
        if criteria.age:
            age = features.age_years
            if age is not None:
                if criteria.age.minimum_age_years is not None:
                    ok = age >= criteria.age.minimum_age_years
                    results.append(ConstraintResult(
                        criterion=f"age >= {criteria.age.minimum_age_years} years",
                        satisfied=ok,
                        patient_value=f"age: {age:.1f} years",
                        reason=f"{age:.1f} >= {criteria.age.minimum_age_years}: {'PASS' if ok else 'FAIL'}",
                    ))
                if criteria.age.maximum_age_years is not None:
                    ok = age <= criteria.age.maximum_age_years
                    results.append(ConstraintResult(
                        criterion=f"age <= {criteria.age.maximum_age_years} years",
                        satisfied=ok,
                        patient_value=f"age: {age:.1f} years",
                        reason=f"{age:.1f} <= {criteria.age.maximum_age_years}: {'PASS' if ok else 'FAIL'}",
                    ))

        # Gender constraint
        if criteria.gender and criteria.gender.allowed != "ALL":
            patient_gender = features.gender.value.upper()
            allowed = criteria.gender.allowed
            ok = patient_gender == allowed or patient_gender == "OTHER"
            results.append(ConstraintResult(
                criterion=f"gender: {allowed}",
                satisfied=ok,
                patient_value=f"gender: {features.gender.value}",
                reason=f"Patient gender {features.gender.value} {'matches' if ok else 'does not match'} required {allowed}",
            ))

        # Required diagnoses
        if criteria.diagnoses and criteria.diagnoses.required_conditions:
            for cond in criteria.diagnoses.required_conditions:
                if _diagnoses_match(patient_diag_names, patient_diag_codes, cond):
                    results.append(ConstraintResult(
                        criterion=f"has diagnosis: {cond}",
                        satisfied=True,
                        patient_value=", ".join(patient_diag_names[:5]) or "none",
                        reason=f"Patient has condition matching '{cond}'",
                    ))
                else:
                    # Check borderline
                    best_score = max(
                        (_fuzzy_match(n, cond) for n in patient_diag_names),
                        default=0.0,
                    )
                    if best_score >= 0.6:
                        borderline.append(f"required:{cond}")
                        results.append(ConstraintResult(
                            criterion=f"has diagnosis: {cond}",
                            satisfied=False,  # pending semantic check
                            patient_value=", ".join(patient_diag_names[:5]) or "none",
                            reason=f"Borderline match (score={best_score:.2f}) — semantic check needed",
                        ))
                    else:
                        results.append(ConstraintResult(
                            criterion=f"has diagnosis: {cond}",
                            satisfied=False,
                            patient_value=", ".join(patient_diag_names[:5]) or "none",
                            reason=f"No matching condition found for '{cond}'",
                        ))

        # Excluded diagnoses
        if criteria.diagnoses and criteria.diagnoses.excluded_conditions:
            for excl in criteria.diagnoses.excluded_conditions:
                if _diagnoses_match(patient_diag_names, patient_diag_codes, excl):
                    results.append(ConstraintResult(
                        criterion=f"no diagnosis: {excl}",
                        satisfied=False,
                        patient_value=", ".join(patient_diag_names[:5]) or "none",
                        reason=f"Patient has excluded condition: {excl}",
                    ))
                else:
                    results.append(ConstraintResult(
                        criterion=f"no diagnosis: {excl}",
                        satisfied=True,
                        patient_value=", ".join(patient_diag_names[:5]) or "none",
                        reason=f"Patient does not have excluded condition: {excl}",
                    ))

        # Lab constraints
        for lab in criteria.labs:
            lab_key = lab.test_name.lower()
            # Fuzzy lab name match
            matching_key = next(
                (k for k in patient_lab_map if _fuzzy_match(k, lab_key) >= 0.8),
                None,
            )
            if matching_key is not None:
                val = patient_lab_map[matching_key]
                ok = _eval_lab(lab, val)
                results.append(ConstraintResult(
                    criterion=f"{lab.test_name} {lab.operator} {lab.value} {lab.unit}",
                    satisfied=ok,
                    patient_value=f"{lab.test_name}: {val} {lab.unit}",
                    reason=f"{val} {lab.operator} {lab.value}: {'PASS' if ok else 'FAIL'}",
                ))
            # If lab not in patient record, skip (not enough info to disqualify)

        # Required medications
        if criteria.medications and criteria.medications.required:
            for med in criteria.medications.required:
                ok = _med_match(patient_active_meds, med)
                results.append(ConstraintResult(
                    criterion=f"on medication: {med}",
                    satisfied=ok,
                    patient_value=", ".join(patient_active_meds[:5]) or "none",
                    reason=f"{'Found' if ok else 'Not found'} required medication: {med}",
                ))

        # Excluded medications
        if criteria.medications and criteria.medications.excluded:
            for med in criteria.medications.excluded:
                ok = not _med_match(patient_active_meds, med)
                results.append(ConstraintResult(
                    criterion=f"not on medication: {med}",
                    satisfied=ok,
                    patient_value=", ".join(patient_active_meds[:5]) or "none",
                    reason=f"{'Does not have' if ok else 'Has'} excluded medication: {med}",
                ))

        return results, borderline
