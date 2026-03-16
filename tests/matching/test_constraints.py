"""Tests for the constraint evaluator — pure function, no mocks."""
from __future__ import annotations

from clinicaltrial_match.matching.constraints import ConstraintEvaluator
from clinicaltrial_match.patients.models import Diagnosis, Gender, LabValue, Medication, PatientFeatures
from clinicaltrial_match.trials.models import (
    AgeConstraint,
    DiagnosisConstraint,
    EligibilityCriteria,
    GenderConstraint,
    LabConstraint,
    MedicationConstraint,
)


def _make_features(**kwargs) -> PatientFeatures:
    defaults = dict(
        patient_id="p-001",
        age_years=50.0,
        gender=Gender.MALE,
        diagnoses=[Diagnosis(name="Type 2 diabetes mellitus", code="E11")],
        lab_values=[LabValue(test_name="HbA1c", value=8.2, unit="%")],
        medications=[Medication(name="Metformin", active=True)],
        clinical_summary="50-year-old male with T2DM",
    )
    defaults.update(kwargs)
    return PatientFeatures(**defaults)


def _make_criteria(**kwargs) -> EligibilityCriteria:
    return EligibilityCriteria(**kwargs)


evaluator = ConstraintEvaluator()


class TestAgeConstraint:
    def test_passes_when_within_range(self):
        criteria = _make_criteria(age=AgeConstraint(minimum_age_years=18, maximum_age_years=75))
        features = _make_features(age_years=50)
        results, _ = evaluator.evaluate(criteria, features)
        age_results = [r for r in results if "age" in r.criterion]
        assert all(r.satisfied for r in age_results)

    def test_fails_below_minimum(self):
        criteria = _make_criteria(age=AgeConstraint(minimum_age_years=18))
        features = _make_features(age_years=15)
        results, _ = evaluator.evaluate(criteria, features)
        age_result = next(r for r in results if ">= 18" in r.criterion)
        assert not age_result.satisfied

    def test_fails_above_maximum(self):
        criteria = _make_criteria(age=AgeConstraint(maximum_age_years=65))
        features = _make_features(age_years=70)
        results, _ = evaluator.evaluate(criteria, features)
        age_result = next(r for r in results if "<= 65" in r.criterion)
        assert not age_result.satisfied

    def test_skipped_when_age_unknown(self):
        criteria = _make_criteria(age=AgeConstraint(minimum_age_years=18))
        features = _make_features(age_years=None)
        results, _ = evaluator.evaluate(criteria, features)
        assert len(results) == 0  # No age to evaluate


class TestGenderConstraint:
    def test_passes_when_all(self):
        criteria = _make_criteria(gender=GenderConstraint(allowed="ALL"))
        features = _make_features(gender=Gender.MALE)
        results, _ = evaluator.evaluate(criteria, features)
        gender_results = [r for r in results if "gender" in r.criterion]
        assert len(gender_results) == 0  # ALL → no constraint added

    def test_fails_wrong_gender(self):
        criteria = _make_criteria(gender=GenderConstraint(allowed="FEMALE"))
        features = _make_features(gender=Gender.MALE)
        results, _ = evaluator.evaluate(criteria, features)
        g = next(r for r in results if "gender" in r.criterion)
        assert not g.satisfied

    def test_passes_correct_gender(self):
        criteria = _make_criteria(gender=GenderConstraint(allowed="MALE"))
        features = _make_features(gender=Gender.MALE)
        results, _ = evaluator.evaluate(criteria, features)
        g = next(r for r in results if "gender" in r.criterion)
        assert g.satisfied


class TestDiagnosisConstraint:
    def test_required_diagnosis_found(self):
        criteria = _make_criteria(
            diagnoses=DiagnosisConstraint(required_conditions=["Type 2 diabetes mellitus"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "Type 2 diabetes mellitus" in r.criterion)
        assert r.satisfied

    def test_required_diagnosis_missing(self):
        criteria = _make_criteria(
            diagnoses=DiagnosisConstraint(required_conditions=["Heart failure"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "Heart failure" in r.criterion)
        assert not r.satisfied

    def test_excluded_diagnosis_found_fails(self):
        criteria = _make_criteria(
            diagnoses=DiagnosisConstraint(excluded_conditions=["Type 2 diabetes mellitus"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "no diagnosis" in r.criterion)
        assert not r.satisfied

    def test_excluded_diagnosis_absent_passes(self):
        criteria = _make_criteria(
            diagnoses=DiagnosisConstraint(excluded_conditions=["Cancer"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "no diagnosis" in r.criterion)
        assert r.satisfied

    def test_icd10_code_exact_match(self):
        criteria = _make_criteria(
            diagnoses=DiagnosisConstraint(required_conditions=["E11"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "E11" in r.criterion)
        assert r.satisfied

    def test_borderline_match_flagged(self):
        criteria = _make_criteria(
            diagnoses=DiagnosisConstraint(required_conditions=["Type 2 diabetes"])
        )
        _, borderline = evaluator.evaluate(criteria, _make_features())
        assert len(borderline) > 0


class TestLabConstraint:
    def test_lab_within_range(self):
        criteria = _make_criteria(
            labs=[LabConstraint(test_name="HbA1c", operator="between", value=7.0, value_upper=10.0, unit="%")]
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "HbA1c" in r.criterion)
        assert r.satisfied  # 8.2 is between 7 and 10

    def test_lab_outside_range(self):
        criteria = _make_criteria(
            labs=[LabConstraint(test_name="HbA1c", operator="<=", value=7.0, unit="%")]
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "HbA1c" in r.criterion)
        assert not r.satisfied  # 8.2 > 7.0

    def test_missing_lab_skipped(self):
        criteria = _make_criteria(
            labs=[LabConstraint(test_name="eGFR", operator=">=", value=30.0, unit="mL/min")]
        )
        features = _make_features(lab_values=[])
        results, _ = evaluator.evaluate(criteria, features)
        lab_results = [r for r in results if "eGFR" in r.criterion]
        assert len(lab_results) == 0  # skipped — not enough info


class TestMedicationConstraint:
    def test_required_medication_found(self):
        criteria = _make_criteria(
            medications=MedicationConstraint(required=["Metformin"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "on medication" in r.criterion)
        assert r.satisfied

    def test_excluded_medication_found_fails(self):
        criteria = _make_criteria(
            medications=MedicationConstraint(excluded=["Metformin"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "not on medication" in r.criterion)
        assert not r.satisfied

    def test_excluded_medication_absent_passes(self):
        criteria = _make_criteria(
            medications=MedicationConstraint(excluded=["Insulin"])
        )
        results, _ = evaluator.evaluate(criteria, _make_features())
        r = next(r for r in results if "not on medication" in r.criterion)
        assert r.satisfied


def test_empty_criteria_returns_no_results():
    criteria = _make_criteria()
    results, borderline = evaluator.evaluate(criteria, _make_features())
    assert results == []
    assert borderline == []
