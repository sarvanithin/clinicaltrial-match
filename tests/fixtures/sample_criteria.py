"""Synthetic parsed EligibilityCriteria fixture."""
from __future__ import annotations

from clinicaltrial_match.trials.models import (
    AgeConstraint,
    DiagnosisConstraint,
    EligibilityCriteria,
    GenderConstraint,
    LabConstraint,
    MedicationConstraint,
)


def make_sample_criteria() -> EligibilityCriteria:
    return EligibilityCriteria(
        raw_inclusion_text=(
            "Adults 18 years or older\n"
            "Confirmed diagnosis of type 2 diabetes mellitus\n"
            "HbA1c between 7.0 and 10.0%"
        ),
        raw_exclusion_text=(
            "Renal impairment (eGFR < 30 mL/min)\n"
            "Current use of insulin\n"
            "Pregnant or breastfeeding"
        ),
        age=AgeConstraint(minimum_age_years=18, maximum_age_years=75),
        gender=GenderConstraint(allowed="ALL"),
        diagnoses=DiagnosisConstraint(
            required_conditions=["Type 2 diabetes mellitus"],
            excluded_conditions=[],
        ),
        labs=[
            LabConstraint(test_name="HbA1c", operator="between", value=7.0, value_upper=10.0, unit="%"),
        ],
        medications=MedicationConstraint(
            required=[],
            excluded=["insulin"],
        ),
        parse_confidence=0.9,
        parsed_by="claude-haiku",
    )
