"""Tests for FHIR R4 bundle parser — pure function, no mocks needed."""

from __future__ import annotations

from clinicaltrial_match.patients.fhir_parser import parse_fhir_bundle
from clinicaltrial_match.patients.models import Gender
from tests.fixtures.sample_fhir import SAMPLE_FHIR_BUNDLE


def test_parse_patient_id():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.patient_id == "patient-001"


def test_parse_age():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    assert patient.features.age_years is not None
    # Born 1975, so ~49-50 years old in 2025
    assert 45 <= patient.features.age_years <= 55


def test_parse_gender():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    assert patient.features.gender == Gender.MALE


def test_parse_diagnoses():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    diagnoses = patient.features.diagnoses
    assert len(diagnoses) >= 1
    names = [d.name.lower() for d in diagnoses]
    assert any("diabetes" in n for n in names)


def test_parse_lab_values():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    labs = patient.features.lab_values
    assert len(labs) >= 1
    hba1c = next((l for l in labs if "hba1c" in l.test_name.lower()), None)
    assert hba1c is not None
    assert hba1c.value == 8.2


def test_parse_medications():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    meds = patient.features.medications
    assert len(meds) >= 1
    assert any("metformin" in m.name.lower() for m in meds)


def test_extraction_confidence_is_1():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    assert patient.features.extraction_confidence == 1.0


def test_extraction_method_is_fhir():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    assert patient.features.extraction_method == "fhir"


def test_clinical_summary_populated():
    patient = parse_fhir_bundle(SAMPLE_FHIR_BUNDLE)
    assert patient.features is not None
    assert len(patient.features.clinical_summary) > 10


def test_single_patient_resource():
    """Should also accept a single Patient resource, not just Bundle."""
    single = {
        "resourceType": "Patient",
        "id": "pt-solo",
        "birthDate": "1990-01-01",
        "gender": "female",
    }
    patient = parse_fhir_bundle(single)
    assert patient.patient_id == "pt-solo"
    assert patient.features is not None
    assert patient.features.gender == Gender.FEMALE


def test_missing_patient_generates_id():
    """Bundle without Patient resource still produces a valid patient_id."""
    bundle = {
        "resourceType": "Bundle",
        "entry": [
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Hypertension"},
                    "clinicalStatus": {"text": "Active"},
                }
            }
        ],
    }
    patient = parse_fhir_bundle(bundle)
    assert patient.patient_id  # generated hash
    assert len(patient.patient_id) > 0
